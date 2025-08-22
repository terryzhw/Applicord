
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from googleapiclient.discovery import build
import base64
import re

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from ml.email_classifier import EmailClassifier
from sheet.sheet_manager import SheetManager
from utils.credential_manager import GoogleCredentialManager
from utils.date import DateRangeUtils


GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
GMAIL_TOKEN_FILE = '../gmail_token.pickle'
GMAIL_CREDENTIALS_FILE = '../credentials.json'

class CompanySearcher:
    
    def __init__(self):
        self.service = None
        self.classifier = None
        self.data_sheet = None
        self.setup_gmail_service()
        self.setup_classifier()
        self.setup_data_sheet()
    
    def setup_gmail_service(self) -> None:
        credential_manager = GoogleCredentialManager(
            GMAIL_TOKEN_FILE, GMAIL_CREDENTIALS_FILE, GMAIL_SCOPES
        )
        creds = credential_manager.get_credentials()
        self.service = build('gmail', 'v1', credentials=creds)
    
    def setup_classifier(self):
        self.classifier = EmailClassifier(model_name='roberta-base')
        model_path = '../model'
        
        # Only load the model if it exists, otherwise searches will just return all emails
        if self.is_model_available(model_path):
            self.classifier.load_model(model_path)
        else:
            print("Error: model not found")
            
    
    def is_model_available(self, model_path):
        if not os.path.exists(model_path):
            return False
        if os.path.isfile(f"{model_path}/config.pkl"):
            return True
    
    def setup_data_sheet(self):
        self.data_sheet = SheetManager()
        # Gracefully handle spreadsheet connection failures
        if not hasattr(self.data_sheet, 'ws') or self.data_sheet.ws is None:
            print("Warning: Could not connect to spreadsheet. Status updates will be disabled.")
            self.data_sheet = None
    
    def search_emails_by_company(self, company_name: str, max_results: Optional[int] = None, application_date: Optional[str] = None) -> List[Dict[str, Any]]:
        print("-" * 50)
        
        email_messages = self.fetch_email_messages(company_name, max_results, application_date)
        
        if not email_messages:
            print("No emails found")
            return []
        
        print(f"Found {len(email_messages)} emails")
        print("-" * 50)
        
        return self.process_email_messages(email_messages, company_name, application_date)
    
    def fetch_email_messages(self, company_name, max_results, application_date=None):
        search_query = self.build_search_query(company_name, application_date)
        
        request_params = {
            'userId': 'me',
            'q': search_query
        }
        
        # Use pagination for comprehensive searches, limited results for quick testing
        if max_results:
            request_params['maxResults'] = max_results
            results = self.service.users().messages().list(**request_params).execute()
            return results.get('messages', [])
        
        return self.fetch_all_pages(search_query)
    
    def build_search_query(self, company_name, application_date):
        search_query = f'{company_name}'
        
        if application_date:
            date_filter = self.create_date_filter(application_date)
            if date_filter:
                search_query += f' {date_filter}'
        
        return search_query
    
    def fetch_all_pages(self, search_query):
        all_messages = []
        page_token = None
        
        while True:
            request_params = {
                'userId': 'me',
                'q': search_query
            }
            if page_token:
                request_params['pageToken'] = page_token
            
            results = self.service.users().messages().list(**request_params).execute()
            page_messages = results.get('messages', [])
            all_messages.extend(page_messages)
            
            page_token = results.get('nextPageToken')
            if not page_token:
                break
        
        return all_messages
    
    def process_email_messages(self, messages, company_name, application_date=None):
        rejection_emails = []
        rejection_count = 0
        
        for message in messages:
            email_details = self.get_email_details(message['id'])
            
            if application_date and not self.is_email_in_date_range(email_details, application_date):
                continue
            
            is_rejection_email = self.classify_email(email_details)
            
            # Only process rejection emails to reduce noise for the user
            if self.should_include_email(is_rejection_email):
                rejection_emails.append(email_details)
                rejection_count += 1
                self.display_email_summary(email_details, rejection_count, is_rejection_email)
                
                # Automatically update spreadsheet to track rejection status
                self.update_company_status_to_rejected(company_name)
        
        return rejection_emails
    
    def get_email_details(self, message_id):
        msg = self.service.users().messages().get(
            userId='me',
            id=message_id,
            format='full'
        ).execute()
        return self.extract_email_data(msg)
    
    def classify_email(self, email_data):
        if not self.classifier:
            return False
            
        # Use email body when available, fallback to snippet for better classification
        body_content = email_data['body'] if email_data['body'].strip() else email_data['snippet']
        email_content = f"{email_data['subject']} {body_content}"
        
        prediction = self.classifier.predict(email_content)
        email_data['classification'] = prediction
        return prediction['is_rejection']
    
    def should_include_email(self, is_rejection):
        return is_rejection
    
    def update_company_status_to_rejected(self, company_name):
        if self.data_sheet is None:
            return
        
        self.data_sheet.updateCompanyStatus(company_name, "Rejected")
    
    def display_email_summary(self, email_data, count, is_rejection):
        status_indicator = "Rejection" if is_rejection else "📧"
        print(f"{status_indicator} Email #{count}")
        print(f"From: {email_data['from']}")
        print(f"Subject: {email_data['subject']}")
        print(f"Date: {email_data['date']}")
        print(f"Snippet: {email_data['snippet'][:100]}...")
        
        if self.classifier and 'classification' in email_data:
            confidence = email_data['classification']['confidence']
            print(f"Confidence: {confidence:.2%}")
        
        print("-" * 50)
    
    def extract_email_data(self, message):
        headers = message['payload'].get('headers', [])
        
        email_data = {
            'id': message['id'],
            'snippet': message.get('snippet', ''),
            'from': '',
            'to': '',
            'subject': '',
            'date': '',
            'body': ''
        }
        
        self.extract_headers(headers, email_data)
        
        email_data['body'] = self.extract_body(message['payload'])
        
        return email_data
    
    def extract_headers(self, headers, email_data):
        header_mapping = {
            'from': 'from',
            'to': 'to', 
            'subject': 'subject',
            'date': 'date'
        }
        
        for header in headers:
            name = header.get('name', '').lower()
            value = header.get('value', '')
            
            if name in header_mapping:
                email_data[header_mapping[name]] = value
    
    def extract_body(self, payload):
        if 'parts' in payload:
            text_body, html_body = self.extract_from_parts(payload['parts'])
            body = text_body if text_body else html_body
        else:
            body = self.extract_simple_body(payload)
        
        return self.clean_html_if_needed(body)
    
    def create_date_filter(self, application_date):
        date_filter = DateRangeUtils.create_gmail_date_filter(application_date)
        if not date_filter:
            print(f"Warning: Could not parse application date '{application_date}'. Skipping date filter.")
        return date_filter
    
    def is_email_in_date_range(self, email_data, application_date):
        return DateRangeUtils.is_email_in_date_range(email_data['date'], application_date)
    
    def decode_base64_data(self, data):
        try:
            return base64.urlsafe_b64decode(data).decode('utf-8')
        except UnicodeDecodeError:
            return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        except Exception:
            return ""
    
    def extract_from_parts(self, parts):
        text_body = ""
        html_body = ""
        
        for part in parts:
            if 'parts' in part:
                # Recursively handle nested email parts (multipart messages)
                nested_text, nested_html = self.extract_from_parts(part['parts'])
                text_body = text_body or nested_text
                html_body = html_body or nested_html
            else:
                mime_type = part.get('mimeType', '')
                body_data = part.get('body', {}).get('data')
                
                if not body_data:
                    continue
                    
                # Prefer plain text over HTML for cleaner classification
                if mime_type == 'text/plain' and not text_body:
                    text_body = self.decode_base64_data(body_data)
                elif mime_type == 'text/html' and not html_body:
                    html_body = self.decode_base64_data(body_data)
        
        return text_body, html_body
    
    def extract_simple_body(self, payload):
        body_data = payload.get('body', {}).get('data')
        return self.decode_base64_data(body_data) if body_data else ""
    
    def clean_html_if_needed(self, body):
        # Strip HTML tags and normalize whitespace for better text classification
        if '<' in body and '>' in body:
            body = re.sub(r'<[^>]+>', '', body)
            body = ' '.join(body.split())
        return body
    
    
    def search_all_companies_from_spreadsheet(self, max_results_per_company=None):
        companies_with_dates = self.get_companies_with_dates_from_spreadsheet()
        if not companies_with_dates:
            return []
        
        companies = [entry[0] for entry in companies_with_dates]
        self.display_search_header(companies)
        all_emails = self.search_multiple_companies_with_dates(companies_with_dates, max_results_per_company)
        self.display_search_summary(all_emails)
        
        return all_emails
    
    def get_companies_from_spreadsheet(self):
        data_sheet = SheetManager()
        companies = data_sheet.getCompanyNames()
        
        if not companies:
            print("No companies found")
            return []
        
        return companies
    
    def get_companies_with_dates_from_spreadsheet(self):
        data_sheet = SheetManager()
        companies_with_dates = data_sheet.getCompaniesWithDates()
        
        if not companies_with_dates:
            print("No companies found")
            return {}
        
        return companies_with_dates
    
    def display_search_header(self, companies):
        print(f"Found {len(companies)} entries")
        print(companies)
        print("-" * 50)
    
    def search_multiple_companies(self, companies, max_results_per_company):
        all_emails = []
        
        for i, company in enumerate(companies, 1):
            print(f"\n[{i}/{len(companies)}] Searching for: {company}")
            
            emails = self.search_emails_by_company(company, max_results_per_company)
            all_emails.extend(emails)
            
            if i < len(companies):
                print(f"Completed {company}")
                print("-" * 50)
        
        return all_emails
    
    def search_multiple_companies_with_dates(self, companies_with_dates, max_results_per_company):
        all_rejection_emails = []
        total_companies = len(companies_with_dates)
        
        for company_index, (company_name, application_date) in enumerate(companies_with_dates, 1):
            print(f"\n[{company_index}/{total_companies}] Searching for: {company_name}")
            
            if application_date:
                print(f"Application date: {application_date}")
            else:
                print("No application date found - searching all emails")
            
            company_emails = self.search_emails_by_company(
                company_name, max_results_per_company, application_date
            )
            all_rejection_emails.extend(company_emails)
            
            if company_index < total_companies:
                print(f"Completed {company_name}")
                print("-" * 50)
        
        return all_rejection_emails
    
    def display_search_summary(self, all_emails):
        print(f"\n\nSUMMARY:")
        print(f"Rejection emails found: {len(all_emails)}")
    

def main():
    searcher = CompanySearcher()
    
    searcher.search_all_companies_from_spreadsheet()

if __name__ == "__main__":
    main()