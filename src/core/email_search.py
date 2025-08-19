
import os
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import base64
import re

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from ml.email_classifier import EmailClassifier
from data.data import DataToSheet


PROJECT_ROOT = Path(__file__).parent.parent.parent
GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
GMAIL_TOKEN_FILE = PROJECT_ROOT / 'token.pickle'
GMAIL_CREDENTIALS_FILE = PROJECT_ROOT / 'credentials.json'

class CompanySearcher:
    
    def __init__(self):
        self.service = None
        self.classifier = None
        self.data_sheet = None
        self.setup_gmail_service()
        self.setup_classifier()
        self.setup_data_sheet()
    
    def setup_gmail_service(self) -> None:
        creds = self.load_credentials()
        
        if not self.are_credentials_valid(creds):
            creds = self.refresh_or_create_credentials(creds)
            self.save_credentials(creds)
        
        self.service = build('gmail', 'v1', credentials=creds)
    
    def load_credentials(self):
        if os.path.exists(GMAIL_TOKEN_FILE):
            with open(GMAIL_TOKEN_FILE, 'rb') as token:
                return pickle.load(token)
        return None
    
    def are_credentials_valid(self, creds):
        return creds is not None and creds.valid
    
    def refresh_or_create_credentials(self, creds):
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CREDENTIALS_FILE, GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        return creds
    
    def save_credentials(self, creds):
        with open(GMAIL_TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
    
    def setup_classifier(self):
        self.classifier = EmailClassifier(model_name='roberta-base')
        model_path = '../../model'
        
        if self.is_model_available(model_path):
            self.classifier.load_model(model_path)
        else:
            print("Pre-trained model not found. Classification will be disabled.")
            self.classifier = None
    
    def is_model_available(self, model_path):
        return os.path.exists(model_path) and os.path.isfile(f"{model_path}/config.pkl")
    
    def setup_data_sheet(self):
        self.data_sheet = DataToSheet()
        if not hasattr(self.data_sheet, 'ws') or self.data_sheet.ws is None:
            print("Warning: Could not connect to spreadsheet. Status updates will be disabled.")
            self.data_sheet = None
    
    def search_emails_by_company(self, company_name: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        print("-" * 50)
        messages = self.fetch_email_messages(company_name, max_results)
        
        if not messages:
            print("No emails found")
            return []
        
        print(f"Found {len(messages)} emails")
        print("-" * 50)
        
        return self.process_email_messages(messages, company_name)
    
    def fetch_email_messages(self, company_name, max_results):
        query = f'{company_name}'
        
        request_params = {
            'userId': 'me',
            'q': query
        }
        
        if max_results:
            request_params['maxResults'] = max_results
            results = self.service.users().messages().list(**request_params).execute()
            return results.get('messages', [])
        
        results = self.service.users().messages().list(**request_params).execute()
        messages = results.get('messages', [])
        
        while 'nextPageToken' in results:
            page_token = results['nextPageToken']
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                pageToken=page_token
            ).execute()
            messages.extend(results.get('messages', []))
        
        return messages
    
    def process_email_messages(self, messages, company_name):
        emails_data = []
        rejection_count = 0
        
        for message in messages:
            email_data = self.get_email_details(message['id'])
            is_rejection = self.classify_email(email_data)
            
            if self.should_include_email(is_rejection):
                emails_data.append(email_data)
                rejection_count += 1
                self.display_email_summary(email_data, rejection_count, is_rejection)
                
                self.update_company_status_to_rejected(company_name)
        
        return emails_data
    
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
                nested_text, nested_html = self.extract_from_parts(part['parts'])
                text_body = text_body or nested_text
                html_body = html_body or nested_html
            else:
                mime_type = part.get('mimeType', '')
                body_data = part.get('body', {}).get('data')
                
                if not body_data:
                    continue
                    
                if mime_type == 'text/plain' and not text_body:
                    text_body = self.decode_base64_data(body_data)
                elif mime_type == 'text/html' and not html_body:
                    html_body = self.decode_base64_data(body_data)
        
        return text_body, html_body
    
    def extract_simple_body(self, payload):
        body_data = payload.get('body', {}).get('data')
        return self.decode_base64_data(body_data) if body_data else ""
    
    def clean_html_if_needed(self, body):
        if '<' in body and '>' in body:
            body = re.sub(r'<[^>]+>', '', body)
            body = ' '.join(body.split())
        return body
    
    
    def search_all_companies_from_spreadsheet(self, max_results_per_company=None):
        companies = self.get_companies_from_spreadsheet()
        if not companies:
            return []
        
        self.display_search_header(companies)
        all_emails = self.search_multiple_companies(companies, max_results_per_company)
        self.display_search_summary(all_emails)
        
        return all_emails
    
    def get_companies_from_spreadsheet(self):
        data_sheet = DataToSheet()
        companies = data_sheet.getCompanyNames()
        
        if not companies:
            print("No companies found")
            return []
        
        return companies
    
    def display_search_header(self, companies):
        print(f"Found {len(companies)} companies")
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
    
    def display_search_summary(self, all_emails):
        print(f"\n\nSUMMARY:")
        print(f"Rejection emails found: {len(all_emails)}")
    

def main():
    searcher = CompanySearcher()
    
    searcher.search_all_companies_from_spreadsheet()

if __name__ == "__main__":
    main()