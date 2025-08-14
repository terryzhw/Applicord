import os
import pickle
import sys
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import base64
import re
from email_classifier import EmailClassifier

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class CompanySearcher:
    def __init__(self):
        self.service = None
        self.classifier = None
        self.setup_gmail_service()
        self.setup_classifier()
    
    def setup_gmail_service(self):
        creds = None
        
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('gmail', 'v1', credentials=creds)
    
    def setup_classifier(self):
        try:
            self.classifier = EmailClassifier(model_name='roberta-base')
            model_path = './model'
            
            if os.path.exists(model_path) and os.path.isfile(f"{model_path}/config.pkl"):
                self.classifier.load_model(model_path)
            else:
                self.classifier = None
        except Exception as e:
            print(f"Error loading email classifier: {e}")
            self.classifier = None
    
    def search_emails_by_company(self, company_name, max_results=None):
        try:
            query = f'{company_name}'
            print("-" * 50)
            
            if max_results:
                results = self.service.users().messages().list(
                    userId='me',
                    q=query,
                    maxResults=max_results
                ).execute()
            else:
                results = self.service.users().messages().list(
                    userId='me',
                    q=query
                ).execute()
                
                messages = results.get('messages', [])
                
                while 'nextPageToken' in results:
                    page_token = results['nextPageToken']
                    results = self.service.users().messages().list(
                        userId='me',
                        q=query,
                        pageToken=page_token
                    ).execute()
                    messages.extend(results.get('messages', []))
                
                results['messages'] = messages
            
            messages = results.get('messages', [])
            
            if not messages:
                print(f"No emails found containing '{company_name}'")
                return []
            
            print(f"Found {len(messages)} emails containing '{company_name}'")
            
            print("-" * 50)
            
            emails_data = []
            rejection_count = 0
            
            for i, message in enumerate(messages, 1):
                try:
                    # Get full message details
                    msg = self.service.users().messages().get(
                        userId='me',
                        id=message['id'],
                        format='full'
                    ).execute()
                    
                    email_data = self.extract_email_data(msg)
                    
                    is_rejection = False
                    if self.classifier:
                        body_content = email_data['body'] if email_data['body'].strip() else email_data['snippet']
                        email_content = f"{email_data['subject']} {body_content}"
                        prediction = self.classifier.predict(email_content)
                        is_rejection = prediction['is_rejection']
                        email_data['classification'] = prediction
                        
                    
                    if not self.classifier or is_rejection:
                        emails_data.append(email_data)
                        
                        if is_rejection or not self.classifier:
                            rejection_count += 1
                            
                            status_indicator = "Rejection" if is_rejection else "📧"
                            print(f"{status_indicator} Email #{rejection_count}")
                            print(f"From: {self.highlight_keyword(email_data['from'], company_name)}")
                            print(f"Subject: {self.highlight_keyword(email_data['subject'], company_name)}")
                            print(f"Date: {email_data['date']}")
                            print(f"Snippet: {self.highlight_keyword(email_data['snippet'][:100], company_name)}...")
                            
                            if self.classifier and 'classification' in email_data:
                                confidence = email_data['classification']['confidence']
                                print(f"Rejection Confidence: {confidence:.2%}")
                            
                            print("-" * 50)
                    
                except Exception as e:
                    print(f"Error processing email {i}: {str(e)}")
                    continue
            
            return emails_data
            
        except Exception as error:
            print(f"An error occurred: {error}")
            return []
    
    def highlight_keyword(self, text, keyword):
        if not text or not keyword:
            return text
        
        return text
    
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
        
        for header in headers:
            name = header.get('name', '').lower()
            value = header.get('value', '')
            
            if name == 'from':
                email_data['from'] = value
            elif name == 'to':
                email_data['to'] = value
            elif name == 'subject':
                email_data['subject'] = value
            elif name == 'date':
                email_data['date'] = value
        
        email_data['body'] = self.extract_body(message['payload'])
        
        return email_data
    
    def extract_body(self, payload):
        body = ""
        
        def decode_data(data):
            try:
                return base64.urlsafe_b64decode(data).decode('utf-8')
            except:
                try:
                    return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                except:
                    return ""
        
        def extract_from_parts(parts):
            text_body = ""
            html_body = ""
            
            for part in parts:
                if 'parts' in part:
                    nested_text, nested_html = extract_from_parts(part['parts'])
                    if not text_body and nested_text:
                        text_body = nested_text
                    if not html_body and nested_html:
                        html_body = nested_html
                elif part['mimeType'] == 'text/plain' and part['body'].get('data'):
                    text_body = decode_data(part['body']['data'])
                elif part['mimeType'] == 'text/html' and part['body'].get('data'):
                    html_body = decode_data(part['body']['data'])
            
            return text_body, html_body
        
        if 'parts' in payload:
            text_body, html_body = extract_from_parts(payload['parts'])
            body = text_body if text_body else html_body
        else:
            if payload['body'].get('data'):
                body = decode_data(payload['body']['data'])
        
        if '<html' in body.lower() or '<body' in body.lower():
            import re
            body = re.sub(r'<[^>]+>', '', body)
            body = ' '.join(body.split())
        
        return body
    
    def search_and_display_detailed(self, company_name, max_results=None):
        emails = self.search_emails_by_company(company_name, max_results)
        
        if emails:
            for email_data in emails:
                if email_data['body']:
                    clean_body = re.sub(r'<[^>]+>', '', email_data['body'])
                    clean_body = ' '.join(clean_body.split())
                    print(clean_body)
        
        return emails

def main():
    searcher = CompanySearcher()

    if len(sys.argv) > 1:
        company_name = sys.argv[1]
        max_results = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
    else:
        try:
            company_name = input("\nEnter: ").strip()
            
            if not company_name:
                print("Company name cannot be empty")
                return
        
            max_results = None
        except EOFError:
            company_name = "test"
            max_results = 5
    
    print(f"\nSearching: {company_name}")
    
    emails = searcher.search_and_display_detailed(company_name, max_results)
    
    filter_type = "rejection emails" if searcher.classifier else "emails"
    print(f"\nFound {len(emails)} {filter_type} containing '{company_name}'")

if __name__ == "__main__":
    main()