
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
import gspread
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError


load_dotenv()


PROJECT_ROOT = Path(__file__).parent.parent.parent
SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    'https://www.googleapis.com/auth/drive.metadata.readonly'
]
SHEETS_TOKEN_FILE = PROJECT_ROOT / 'sheets_token.pickle'
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')
WORKSHEET_NAME = os.getenv('WORKSHEET_NAME')
CREDENTIALS_PATH = os.getenv('CREDENTIALS')

class DataToSheet:
    def __init__(self):
        creds = None

        if os.path.exists(SHEETS_TOKEN_FILE):
            with open(SHEETS_TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_PATH, SHEETS_SCOPES)
                creds = flow.run_local_server(port=0)
            with open(SHEETS_TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)

        self.gc = gspread.authorize(creds)
        ss = self.gc.open_by_key(SPREADSHEET_ID)
        self.ws = ss.worksheet(WORKSHEET_NAME)

    
    def addData(self, company, position, date, status):
        all_val = self.ws.get_all_values()
        next_row = len(all_val) + 1

        cell_range = f"A{next_row}:D{next_row}"
        self.ws.update(cell_range, 
                [[company, position, date, status]],
                value_input_option="USER_ENTERED")
    
    def getCompanyNames(self):
        all_val = self.ws.get_all_values()
        companies = []
        
        header_terms = {
            'company', 'position', 'date', 'status', 'submitted', 'rejected', 
            'in progress', 'key', 'applicord', 'notes', 'application'
        }
        
        for row in all_val[1:] if len(all_val) > 1 else all_val:
            if len(row) > 0 and row[0]:
                company_name = row[0].strip()
                if company_name and company_name.lower() not in header_terms:
                    companies.append(company_name)
        
        unique_companies = []
        seen = set()
        for company in companies:
            if company not in seen:
                unique_companies.append(company)
                seen.add(company)
        
        return unique_companies
    
    def updateCompanyStatus(self, company_name, new_status):
        all_val = self.ws.get_all_values()
        
        for i, row in enumerate(all_val):
            if len(row) > 0 and row[0].strip().lower() == company_name.lower():
                row_number = i + 1
                self.ws.update(f"D{row_number}", [[new_status]])
                
        




    