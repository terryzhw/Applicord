
import os
from pathlib import Path
from dotenv import load_dotenv
import gspread

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.credential_manager import GoogleCredentialManager


load_dotenv()

SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    'https://www.googleapis.com/auth/drive.metadata.readonly'
]
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')
WORKSHEET_NAME = os.getenv('WORKSHEET_NAME')
TOKEN = "../sheets_token.pickle"
CREDENTIALS_PATH = os.getenv('CREDENTIALS', '../credentials.json')

class SheetManager:
    HEADER_TERMS = {
        'company', 'position', 'date', 'status', 'submitted', 'rejected', 
        'in progress', 'key', 'applicord', 'notes', 'application'
    }
    
    def __init__(self):
        credential_manager = GoogleCredentialManager(
            TOKEN, CREDENTIALS_PATH, SHEETS_SCOPES
        )
        creds = credential_manager.get_credentials()
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
        
        
        for row in all_val[1:] if len(all_val) > 1 else all_val:
            if len(row) > 0 and row[0]:
                company_name = row[0].strip()
                if company_name and company_name.lower() not in self.HEADER_TERMS:
                    companies.append(company_name)
        
        unique_companies = []
        seen = set()
        for company in companies:
            if company not in seen:
                unique_companies.append(company)
                seen.add(company)
        
        return unique_companies
    
    def getCompanyApplicationDate(self, company_name):
        all_val = self.ws.get_all_values()
        
        for row in all_val:
            if len(row) >= 3 and row[0].strip().lower() == company_name.lower():
                return row[2].strip() if row[2] else None
        
        return None
    
    def getCompaniesWithDates(self):
        from dateutil import parser
        
        all_val = self.ws.get_all_values()
        companies_with_dates = []
        
        
        for row in all_val[1:] if len(all_val) > 1 else all_val:
            if len(row) >= 3 and row[0]:
                company_name = row[0].strip()
                application_date = row[2].strip() if row[2] else None
                
                if company_name and company_name.lower() not in self.HEADER_TERMS:
                    companies_with_dates.append((company_name, application_date))
        
        def sort_key(entry):
            _, date = entry
            try:
                if date:
                    return parser.parse(date)
                # Sort entries without dates to the beginning for easier processing
                return parser.parse('1900-01-01')
            except:
                # Handle invalid date formats gracefully
                return parser.parse('1900-01-01')  
        
        companies_with_dates.sort(key=sort_key)
        return companies_with_dates
    
    def updateCompanyStatus(self, company_name, new_status):
        all_val = self.ws.get_all_values()
        
        for i, row in enumerate(all_val):
            if len(row) > 0 and row[0].strip().lower() == company_name.lower():
                row_number = i + 1
                self.ws.update(f"D{row_number}", [[new_status]])
                
        




    