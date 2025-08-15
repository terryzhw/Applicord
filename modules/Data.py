import os
import pickle
from dotenv import load_dotenv
import gspread
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError


load_dotenv()

class DataToSheet:
    def __init__(self):
        try: 
            SCOPES = [
                "https://www.googleapis.com/auth/spreadsheets",
                'https://www.googleapis.com/auth/drive.metadata.readonly'
            ]
            TOKEN_PATH = 'sheets_token.pickle'
            creds = None

            if os.path.exists(TOKEN_PATH):
                with open(TOKEN_PATH, 'rb') as token:
                    creds = pickle.load(token)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        os.getenv('CREDENTIALS'), SCOPES)
                    creds = flow.run_local_server(port=0)
                with open(TOKEN_PATH, 'wb') as token:
                    pickle.dump(creds, token)

            self.gc = gspread.authorize(creds)
            try:
                ss = self.gc.open_by_key(os.getenv('SPREADSHEET_ID'))
                self.ws = ss.worksheet(os.getenv('WORKSHEET_NAME'))
            except gspread.SpreadsheetNotFound:
                print("Error: Can't find spreadsheet ID")
                return
            except gspread.WorksheetNotFound:
                print("Error: Can't find worksheet name")
                return
        except HttpError as error: 
            print(f"Error opening sheet: {error}")
            return error

    
    def addData(self, company, position, date, status):

        try: 
            all_val = self.ws.get_all_values()
            next_row = len(all_val) + 1

            cell_range = f"A{next_row}:D{next_row}"
            self.ws.update(cell_range, 
                    [[company, position, date, status]],
                    value_input_option="USER_ENTERED")
        except HttpError as error:
            print(f"Error occurred: {error}")
            return error
    
    def getCompanyNames(self):
        try:
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
        except HttpError as error:
            print(f"Error reading spreadsheet: {error}")
            return []
        




    