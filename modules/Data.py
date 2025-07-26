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
            TOKEN_PATH = 'token.pickle'

            creds  = InstalledAppFlow.from_client_secrets_file(
                os.getenv('CREDENTIALS'),
                SCOPES
            ).run_local_server(port=0)

            self.gc = gspread.authorize(creds)
            try:
                ss = self.gc.open_by_key(os.getenv('SPREADSHEET_ID'))
                self.ws = ss.worksheet(os.getenv('WORKSHEET_NAME'))
            except gspread.SpreadsheetNotFound:
                print("Error: Can't find spreadsheet ID")
                return
            except gspread.WorksheetNotFound:
                print("Error: Can't find Worksheet Name")
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
            print(f"Wrote to {cell_range}")
        except HttpError as error:
            print(f"Error occurred: {error}")
            return error
        




    