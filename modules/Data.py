import os
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials

load_dotenv()

class DataToSheet:
    def __init__(self):
        creds = Credentials.from_service_account_file(
            os.getenv("CREDENTIALS"),
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
        )
        client = gspread.authorize(creds)
        self.ws = client.open(os.getenv("SPREADSHEET")).worksheet(os.getenv("WORKSHEET"))

    def addData(self, company, position, date, status):
        self.ws.append_row([company, position, date, status])
    
    def run(self):
        while True:
            company = input("Company: ").strip()
            if not company:
                print("Quit")
                break
            position = input("Position: ").strip()
            date = "7/22/25"
            status = "Submitted"
            self.addData(company, position, date, status)
            print(f"Added {company!r}\n")

if __name__ == "__main__":
    DataToSheet().run()




    