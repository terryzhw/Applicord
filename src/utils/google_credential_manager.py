import os
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow


class GoogleCredentialManager:
    
    def __init__(self, token_file: str, credentials_file: str, scopes: list):
        self.token_file = token_file
        self.credentials_file = credentials_file
        self.scopes = scopes
    
    def get_credentials(self):
        creds = self.load_credentials()
        
        if not self.are_credentials_valid(creds):
            creds = self.refresh_or_create_credentials(creds)
            self.save_credentials(creds)
        
        return creds
    
    def load_credentials(self):
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                return pickle.load(token)
        return None
    
    def are_credentials_valid(self, creds):
        return creds is not None and creds.valid
    
    def refresh_or_create_credentials(self, creds):
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                self.credentials_file, self.scopes)
            creds = flow.run_local_server(port=0)
        return creds
    
    def save_credentials(self, creds):
        with open(self.token_file, 'wb') as token:
            pickle.dump(creds, token)