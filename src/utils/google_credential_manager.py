import os
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow


class GoogleCredentialManager:
    """Centralized Google credential management for OAuth authentication."""
    
    def __init__(self, token_file: str, credentials_file: str, scopes: list):
        self.token_file = token_file
        self.credentials_file = credentials_file
        self.scopes = scopes
    
    def get_credentials(self):
        """Get valid Google credentials, refreshing or creating new ones if needed."""
        creds = self._load_credentials()
        
        if not self._are_credentials_valid(creds):
            creds = self._refresh_or_create_credentials(creds)
            self._save_credentials(creds)
        
        return creds
    
    def _load_credentials(self):
        """Load credentials from token file if it exists."""
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                return pickle.load(token)
        return None
    
    def _are_credentials_valid(self, creds):
        """Check if credentials are valid and not expired."""
        return creds is not None and creds.valid
    
    def _refresh_or_create_credentials(self, creds):
        """Refresh expired credentials or create new ones."""
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                self.credentials_file, self.scopes)
            creds = flow.run_local_server(port=0)
        return creds
    
    def _save_credentials(self, creds):
        """Save credentials to token file."""
        with open(self.token_file, 'wb') as token:
            pickle.dump(creds, token)