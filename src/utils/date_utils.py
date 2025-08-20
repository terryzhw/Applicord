from datetime import datetime, timedelta
from dateutil import parser
from typing import Optional, Tuple


class DateRangeUtils:
    """Utility functions for date parsing and range calculations."""
    
    @staticmethod
    def parse_application_date(application_date: str) -> Optional[datetime]:
        """Parse application date string into datetime object."""
        try:
            return parser.parse(application_date)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def create_date_range(application_date: str, days: int = 180) -> Optional[Tuple[datetime, datetime]]:
        """Create date range from application date."""
        app_date = DateRangeUtils.parse_application_date(application_date)
        if not app_date:
            return None
        
        start_date = app_date
        end_date = app_date + timedelta(days=days)
        return start_date, end_date
    
    @staticmethod
    def create_gmail_date_filter(application_date: str, days: int = 180) -> Optional[str]:
        """Create Gmail query date filter string."""
        date_range = DateRangeUtils.create_date_range(application_date, days)
        if not date_range:
            return None
        
        start_date, end_date = date_range
        start_str = start_date.strftime('%Y/%m/%d')
        end_str = end_date.strftime('%Y/%m/%d')
        
        return f'after:{start_str} before:{end_str}'
    
    @staticmethod
    def is_email_in_date_range(email_date: str, application_date: str, days: int = 180) -> bool:
        """Check if email date falls within application date range."""
        try:
            app_date = DateRangeUtils.parse_application_date(application_date)
            parsed_email_date = parser.parse(email_date)
            
            if not app_date:
                return True  # Default to True if we can't parse application date
            
            start_date = app_date
            end_date = app_date + timedelta(days=days)
            
            return start_date <= parsed_email_date <= end_date
        except (ValueError, TypeError):
            return True  # Default to True if we can't parse dates