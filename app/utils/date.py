from datetime import datetime, timedelta
from dateutil import parser
from typing import Optional, Tuple


def parse_application_date(application_date: str) -> Optional[datetime]:
    try:
        return parser.parse(application_date)
    except (ValueError, TypeError):
        return None


def create_date_range(application_date: str, days: int = 180) -> Optional[Tuple[datetime, datetime]]:
    app_date = parse_application_date(application_date)
    if not app_date:
        return None
    
    start_date = app_date
    end_date = app_date + timedelta(days=days)
    return start_date, end_date


def create_gmail_date_filter(application_date: str, days: int = 180) -> Optional[str]:
    date_range = create_date_range(application_date, days)
    if not date_range:
        return None
    
    start_date, end_date = date_range
    start_str = start_date.strftime('%Y/%m/%d')
    end_str = end_date.strftime('%Y/%m/%d')
    
    return f'after:{start_str} before:{end_str}'


def is_email_in_date_range(email_date: str, application_date: str, days: int = 180) -> bool:
    try:
        app_date = parse_application_date(application_date)
        parsed_email_date = parser.parse(email_date)
        
        if not app_date:
            return True  
        
        start_date = app_date
        end_date = app_date + timedelta(days=days)
        
        return start_date <= parsed_email_date <= end_date
    except (ValueError, TypeError):
        return True