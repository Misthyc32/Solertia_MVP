import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from config import SCOPES, TZ

CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"

def get_calendar_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    try:
        return build("calendar", "v3", credentials=creds)
    except HttpError as e:
        raise RuntimeError(f"Calendar build failed: {e}")

def create_event(service, calendar_id, summary, description, start_iso, end_iso, tz=TZ):
    body = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_iso, "timeZone": tz},
        "end":   {"dateTime": end_iso,   "timeZone": tz},
    }
    return service.events().insert(calendarId=calendar_id, body=body).execute()

def generate_calendar_invitation_link(summary, start_iso, end_iso, name, num_people, des="", tz=TZ):
    """
    Generates a calendar invitation link that users can add to their own calendar.
    Creates links for Google Calendar, Outlook, and Apple Calendar.
    """
    import urllib.parse
    import datetime as dt
    
    # Parse the ISO datetime strings properly
    start_dt = dt.datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
    end_dt = dt.datetime.fromisoformat(end_iso.replace('Z', '+00:00'))
    
    # Convert to UTC for Google Calendar (required format)
    start_utc = start_dt.astimezone(dt.timezone.utc)
    end_utc = end_dt.astimezone(dt.timezone.utc)
    
    # Format for Google Calendar (YYYYMMDDTHHMMSSZ)
    start_google = start_utc.strftime('%Y%m%dT%H%M%SZ')
    end_google = end_utc.strftime('%Y%m%dT%H%M%SZ')
    
    # Format for ICS (YYYYMMDDTHHMMSS)
    start_ics = start_utc.strftime('%Y%m%dT%H%M%S')
    end_ics = end_utc.strftime('%Y%m%dT%H%M%S')
    dtstamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    
    # Create client-friendly description (simple and nice)
    client_summary = f"Reservación en La Casona"
    client_description = f"Tienes una reservación en La Casona para {num_people} personas"
    if des:
        client_description += f"\n\nNotas: {des}"
    
    # Google Calendar link
    google_params = {
        'action': 'TEMPLATE',
        'text': client_summary,
        'dates': f"{start_google}/{end_google}",
        'details': client_description,
        'location': 'La Casona Restaurant'
    }
    google_link = f"https://calendar.google.com/calendar/render?{urllib.parse.urlencode(google_params)}"
    
    # Outlook/Hotmail link
    outlook_params = {
        'subject': client_summary,
        'startdt': start_iso,
        'enddt': end_iso,
        'body': client_description,
        'location': 'La Casona Restaurant'
    }
    outlook_link = f"https://outlook.live.com/calendar/0/deeplink/compose?{urllib.parse.urlencode(outlook_params)}"
    
    # Apple Calendar (ICS file) - properly formatted
    ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//La Casona//Restaurant Booking//EN
BEGIN:VEVENT
UID:{name.replace(' ', '_').replace(':', '_')}@lacasona.com
DTSTAMP:{dtstamp}
DTSTART:{start_ics}
DTEND:{end_ics}
SUMMARY:{client_summary}
DESCRIPTION:{client_description}
LOCATION:La Casona Restaurant
END:VEVENT
END:VCALENDAR"""
    
    ics_data = urllib.parse.quote(ics_content)
    apple_link = f"data:text/calendar;charset=utf8,{ics_data}"
    
    return {
        "google": google_link,
        "outlook": outlook_link,
        "apple": apple_link
    }


def update_event(service, calendar_id, event_id, summary: str | None = None, description: str | None = None, start_iso: str | None = None, end_iso: str | None = None, tz: str = TZ):
    event = service.events().get(calendarId=calendar_id, eventId=event_id).execute()
    updated_event_body = {
        "summary": event.get("summary") if summary is None else summary ,
        "description": event.get("description") if description is None else description,
        "start" : {"dateTime": event.get("start", {}).get("dateTime") if start_iso is None else start_iso,
                   "timeZone":tz
                },
        "end" : {"dateTime": event.get("end", {}).get("dateTime") if end_iso is None else end_iso,
                   "timeZone":tz
                }        
    }
    try:
        updated_event = service.events().update(calendarId=calendar_id, eventId=event_id, body=updated_event_body).execute()
        print(f"Event updated: {updated_event.get('htmlLink')}")
        return updated_event

    except Exception as e:
        print(f"Tuvimos un error al actualizar tu reserva: {e}")
