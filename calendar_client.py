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
