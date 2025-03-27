import asyncio
import base64
import datetime
import json
import logging
import os
import pickle
import re
import signal
from http.server import BaseHTTPRequestHandler, HTTPServer
import subprocess
import sys
import urllib
import tempfile
import threading
from datetime import UTC
from dotenv import load_dotenv
from email.mime.text import MIMEText
from gnews import GNews
from google import genai
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from gradio_client import Client, handle_file
from io import BytesIO
from PIL import Image
import pytz
import python_weather
import requests
from telegram import Update
from telegram.error import Conflict, NetworkError
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, JobQueue

def kill_process_using_port(port):
    # Check if the port is already in use and kill the process
    try:
        # Use `lsof` to check the process using port
        result = subprocess.run(
            ['lsof', '-t', f'-i:{port}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        pid = result.stdout.decode('utf-8').strip()
        if pid:
            # If there is a process using the port, kill it
            subprocess.run(['kill', '-9', pid])
            print(f"Terminated process {pid} using port {port}")
    except Exception as e:
        print(f"Error checking or killing process on port {port}: {e}")

# Define global server classes
class GlobalOAuthServer(HTTPServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_to_flow = {}  # Maps state to (flow, google_services)

class GlobalOAuthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        code = params.get('code', [None])[0]
        state = params.get('state', [None])[0]

        if state not in self.server.state_to_flow:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid state parameter")
            return

        flow, google_services = self.server.state_to_flow[state]
        try:
            flow.fetch_token(code=code)
            google_services.creds = flow.credentials
            token_dir = os.path.dirname(google_services.token_path)
            if token_dir:
                os.makedirs(token_dir, exist_ok=True)
            with open(google_services.token_path, 'wb') as token:
                pickle.dump(google_services.creds, token)
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"Authorization successful. You can close this window and return to Telegram.")
            del self.server.state_to_flow[state]  # Clean up after success
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode())
            raise

oauth_server = GlobalOAuthServer(('localhost', 8080), GlobalOAuthHandler)
server_thread = threading.Thread(target=oauth_server.serve_forever)
server_thread.daemon = True
server_thread.start()

# -------------------------------
# Utility Functions & Gemini API Setup
# -------------------------------
def clean_gemini_response(response_text):
    """Remove markdown fences and extraneous text from Gemini's response."""
    cleaned = re.sub(r"^```(json)?\s*", "", response_text)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned

# Load environment variables (ensure your .env file contains gemini_api_key_1, gemini_api_key_2, etc.)
load_dotenv()
gemini_api_keys = [os.getenv(f"gemini_api_key_{i}") for i in range(1, 11)]
gemini_clients = [genai.Client(api_key=key) for key in gemini_api_keys if key]

def get_gemini_response(prompt):
    """
    Iterates through Gemini clients to generate a response.
    Returns the cleaned text response from Gemini.
    """
    for genai_client in gemini_clients:
        try:
            response = genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            if response.text:
                print("Response from Gemini:", response.text)
                return clean_gemini_response(response.text)
        except Exception as e:
            print(f"Error with a Gemini client: {e}")
            continue
    return None

# -------------------------------
# Google Services Class
# -------------------------------
class GoogleServices:
    SCOPES = [
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/contacts.readonly",
        "https://www.googleapis.com/auth/contacts",
        "https://www.googleapis.com/auth/tasks"
    ]

    def __init__(self, user_name):
        self.user_name = user_name
        self.users_dir = 'users'
        self.token_path = os.path.join(self.users_dir, f'{user_name}_token.pickle')
        self.creds = None
        if not os.path.exists(self.users_dir):
            os.makedirs(self.users_dir)

    def build_services(self):
        self.calendar_service = build('calendar', 'v3', credentials=self.creds)
        self.gmail_service = build('gmail', 'v1', credentials=self.creds)
        self.contacts_service = build('people', 'v1', credentials=self.creds)
        self.tasks_service = build('tasks', 'v1', credentials=self.creds)

    def authenticate(self, return_auth_url=False, user_id=None):
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                self.creds = pickle.load(token)
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
                with open(self.token_path, 'wb') as token:
                    pickle.dump(self.creds, token)
            else:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', self.SCOPES)
                flow.redirect_uri = os.getenv('GOOGLE_OAUTH_REDIRECT_URI', 'https://game-national-fly.ngrok-free.app/')
                if not flow.redirect_uri:
                    raise ValueError("GOOGLE_OAUTH_REDIRECT_URI not set")
                if return_auth_url:
                    auth_state = base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8')
                    auth_url, _ = flow.authorization_url(
                        prompt='consent',
                        state=auth_state,
                        access_type='offline'
                    )
                    oauth_server.state_to_flow[auth_state] = (flow, self)
                    return auth_url, flow.redirect_uri
                else:
                    # Blocking mode (optional, if needed outside bot context)
                    creds = flow.run_local_server(port=8080)
                    self.creds = creds
                    with open(self.token_path, 'wb') as token:
                        pickle.dump(self.creds, token)

        # Build services after credentials are obtained
        if self.creds:
            self.build_services()

    # ---- Calendar Functions ----
    def list_events(self):
        now = datetime.datetime.utcnow().isoformat() + 'Z'
        try:
            events = self.calendar_service.events().list(
                calendarId='primary', timeMin=now, maxResults=10,
                singleEvents=True, orderBy='startTime'
            ).execute()
            return events.get('items', [])
        except HttpError as error:
            print(f"Error listing events: {error}")
            return None

    def create_event(self, title, description, location, start_time, end_time):
        event = {
            'summary': title,
            'location': location,
            'description': description,
            'start': {'dateTime': start_time, 'timeZone': 'UTC'},
            'end': {'dateTime': end_time, 'timeZone': 'UTC'}
        }
        return self.calendar_service.events().insert(calendarId='primary', body=event).execute()

    def update_event(self, event_id, title=None, description=None, location=None, start_time=None, end_time=None):
        event = self.calendar_service.events().get(calendarId='primary', eventId=event_id).execute()
        if title:
            event['summary'] = title
        if description:
            event['description'] = description
        if location:
            event['location'] = location
        if start_time:
            event['start']['dateTime'] = start_time
        if end_time:
            event['end']['dateTime'] = end_time
        return self.calendar_service.events().update(calendarId='primary', eventId=event_id, body=event).execute()

    def delete_event(self, event_id):
        return self.calendar_service.events().delete(calendarId='primary', eventId=event_id).execute()

    # ---- Tasks Functions ----
    def list_tasks(self):
        try:
            tasks_result = self.tasks_service.tasks().list(tasklist='@default').execute()
            return tasks_result.get('items', [])
        except HttpError as error:
            print(f"Error listing tasks: {error}")
            return None

    def create_task(self, title, notes, due_date):
        task = {"title": title, "notes": notes, "due": due_date}
        return self.tasks_service.tasks().insert(tasklist='@default', body=task).execute()

    def update_task(self, task_id, title=None, notes=None, due_date=None):
        task = self.tasks_service.tasks().get(tasklist='@default', task=task_id).execute()
        if title:
            task['title'] = title
        if notes:
            task['notes'] = notes
        if due_date:
            task['due'] = due_date
        return self.tasks_service.tasks().update(tasklist='@default', task=task_id, body=task).execute()

    def delete_task(self, task_id):
        return self.tasks_service.tasks().delete(tasklist='@default', task=task_id).execute()

    # ---- Gmail Functions ----
    def send_email(self, recipient, subject, body):
        from email.mime.text import MIMEText
        message = MIMEText(body)
        message['to'] = recipient
        message['subject'] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        return self.gmail_service.users().messages().send(userId='me', body={'raw': raw}).execute()

    def search_emails(self, query, maxResults=10):
        try:
            results = self.gmail_service.users().messages().list(
                userId='me', q=query, maxResults=maxResults
            ).execute()
            messages = results.get('messages', [])
            emails = []
            for msg in messages:
                message_detail = self.gmail_service.users().messages().get(
                    userId='me', id=msg['id'], format='full'
                ).execute()
                emails.append(message_detail)
            return emails
        except HttpError as error:
            print(f"Error searching emails: {error}")
            return None

    def read_email(self, email_id):
        try:
            message_detail = self.gmail_service.users().messages().get(
                userId='me', id=email_id, format='full'
            ).execute()
            return message_detail
        except HttpError as error:
            print(f"Error reading email {email_id}: {error}")
            return None

    def list_emails(self):
        try:
            results = self.gmail_service.users().messages().list(userId='me', maxResults=10).execute()
            messages = results.get('messages', [])
            emails = []
            for msg in messages:
                message_detail = self.gmail_service.users().messages().get(userId='me', id=msg['id']).execute()
                emails.append(message_detail)
            return emails
        except HttpError as error:
            print(f"Error listing emails: {error}")
            return None

    # ---- Contacts Functions ----
    def list_contacts(self):
        try:
            results = self.contacts_service.people().connections().list(
                resourceName='people/me',
                pageSize=100,
                personFields='names,emailAddresses,phoneNumbers'
            ).execute()
            return results.get('connections', [])
        except HttpError as error:
            print(f"Error listing contacts: {error}")
            return None

    def create_contact(self, name, email, phone=None):
        contact_body = {
            "names": [{"givenName": name}],
            "emailAddresses": [{"value": email}]
        }
        if phone:
            contact_body["phoneNumbers"] = [{"value": phone}]
        return self.contacts_service.people().createContact(body=contact_body).execute()

    def update_contact(self, contact_id, name=None, email=None, phone=None):
        update_fields = []
        person = {}
        if name:
            person["names"] = [{"givenName": name}]
            update_fields.append("names")
        if email:
            person["emailAddresses"] = [{"value": email}]
            update_fields.append("emailAddresses")
        if phone:
            person["phoneNumbers"] = [{"value": phone}]
            update_fields.append("phoneNumbers")
        return self.contacts_service.people().updateContact(
            resourceName=contact_id,
            updatePersonFields=",".join(update_fields),
            body=person
        ).execute()

    def delete_contact(self, contact_id):
        return self.contacts_service.people().deleteContact(resourceName=contact_id).execute()
# -------------------------------
# Multi-Turn Conversation & Action Execution
# -------------------------------
# A mapping of actions to their required parameters (for pending clarification)
REQUIRED_PARAMS = {
    "create_event": ["date", "time", "location", "title"],
    "list_events": [],
    "update_event": ["event_id"],
    "delete_event": ["event_id"],
    "create_task": ["title", "notes", "due"],
    "list_tasks": [],
    "update_task": ["task_id"],
    "delete_task": ["task_id"],
    "send_email": ["recipient", "subject", "body"],
    "list_emails": [],
    "search_email": ["query"],
    "read_email": ["email_id"],
    "list_contacts": [],
    "create_contact": ["name", "email"],
    "update_contact": ["contact_id"],
    "delete_contact": ["contact_id"],
    # New action for complex email analysis:
    "analyze_emails": ["query"]
}

# Global variable to store pending action details (when clarification is needed)
pending_action = None

def execute_action(action_data, google_services):
    """
    Execute the action based on the provided action_data dictionary.
    """
    action = action_data.get("action")
    parameters = action_data.get("parameters", {})
    try:
        if action == "create_event":
            required = ["date", "time", "location", "title"]
            if not all(key in parameters for key in required):
                return "Missing parameters for create_event."
            dt = datetime.datetime.strptime(parameters["date"] + " " + parameters["time"], "%Y-%m-%d %H:%M")
            end_dt = dt + datetime.timedelta(hours=1)  # default duration
            start_iso = dt.isoformat() + "Z"
            end_iso = end_dt.isoformat() + "Z"
            result = google_services.create_event(
                title=parameters["title"],
                description="",
                location=parameters["location"],
                start_time=start_iso,
                end_time=end_iso
            )
            return f"Event created: {result}"
        
        elif action == "list_events":
            events = google_services.list_events()
            if not events:
                return "No upcoming events found."
            formatted_events = []
            for event in events:
                start = event.get('start', {}).get('dateTime', event.get('start', {}).get('date'))
                if start:
                    start_dt = datetime.datetime.fromisoformat(start.replace('Z', '+00:00'))
                    formatted_start = start_dt.strftime("%B %d, %Y at %I:%M %p")
                else:
                    formatted_start = "No start time"
                formatted_events.append(f"• {event.get('summary', 'Untitled Event')} - {formatted_start}")
            return "Upcoming Events:\n" + "\n".join(formatted_events)

        elif action == "update_event":
            event_id = parameters.get("event_id")
            if not event_id:
                return "Missing event_id for update_event."
            start_iso = None
            end_iso = None
            if "date" in parameters and "time" in parameters:
                dt = datetime.datetime.strptime(parameters["date"] + " " + parameters["time"], "%Y-%m-%d %H:%M")
                start_iso = dt.isoformat() + "Z"
                end_dt = dt + datetime.timedelta(hours=1)
                end_iso = end_dt.isoformat() + "Z"
            result = google_services.update_event(
                event_id,
                title=parameters.get("title"),
                description=parameters.get("description"),
                location=parameters.get("location"),
                start_time=start_iso,
                end_time=end_iso
            )
            return f"Event updated: {result}"

        elif action == "delete_event":
            event_id = parameters.get("event_id")
            if not event_id:
                return "Missing event_id for delete_event."
            result = google_services.delete_event(event_id)
            return f"Event deleted: {result}"

        elif action == "create_task":
            required = ["title", "notes", "due"]
            if not all(key in parameters for key in required):
                return "Missing parameters for create_task."
            result = google_services.create_task(parameters["title"], parameters["notes"], parameters["due"])
            return f"Task created: {result}"

        elif action == "list_tasks":
            tasks = google_services.list_tasks()
            if not tasks:
                return "No tasks found."
            formatted_tasks = []
            for task in tasks:
                due_date = task.get('due')
                if due_date:
                    due_dt = datetime.datetime.fromisoformat(due_date.replace('Z', '+00:00'))
                    formatted_due = due_dt.strftime("%B %d, %Y")
                else:
                    formatted_due = "No due date"
                formatted_tasks.append(f"• {task.get('title', 'Untitled Task')} - Due: {formatted_due}")
            return "Your Tasks:\n" + "\n".join(formatted_tasks)

        elif action == "update_task":
            task_id = parameters.get("task_id")
            if not task_id:
                return "Missing task_id for update_task."
            result = google_services.update_task(
                task_id,
                title=parameters.get("title"),
                notes=parameters.get("notes"),
                due_date=parameters.get("due")
            )
            return f"Task updated: {result}"

        elif action == "delete_task":
            task_id = parameters.get("task_id")
            if not task_id:
                return "Missing task_id for delete_task."
            result = google_services.delete_task(task_id)
            return f"Task deleted: {result}"

        elif action == "send_email":
            required = ["recipient", "subject", "body"]
            if not all(key in parameters for key in required):
                return "Missing parameters for send_email."
            result = google_services.send_email(parameters["recipient"], parameters["subject"], parameters["body"])
            return f"Email sent: {result}"

        elif action == "search_email":
            required = ["query"]
            if not all(key in parameters for key in required):
                return "Missing parameters for search_email."
            result = google_services.search_emails(parameters["query"])
            if not result:
                return "No emails found matching the query."
            formatted_emails = []
            for email in result:
                headers = {h['name']: h['value'] for h in email.get('payload', {}).get('headers', [])}
                subject = headers.get('Subject', 'No Subject')
                sender = headers.get('From', 'Unknown Sender')
                formatted_emails.append(f"• From: {sender}\n  Subject: {subject}")
            return "Search results:\n" + "\n".join(formatted_emails)

        elif action == "read_email":
            required = ["email_id"]
            if not all(key in parameters for key in required):
                return "Missing parameters for read_email."
            email = google_services.read_email(parameters["email_id"])
            if not email:
                return "Email not found or could not be read."
            headers = {h['name']: h['value'] for h in email.get('payload', {}).get('headers', [])}
            subject = headers.get('Subject', 'No Subject')
            sender = headers.get('From', 'Unknown Sender')
            snippet = email.get('snippet', '')
            return f"Email details:\nFrom: {sender}\nSubject: {subject}\nSnippet: {snippet}"

        elif action == "list_emails":
            emails = google_services.list_emails()
            if not emails:
                return "No emails found."
            formatted_emails = []
            for email in emails:
                headers = {h['name']: h['value'] for h in email.get('payload', {}).get('headers', [])}
                subject = headers.get('Subject', 'No Subject')
                sender = headers.get('From', 'Unknown Sender')
                formatted_emails.append(f"• From: {sender}\n  Subject: {subject}")
            return "Recent Emails:\n" + "\n".join(formatted_emails)

        elif action == "list_contacts":
            contacts = google_services.list_contacts()
            if not contacts:
                return "No contacts found."
            formatted_contacts = []
            for contact in contacts:
                names = contact.get('names', [{}])[0].get('displayName', 'Unnamed')
                emails = contact.get('emailAddresses', [{}])[0].get('value', 'No email')
                # Add phone number formatting
                phone = contact.get('phoneNumbers', [{}])[0].get('value', 'No phone')
                formatted_contacts.append(f"• {names} - {emails} - {phone}")
            return "Your Contacts:\n" + "\n".join(formatted_contacts)

        elif action == "create_contact":
            required = ["name", "email"]
            if not all(key in parameters for key in required):
                return "Missing parameters for create_contact."
            result = google_services.create_contact(parameters["name"], parameters["email"], parameters.get("phone"))
            return f"Contact created: {result}"

        elif action == "update_contact":
            contact_id = parameters.get("contact_id")
            if not contact_id:
                return "Missing contact_id for update_contact."
            result = google_services.update_contact(
                contact_id,
                name=parameters.get("name"),
                email=parameters.get("email"),
                phone=parameters.get("phone")
            )
            return f"Contact updated: {result}"

        elif action == "delete_contact":
            contact_id = parameters.get("contact_id")
            if not contact_id:
                return "Missing contact_id for delete_contact."
            result = google_services.delete_contact(contact_id)
            return f"Contact deleted: {result}"

        # New branch for analyzing emails based on a search query.
        elif action == "analyze_emails":
            required = ["query"]
            if not all(key in parameters for key in required):
                return "Missing parameters for analyze_emails."
            emails = google_services.search_emails(parameters["query"])
            if not emails:
                return f"No emails found for query: {parameters['query']}."
            aggregated_content = ""
            for email in emails:
                headers = {h['name']: h['value'] for h in email.get('payload', {}).get('headers', [])}
                subject = headers.get('Subject', 'No Subject')
                snippet = email.get('snippet', '')
                aggregated_content += f"Subject: {subject}\nSnippet: {snippet}\n\n"
            analysis_prompt = (
                f"Please analyze the following emails and provide insights or a summary:\n\n{aggregated_content}"
            )
            analysis_response = get_gemini_response(analysis_prompt)
            return analysis_response

        else:
            return f"Unknown action: {action}"

    except HttpError as http_err:
        return f"An error occurred: {http_err}"
    except Exception as e:
        return f"Unexpected error: {e}"

def process_user_input(user_input, google_services):
    """
    Processes user input in a multi–turn conversation.
    If a pending action exists, assume the input provides the missing parameter.
    Otherwise, use a system prompt for Gemini to decide on the intended action.
    """
    global pending_action

    # If a pending action is waiting for missing parameters:
    if pending_action is not None:
        action = pending_action.get("action")
        required = REQUIRED_PARAMS.get(action, [])
        current_params = pending_action.get("parameters", {})
        missing = [p for p in required if p not in current_params]
        if missing:
            # Assume the new input provides the first missing parameter
            current_params[missing[0]] = user_input.strip()
            pending_action["parameters"] = current_params
            pending_action.pop("clarification", None)
            missing = [p for p in required if p not in current_params]
            if missing:
                return f"Still missing: {', '.join(missing)}."
            else:
                response = execute_action(pending_action, google_services)
                pending_action = None
                return response
        else:
            response = execute_action(pending_action, google_services)
            pending_action = None
            return response

    # No pending action: Use a system prompt to interpret the new input.
    system_prompt = (
        "You are an assistant that determines if a user's input should trigger an action using Google services "
        "(e.g., managing calendar events, tasks, emails, contacts) or if the input is casual conversation.\n"
        "If the input is actionable, extract the intended action and any parameters from the user's text.\n\n"
        "**Listing Items Examples:**\n"
        '- "Show my calendar events" => {"action": "list_events"}\n'
        '- "What\'s on my todo list?" => {"action": "list_tasks"}\n\n'
        "For the following actions, please output JSON in the following structure:\n\n"
        "- create_event: parameters: date (YYYY-MM-DD), time (HH:MM in 24-hour format), location, title.\n"
        "- list_events: no parameters.\n"
        "- update_event: parameters: event_id, and optionally title, description, date, time, location.\n"
        "- delete_event: parameters: event_id.\n"
        "- create_task: parameters: title, notes, due (YYYY-MM-DD).\n"
        "- list_tasks: no parameters.\n"
        "- update_task: parameters: task_id, and optionally title, notes, due.\n"
        "- delete_task: parameters: task_id.\n"
        "- send_email: parameters: recipient, subject, body.\n"
        "- list_emails: no parameters.\n"
        "- search_email: parameters: query.\n"
        "- read_email: parameters: email_id (low periority than analyze_emails)\n"
        "- list_contacts: no parameters.\n"
        "- create_contact: parameters: name, email, phone (optional).\n"
        "- update_contact: parameters: contact_id, and optionally name, email, phone.\n"
        "- delete_contact: parameters: contact_id.\n"
        # New action added here:
        "- analyze_emails: parameters: query.(high periority than read emails)\n\n"
        "Your response should include:\n"
        "1. 'action': the intended action to perform,\n"
        "2. 'parameters': required parameters for the action,\n"
        "3. 'clarification': a follow-up question if parameters are missing,\n"
        "4. 'search_criteria': when deleting/updating, include search terms to find relevant items.\n\n"
        "If no action is needed, output a JSON object with 'action': 'none' and an appropriate response. "
        "In response field, provide your response to your user as a virtual assistant, you don't need to be concise.\n\n"
        "Example for a deletion request:\n"
        "{\n"
        '    "action": "list_tasks",\n'
        '    "parameters": {},\n'
        '    "search_criteria": "hospital"\n'
        "}\n\n"
        "Please output in JSON format.\n\n"
        "Input:\n    " + user_input
    )

    response_text = get_gemini_response(system_prompt)
    if not response_text:
        return "Failed to process input."
    try:
        action_data = json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback: parse simple key: value pairs
        action_data = {}
        for part in response_text.split(';'):
            if ':' in part:
                key, value = part.split(':', 1)
                action_data[key.strip()] = value.strip()
    
    action = action_data.get("action", "none")
    if action == "none":
        return action_data.get("response", "No action needed.")
    
    # If search criteria exists, first list matching items
    if "search_criteria" in action_data:
        search_term = action_data["search_criteria"]
        if action.startswith("delete_") or action.startswith("update_"):
            # List relevant items based on search term
            matching_items = []
            if "task" in action:
                tasks = google_services.list_tasks()
                if tasks:
                    matching_items.extend([t for t in tasks if search_term.lower() in t.get('title', '').lower()])
            elif "event" in action:
                events = google_services.list_events()
                if events:
                    matching_items.extend([e for e in events if search_term.lower() in e.get('summary', '').lower()])
            
            if matching_items:
                return f"Found matching items:\n" + "\n".join([f"- {item.get('title', item.get('summary', 'Untitled'))}" for item in matching_items])
    
    # Check if required parameters are missing
    required = REQUIRED_PARAMS.get(action, [])
    parameters = action_data.get("parameters", {})
    missing = [p for p in required if p not in parameters]
    if missing:
        pending_action = action_data
        # Return the response or clarification
        return action_data.get("clarification", f"Missing parameters: {', '.join(missing)}. Please provide them.")
    else:
        result = execute_action(action_data, google_services)
        # For list actions, prioritize the formatted results from execute_action
        if action in ["list_events", "list_tasks", "list_emails", "list_contacts"]:
            return result
        return result

user_google_services = {}

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Server Management Class
class ServerManager:
    def __init__(self):
        self.active_servers = {}
        self.port = 8080
    
    def start_server(self, user_id):
        if user_id in self.active_servers:
            self.stop_server(user_id)
        kill_process_using_port(self.port)
        self.active_servers[user_id] = {
            'start_time': datetime.datetime.now(datetime.UTC),
            'port': self.port
        }
    
    def stop_server(self, user_id):
        if user_id in self.active_servers:
            kill_process_using_port(self.active_servers[user_id]['port'])
            del self.active_servers[user_id]
    
    def cleanup_inactive_servers(self, max_age_minutes=30):
        current_time = datetime.datetime.now()
        for user_id in list(self.active_servers.keys()):
            server_start_time = self.active_servers[user_id]['start_time']
            if (current_time - server_start_time).total_seconds() > max_age_minutes * 60:
                self.stop_server(user_id)

# Initialize API tokens
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # Make sure to add this to your .env file
hf_token = os.getenv("hf_token")
gemini_api_keys = [
    os.getenv(f"gemini_api_key_{i}") for i in range(1, 11)
]

# Initialize Gemini clients
gemini_clients = [genai.Client(api_key=key) for key in gemini_api_keys]

# Initialize GNews
google_news = GNews(language='en', country='US', period='7d', max_results=5)

# Initialize global managers
user_google_services = {}
server_manager = ServerManager()


# Add this class after the imports and before the command handlers
class ChatHistoryManager:
    def __init__(self):
        self.base_dir = "chat_history"
        os.makedirs(self.base_dir, exist_ok=True)
    
    def get_history_file(self, username):
        """Get the path to the user's chat history file."""
        safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_'))
        return os.path.join(self.base_dir, f"{safe_username}.json")
    
    def load_history(self, username):
        """Load chat history for a user."""
        file_path = self.get_history_file(username)
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading chat history for {username}: {e}")
            return []
    
    def save_message(self, username, role, content):
        """Save a message to the user's chat history."""
        try:
            history = self.load_history(username)
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            history.append(message)
            
            # Keep only last 50 messages
            if len(history) > 50:
                history = history[-50:]
            
            file_path = self.get_history_file(username)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving chat history for {username}: {e}")

# Add this after the existing global variables
chat_history_manager = ChatHistoryManager()

# Function to cleanup resources on shutdown
def cleanup_resources():
    # Stop all active authorization servers
    for user_id in list(server_manager.active_servers.keys()):
        server_manager.stop_server(user_id)
    
    # Close all active Google service connections
    for service in user_google_services.values():
        if service.server_thread and service.server_thread.is_alive():
            kill_process_using_port(8080)

# Register cleanup function for graceful shutdown
signal.signal(signal.SIGINT, lambda s, f: cleanup_resources())
signal.signal(signal.SIGTERM, lambda s, f: cleanup_resources())

# Helper Functions
def download_image(url):
    """Downloads an image from a URL and returns its bytes."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image: {e}")
        return None

async def safe_reply_text(message, text, max_retries=3, retry_delay=1):
    """Safely send a text message with retry logic."""
    for attempt in range(max_retries):
        try:
            return await message.reply_text(text)
        except TimedOut:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(retry_delay * (attempt + 1))
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise


# Command Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    welcome_message = (
        "👋 Welcome to the AI Assistant Bot!\n\n"
        "Available commands:\n"
        "/help - Show this help message\n"
        "/generate - Generate an image from text\n"
        "/weather [location] - Get weather forecast\n"
        "/news [location] - Get latest news\n"
        "/advice [question] - Get advice\n"
        "/affirmation - Get a motivational quote\n"
        "/joke - Get a random joke\n"
        "/missing_person - Create a missing person poster from your photo\n"
        "/wanted_person - Create a wanted poster from your photo\n"
        "/slap - Generate a slap image using your photo\n"
        "/truth_or_dare [truth/dare] - Get a truth or dare question\n\n"
        "👉 *Tip for private chats*: Simply send an image and it will be automatically analyzed!"
    )
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    await start(update, context)

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate an image from text prompt."""
    if not context.args:
        await update.message.reply_text("Please provide a text prompt after /generate command.")
        return

    prompt = " ".join(context.args)
    await update.message.reply_text("Generating image... Please wait.")

    try:
        client_gradio = Client("black-forest-labs/FLUX.1-dev", hf_token=hf_token)
        result = client_gradio.predict(
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            width=1024,
            height=1024,
            guidance_scale=3.5,
            num_inference_steps=28,
            api_name="/infer"
        )
        
        # Send the generated image
        await update.message.reply_photo(photo=open(result[0], 'rb'))

    except Exception as e:
        logger.error(f"Error generating image: {e}")
        await update.message.reply_text(f"Error generating image: {str(e)}")

async def llm_response_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle plain text messages in private chats and respond using Gemini LLM."""
    # Ignore commands (which start with '/')
    if update.message.text.startswith("/"):
        return
    # Get username or fallback to user ID as string
    username = update.effective_user.username or str(update.effective_user.id)
    user_message = update.message.text
    
    history = chat_history_manager.load_history(username) 
    recent_messages = history[-5:]
    context_messages = "\n".join([
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in recent_messages
    ])
    prompt = (
        f"Previous conversation:\n{context_messages}\n\n"
        f"User's latest message: '{user_message}'\n"
        "Please provide a thoughtful and helpful response, taking into account the conversation context. Talk freely and you dont need to be concise"
    )
    response = None
    for genai_client in gemini_clients:
        try:
            response = genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            if response.text:
                break
        except Exception:
            continue
    
    if response and response.text:
        chat_history_manager.save_message(username, "user", user_message)
        chat_history_manager.save_message(username, "assistant", response.text)
        await safe_reply_text(update.message, response.text)
    else:
        await safe_reply_text(update.message, "Sorry, I'm not sure how to respond to that right now.")

async def weather_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get weather forecast for a location."""
    if not context.args:
        await update.message.reply_text("Please provide a location after /weather command.")
        return

    location = " ".join(context.args)
    await update.message.reply_text(f"Fetching weather for {location}...")

    try:
        async with python_weather.Client(unit=python_weather.IMPERIAL) as weather_client:
            weather_data = await weather_client.get(location)

            # Create formatted weather message
            weather_message = (
                f"🌤️ Weather in {location.title()}\n\n"
                f"Current Weather:\n"
                f"🌡️ Temperature: **{weather_data.temperature}°F**\n"
                f"🎐 Feels like: **{weather_data.feels_like}°F**\n"
                f"💨 Wind: **{weather_data.wind_speed} mph**\n"
                f"💧 Humidity: **{weather_data.humidity}%**\n"
                f"🌅 Conditions: **{weather_data.description}**\n\n"
                f"📅 3-Day Forecast:\n"
            )

            # Add forecast for next 3 days
            forecast_count = 0
            for forecast in weather_data:
                if forecast_count >= 3:
                    break
                day = forecast.date.strftime("%A")
                # Get first hourly forecast for conditions
                hourly_forecast = next(iter(forecast), None)
                conditions = hourly_forecast.description if hourly_forecast else "N/A"
                
                weather_message += (
                    f"\n{day} ({forecast.date.strftime('%Y-%m-%d')}):\n"
                    f"🌡️ Temperature: **{forecast.temperature}°F**\n"
                    f"🌅 Conditions: **{conditions}**\n"
                )
                forecast_count += 1

            weather_message += "\nData from python-weather"
            await update.message.reply_text(weather_message)

    except Exception as e:
        logger.error(f"Error fetching weather: {str(e)}", exc_info=True)
        await update.message.reply_text(f"Error fetching weather data: {str(e)}")

async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get latest news for a location."""
    if not context.args:
        await update.message.reply_text("Please provide a location after /news command.")
        return

    location = " ".join(context.args)
    await update.message.reply_text(f"Fetching news for {location}...")

    try:
        news_results = google_news.get_news(location)
        
        if not news_results:
            await update.message.reply_text(f"No news found for location: {location}")
            return

        # Format news articles
        news_text = f"📰 Latest News for {location.title()}\n\n"
        for i, article in enumerate(news_results[:5], 1):
            title = article.get('title', 'No title available')
            link = article.get('url', 'No link available')
            description = article.get('description', 'No description available')
            if len(description) > 100:
                description = description[:97] + "..."
            
            news_text += f"{i}. {title}\n{description}\n{link}\n\n"

        await update.message.reply_text(news_text)

    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        await update.message.reply_text(f"Error fetching news: {str(e)}")

async def advice_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get advice using Gemini."""
    if not context.args:
        await update.message.reply_text("Please provide a question after /advice command.")
        return

    question = " ".join(context.args)
    
    try:
        prompt = (
            f"You are a wise counselor. The user asks: '{question}'\n"
            "Give practical, concise advice considering multiple perspectives. "
            "Structure your response with:\n- Key considerations\n- Recommended actions\n- Potential pitfalls\n"
            "Keep each section brief and focused."
        )
        
        response = None
        for genai_client in gemini_clients:
            try:
                response = genai_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                if response.text:
                    break
            except Exception:
                continue

        if not response or not response.text:
            await update.message.reply_text("Couldn't generate advice at this time.")
            return

        await update.message.reply_text(f"Advice for '{question}':\n\n{response.text}")

    except Exception as e:
        logger.error(f"Error generating advice: {e}")
        await update.message.reply_text(f"Error generating advice: {str(e)}")

async def affirmation_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate an affirmation using Gemini."""
    try:
        prompt = (
            "Generate an inspiring affirmation with these components:\n"
            "1. A meaningful quote (2-3 sentences)\n"
            "2. A brief explanation of its significance\n"
            "3. A practical tip for applying it\n"
            "Include relevant emojis and format for readability."
        )
        
        response = None
        for genai_client in gemini_clients:
            try:
                response = genai_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                if response.text:
                    break
            except Exception:
                continue

        if not response or not response.text:
            await update.message.reply_text("Couldn't generate affirmation at this time.")
            return

        await update.message.reply_text(f"Daily Affirmation:\n\n{response.text}")

    except Exception as e:
        logger.error(f"Error generating affirmation: {e}")
        await update.message.reply_text(f"Error generating affirmation: {str(e)}")

async def joke_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate a joke using Gemini."""
    try:
        category = context.args[0] if context.args else "any"
        prompt = f"Tell me a funny, clean, and appropriate {category} joke. Format it with setup and punchline structure."
        
        response = None
        for genai_client in gemini_clients:
            try:
                response = genai_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                if response.text:
                    break
            except Exception:
                continue

        if not response or not response.text:
            await update.message.reply_text("Couldn't generate a joke at this time.")
            return

        await update.message.reply_text(f"😄 Here's your joke!\n\n{response.text}")

    except Exception as e:
        logger.error(f"Error generating joke: {e}")
        await update.message.reply_text(f"Error generating joke: {str(e)}")

async def missing_person_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate a missing person poster using a user's photo."""
    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        await update.message.reply_text("Please reply to a photo with /missing_person command.")
        return

    await update.message.reply_text("Generating missing person poster... Please wait.")
    
    try:
        # Get the largest photo size
        photo = update.message.reply_to_message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        
        # Download the image
        image_bytes = await file.download_as_bytearray()
        avatar_image = Image.open(BytesIO(image_bytes)).convert("RGBA")
        avatar_size = (388, 397)
        avatar_image = avatar_image.resize(avatar_size)
        
        # Open template and paste avatar
        template_path = os.path.join("imgs", "missing_person.jpg")
        template_image = Image.open(template_path).convert("RGBA")
        paste_position = (159, 162)
        template_image.paste(avatar_image, paste_position, avatar_image)
        
        # Save and send result
        with BytesIO() as img_bytes:
            template_image.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            await update.message.reply_photo(photo=img_bytes)

    except Exception as e:
        logger.error(f"Error generating missing person poster: {e}")
        await update.message.reply_text(f"Error generating missing person poster: {str(e)}")

async def wanted_person_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate a wanted poster using a user's photo."""
    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        await update.message.reply_text("Please reply to a photo with /wanted_person command.")
        return

    await update.message.reply_text("Generating wanted poster... Please wait.")
    
    try:
        # Get the largest photo size
        photo = update.message.reply_to_message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        
        # Download the image
        image_bytes = await file.download_as_bytearray()
        avatar_image = Image.open(BytesIO(image_bytes)).convert("RGBA")
        avatar_size = (388, 397)
        avatar_image = avatar_image.resize(avatar_size)
        
        # Open template and paste avatar
        template_path = os.path.join("imgs", "wanted_person.jpeg")
        template_image = Image.open(template_path).convert("RGBA")
        paste_position = (159, 162)
        template_image.paste(avatar_image, paste_position, avatar_image)
        
        # Save and send result
        with BytesIO() as img_bytes:
            template_image.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            await update.message.reply_photo(photo=img_bytes)

    except Exception as e:
        logger.error(f"Error generating wanted poster: {e}")
        await update.message.reply_text(f"Error generating wanted poster: {str(e)}")

async def slap_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate a slap image using a user's photo."""
    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        await update.message.reply_text("Please reply to a photo with /slap command.")
        return

    await update.message.reply_text("Generating slap image... Please wait.")
    
    try:
        # Get the largest photo size
        photo = update.message.reply_to_message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        
        # Download the image
        image_bytes = await file.download_as_bytearray()
        avatar_image = Image.open(BytesIO(image_bytes)).convert("RGBA")
        avatar_size = (388, 397)
        avatar_image = avatar_image.resize(avatar_size)
        
        # Open template and paste avatar
        template_path = os.path.join("imgs", "slapping_person.jpg")
        template_image = Image.open(template_path).convert("RGBA")
        paste_position = (159, 162)
        template_image.paste(avatar_image, paste_position, avatar_image)
        
        # Save and send result
        with BytesIO() as img_bytes:
            template_image.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            await update.message.reply_photo(photo=img_bytes)

    except Exception as e:
        logger.error(f"Error generating slap image: {e}")
        await update.message.reply_text(f"Error generating slap image: {str(e)}")

async def truth_or_dare_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate a truth or dare question."""
    try:
        choice = context.args[0].lower() if context.args else None
        if choice not in ["truth", "dare"]:
            await update.message.reply_text("Please specify 'truth' or 'dare' after the command.")
            return

        prompt = f"Generate a {choice} question that is fun and appropriate for all ages."
        
        response = None
        for genai_client in gemini_clients:
            try:
                response = genai_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                if response.text:
                    break
            except Exception:
                continue

        if not response or not response.text:
            await update.message.reply_text(f"Couldn't generate a {choice} question at this time.")
            return

        await update.message.reply_text(f"🎮 Here's your {choice} question:\n\n{response.text}")

    except Exception as e:
        logger.error(f"Error generating truth or dare: {e}")
        await update.message.reply_text(f"Error generating truth or dare: {str(e)}")

# New handler for automatically analyzing images in private chats
async def auto_analyze_private_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Automatically analyze an image when sent in a private chat."""
    # Only proceed if this is a private chat and the message contains a photo.
    if update.effective_chat.type != "private" or not update.message.photo:
        return

    await update.message.reply_text("Analyzing image... Please wait.")
    try:
        # Get the largest photo size
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        
        # Download the image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            await file.download_to_drive(temp_file.name)
            image_path = temp_file.name

        # Use the Florence model via Gradio for analysis
        client_gradio = Client("MegaTronX/Florence-2-Image-To-Flux-Prompt", hf_token=hf_token)
        result = client_gradio.predict(image=handle_file(image_path), api_name="/feifeichat")

        # Clean up temporary file
        os.unlink(image_path)

        # Send results
        await update.message.reply_text(f"Analysis result:\n{result}")

    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        await update.message.reply_text(f"Error analyzing image: {str(e)}")

async def handle_private_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle private messages: text for interactive chat, photos for analysis."""
    if update.effective_chat.type != "private":
        return

    user_id = update.effective_user.username or str(update.effective_user.id)
    
    # Initialize GoogleServices for new users
    if user_id not in user_google_services:
        google_services = GoogleServices(user_name=user_id)
        try:
            # Check if token file exists
            token_path = os.path.join('users', f'{user_id}_token.pickle')
            if not os.path.exists(token_path):
                # Get authorization URL
                auth_url, redirect_uri = google_services.authenticate(return_auth_url=True)
                
                # Create an embedded message with the authorization link
                auth_message = (
                    "🔐 Google Account Authorization Required\n\n"
                    "To use Google services (Calendar, Tasks, etc.), please:\n\n"
                    "1️⃣ Click the link below to authorize your Google account\n"
                    "2️⃣ Sign in and grant the required permissions\n"
                    "3️⃣ After authorization, you'll be redirected to a local page\n"
                    "4️⃣ Once complete, you can continue using the bot\n\n"
                    f"🔗 Authorization Link:\n{auth_url}\n\n"
                    "Note: This is a one-time setup process."
                )
                
                await update.message.reply_text(auth_message)
                return
            
            google_services.authenticate()
            user_google_services[user_id] = google_services
        except Exception as e:
            await update.message.reply_text("Error authenticating with Google services. Please try again later.")
            logger.error(f"Authentication error for user {user_id}: {e}")
            return

    # Handle photos
    if update.message.photo:
        await auto_analyze_private_image(update, context)
        return

    # Handle text messages using process_user_input
    if update.message.text:
        try:
            reply = process_user_input(update.message.text, user_google_services[user_id])

            chat_history_manager.save_message(username=user_id, role="user", content=update.message.text)
            chat_history_manager.save_message(username=user_id, role="assistant", content=reply)
            if "No action needed." in reply:
                await llm_response_handler(update, context)
            else:
                await update.message.reply_text(reply)
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            await update.message.reply_text("Sorry, there was an error processing your request.")

def main():
    """Start the bot."""
    try:
        # Build the application with JobQueue explicitly enabled
        application = Application.builder().token(TOKEN).job_queue(JobQueue()).build()

        # Configure the JobQueue timezone
        job_queue = application.job_queue
        if job_queue is None:
            logger.error("JobQueue is not initialized!")
            sys.exit(1)
        job_queue.scheduler.configure(timezone=pytz.timezone("UTC"))

        # Add error handler
        async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            try:
                logger.error("Exception while handling an update:", exc_info=context.error)
                
                if isinstance(context.error, Conflict):
                    logger.error("Bot instance conflict detected. Shutting down...")
                    await application.stop()
                    sys.exit(1)
                elif isinstance(context.error, NetworkError):
                    logger.error("Network error occurred. Will retry...")
                else:
                    logger.error("Unexpected error occurred")
            except Exception as e:
                logger.error(f"Error in error handler: {e}", exc_info=True)

        application.add_error_handler(error_handler)
    
        # Add command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("generate", generate_image))
        application.add_handler(CommandHandler("weather", weather_command))
        application.add_handler(CommandHandler("news", news_command))
        application.add_handler(CommandHandler("advice", advice_command))
        application.add_handler(CommandHandler("affirmation", affirmation_command))
        application.add_handler(CommandHandler("joke", joke_command))
        application.add_handler(CommandHandler("missing_person", missing_person_command))
        application.add_handler(CommandHandler("wanted_person", wanted_person_command))
        application.add_handler(CommandHandler("slap", slap_command))
        application.add_handler(CommandHandler("truth_or_dare", truth_or_dare_command))
    
        # Add message handler for private chats (both photos and text)
        application.add_handler(MessageHandler(
            (filters.PHOTO | filters.TEXT) & filters.ChatType.PRIVATE,
            handle_private_message
        ))
    
        # Start the Bot
        logger.info("Starting bot...")
        application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user. Shutting down...")
        sys.exit(0)