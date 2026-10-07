import base64
import os
import time
import pandas as pd

from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def gmail_authenticate():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Store OAuth token in Windows user folder
    token_path = os.path.join(os.path.expanduser("~"), "gmail_token.json")

    # credentials.json remains inside the project folder
    credentials_path = os.path.join(base_dir, "credentials.json")

    creds = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            credentials_path,
            SCOPES
        )

        creds = flow.run_local_server(port=0)

        with open(token_path, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def get_full_body(payload):
    # Handle simple non-multipart emails
    if payload.get("body", {}).get("data"):
        data = payload["body"]["data"]

        return base64.urlsafe_b64decode(data).decode(
            "utf-8",
            errors="ignore"
        )

    # Handle multipart emails
    parts = payload.get("parts", [])

    for part in parts:
        mime_type = part.get("mimeType", "")

        if mime_type == "text/plain":
            data = part.get("body", {}).get("data")

            if data:
                return base64.urlsafe_b64decode(data).decode(
                    "utf-8",
                    errors="ignore"
                )

        elif mime_type == "text/html":
            data = part.get("body", {}).get("data")

            if data:
                html = base64.urlsafe_b64decode(data).decode(
                    "utf-8",
                    errors="ignore"
                )

                return BeautifulSoup(
                    html,
                    "html.parser"
                ).get_text(" ", strip=True)

        # Check nested MIME parts
        nested_body = get_full_body(part)

        if nested_body.strip():
            return nested_body

    return ""


def fetch_message(service, message_id):
    """
    Retrieve one Gmail message.

    If Gmail rate-limits us, wait and retry using
    exponential backoff.
    """

    delay = 2

    for attempt in range(6):

        try:
            return (
                service.users()
                .messages()
                .get(
                    userId="me",
                    id=message_id,
                    format="full"
                )
                .execute()
            )

        except HttpError as error:

            if error.resp.status in [403, 429]:

                print(
                    f"Rate limit reached. "
                    f"Waiting {delay} seconds..."
                )

                time.sleep(delay)

                delay *= 2

            else:
                raise

    print("Skipping message after repeated failures.")
    return None


def get_emails(service, label, label_name, max_results=100):

    print(f"\nFetching {label_name} emails...")

    results = (
        service.users()
        .messages()
        .list(
            userId="me",
            labelIds=[label],
            maxResults=max_results
        )
        .execute()
    )

    messages = results.get("messages", [])

    data = []

    total = len(messages)

    for index, msg in enumerate(messages, start=1):

        email_data = fetch_message(
            service,
            msg["id"]
        )

        if email_data is None:
            continue

        payload = email_data.get("payload", {})

        body = get_full_body(payload)

        if body.strip():

            data.append({
                "body": body,
                "label": label_name
            })

        print(
            f"\rProcessed {index}/{total} {label_name} emails",
            end=""
        )

        # Small pause to avoid Gmail API rate limits
        time.sleep(0.25)

    print()

    return data


# ---------------------------------------
# MAIN
# ---------------------------------------

service = gmail_authenticate()

print("\n📥 Fetching emails from Gmail...")

# Start with 100 + 100.
# This is enough to verify the ML pipeline works.
inbox_data = get_emails(
    service,
    "INBOX",
    "ham",
    max_results=100
)

spam_data = get_emails(
    service,
    "SPAM",
    "spam",
    max_results=100
)

all_data = inbox_data + spam_data

df = pd.DataFrame(all_data)

dataset_path = os.path.join(
    os.path.expanduser("~"),
    "emails_dataset.csv"
)

df.to_csv(
    dataset_path,
    index=False
)

print("\n--------------------------------")
print("Dataset created successfully")
print("--------------------------------")

print(f"Ham emails : {len(inbox_data)}")
print(f"Spam emails: {len(spam_data)}")
print(f"Total      : {len(df)}")

print(f"\nSaved as {dataset_path}")