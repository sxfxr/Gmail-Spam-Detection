import os
import re
import base64
import joblib
import pandas as pd

from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials


# =========================================================
# SETTINGS
# =========================================================

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

HOME = os.path.expanduser("~")

TOKEN_PATH = os.path.join(HOME, "gmail_token.json")
MODEL_PATH = os.path.join(HOME, "spam_model.pkl")
FEATURES_PATH = os.path.join(HOME, "spambase_features.pkl")


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading spam detection model...")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

print("Model loaded successfully.")


# =========================================================
# GMAIL AUTHENTICATION
# =========================================================

def gmail_authenticate():

    creds = Credentials.from_authorized_user_file(
        TOKEN_PATH,
        SCOPES
    )

    service = build(
        "gmail",
        "v1",
        credentials=creds
    )

    return service


# =========================================================
# EMAIL BODY EXTRACTION
# =========================================================

def decode_data(data):

    if not data:
        return ""

    try:
        return base64.urlsafe_b64decode(data).decode(
            "utf-8",
            errors="ignore"
        )
    except Exception:
        return ""


def extract_body(payload):

    mime_type = payload.get("mimeType", "")

    body_data = payload.get("body", {}).get("data")

    if body_data:

        text = decode_data(body_data)

        if mime_type == "text/html":
            text = BeautifulSoup(
                text,
                "html.parser"
            ).get_text(" ", strip=True)

        return text

    for part in payload.get("parts", []):

        body = extract_body(part)

        if body.strip():
            return body

    return ""


# =========================================================
# SPAMBASE FEATURE EXTRACTION
# =========================================================

def word_frequency(text, word):

    text_lower = text.lower()

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text_lower
    )

    if not words:
        return 0.0

    count = sum(
        1 for token in words
        if token == word.lower()
    )

    return (count / len(words)) * 100


def char_frequency(text, character):

    if not text:
        return 0.0

    count = text.count(character)

    return (count / len(text)) * 100


def capital_statistics(text):

    runs = re.findall(
        r"[A-Z]+",
        text
    )

    if not runs:
        return 1.0, 1.0, 0.0

    lengths = [
        len(run)
        for run in runs
    ]

    average = sum(lengths) / len(lengths)
    longest = max(lengths)
    total = sum(lengths)

    return average, longest, total


def extract_spambase_features(text):

    features = []

    capital_avg, capital_longest, capital_total = (
        capital_statistics(text)
    )

    for feature in feature_names:

        name = str(feature).lower()

        # Word-frequency features
        if name.startswith("word_freq_"):

            word = name.replace(
                "word_freq_",
                ""
            )

            value = word_frequency(
                text,
                word
            )

        # Character-frequency features
        elif name.startswith("char_freq_"):

            char_name = name.replace(
                "char_freq_",
                ""
            )

            char_map = {
                "semicolon": ";",
                "(": "(",
                "[": "[",
                "!": "!",
                "$": "$",
                "#": "#"
            }

            character = char_map.get(
                char_name,
                char_name
            )

            value = char_frequency(
                text,
                character
            )

        elif "capital_run_length_average" in name:

            value = capital_avg

        elif "capital_run_length_longest" in name:

            value = capital_longest

        elif "capital_run_length_total" in name:

            value = capital_total

        else:
            value = 0.0

        features.append(value)

    return pd.DataFrame(
    [features],
    columns=feature_names
)


# =========================================================
# READ AND CLASSIFY GMAIL
# =========================================================

def check_recent_emails(service, max_results=10):

    result = (
        service.users()
        .messages()
        .list(
            userId="me",
            labelIds=["INBOX"],
            maxResults=max_results
        )
        .execute()
    )

    messages = result.get(
        "messages",
        []
    )

    if not messages:

        print("No emails found.")
        return

    print(
        f"\nChecking {len(messages)} recent inbox emails...\n"
    )

    for number, message in enumerate(
        messages,
        start=1
    ):

        email = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=message["id"],
                format="full"
            )
            .execute()
        )

        payload = email.get(
            "payload",
            {}
        )

        headers = payload.get(
            "headers",
            []
        )

        subject = "No Subject"

        for header in headers:

            if header.get(
                "name",
                ""
            ).lower() == "subject":

                subject = header.get(
                    "value",
                    "No Subject"
                )

                break

        body = extract_body(
            payload
        )

        # Include subject because it often contains
        # useful spam indicators.
        full_text = subject + " " + body

        features = extract_spambase_features(
            full_text
        )

        prediction = model.predict(
            features
        )[0]

        if hasattr(
            model,
            "predict_proba"
        ):

            probability = model.predict_proba(
                features
            )[0][1]

        else:
            probability = None

        print("=" * 60)

        print(
            f"EMAIL {number}"
        )

        print(
            f"Subject: {subject[:100]}"
        )

        if prediction == 1:

            print("Prediction: SPAM")

        else:

            print("Prediction: NOT SPAM")

        if probability is not None:

            print(
                f"Spam probability: "
                f"{probability * 100:.2f}%"
            )

        print()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    service = gmail_authenticate()

    check_recent_emails(
        service,
        max_results=10
    )