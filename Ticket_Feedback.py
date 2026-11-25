import json
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


# Load secrets and settings from the .env file beside this Python file.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.5-flash")

if not API_KEY:
    raise RuntimeError(
        "Missing GEMINI_API_KEY. Create a .env file and add your Gemini API key."
    )

client = genai.Client(api_key=API_KEY)


SAMPLE_TICKETS = [
    "yo wtf?? i got charged $249.99 TWICE on my visa ending 4821 this morning!! i only signed up for the starter plan not enterprise. fix this or im calling my bank for a chargeback asap",
    "Hi team, I just got 4 password reset emails I didn't request and now my 2FA phone number was changed to a number in another country?? Please lock my account immediately I think someone hacked me.",
    "app keeps crashing every single time i click export to pdf!! error code 502 pops up and i have a client presentation in 20 mins pls help",
    "Loving the new dashboard update btw!! Quick idea: would be super cool if we could schedule weekly CSV reports to go straight to Slack every Monday morning. No rush, just thought I'd share :)",
    "Oh great, another $15 'mystery add-on fee' on my invoice that I definitely never clicked. Love paying extra for stuff I don't use. Can someone take this off next month's bill?",
    "Your sync bug wiped three weeks of our team's project notes yesterday. Our legal counsel is reviewing your SLA terms right now and we will pursue damages if data isn't restored by 5 PM.",
    "hello i changed my work email after our company rebranded from oldco.com to newco.io and now SSO wont let me sign in. can u update my login email?",
    "hey so dark mode looks kinda weird on safari like the sidebar text is grey on grey barely readable also while ur fixing css pls add keyboard shortcuts for archiving tasks thx",
    "Hey! Just renewed our annual team workspace—everything has been running smoothly. Could you send over a GST/tax invoice with our company address for our accounting folder?",
    "cant log in on my new phone bcuz my old phone screen broke and google authenticator was on it. still know my password tho how do i reset 2fa",
]


class TicketAnalysis(BaseModel):
    category: Literal[
        "billing", "technical_bug", "account_access", "feature_request"
    ] = Field(description="The single best category for the customer's main issue.")
    urgency: Literal["low", "medium", "high"] = Field(
        description="High for security incidents, legal threats, major data loss, "
                    "or urgent service failures; medium for normal blocking issues; "
                    "low for routine requests or positive feedback."
    )
    sentiment: Literal["positive", "neutral", "angry"] = Field(
        description="Overall customer tone. Choose angry for frustration, threats, "
                    "or strongly negative language."
    )
    requires_human: bool = Field(
        description="True if the ticket mentions a suspected account compromise or "
                    "security breach, a legal threat or chargeback, a refund or "
                    "disputed charge greater than $100, or serious/irreversible data "
                    "loss. Otherwise false."
    )
    short_summary: str = Field(
        description="One-sentence summary of the customer's main issue, under 15 words."
    )


SYSTEM_PROMPT = """
You are a careful customer-support ticket triage classifier.
Classify the customer's message using only the facts in that message.

Rules:
- category must be exactly one of: billing, technical_bug, account_access,
  feature_request.
- urgency must be exactly one of: low, medium, high.
- sentiment must be exactly one of: positive, neutral, angry.
- requires_human must be true for any suspected security breach/account takeover,
  legal threat, chargeback threat, disputed/refund amount above $100, or serious
  data loss. Otherwise it is false.
- Do not expose or repeat full payment-card numbers, passwords, or authentication
  secrets. If a ticket contains sensitive data, do not include it in the summary.
- short_summary must be one sentence and fewer than 15 words.
- Do not invent facts. If a detail is unclear, classify conservatively.

Examples:
1. "Someone changed my password and logged in from another country."
   -> account_access, high urgency, angry or neutral based on wording,
      requires_human=true.
2. "Please add scheduled weekly reports to Slack."
   -> feature_request, low urgency, neutral, requires_human=false.
3. "I was charged $250 twice and will dispute it with my bank."
   -> billing, high urgency, angry, requires_human=true.
"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def classify_ticket(ticket_text: str) -> tuple[TicketAnalysis, float]:
    """Send one ticket to Gemini and return validated output and latency in ms."""
    start = time.perf_counter()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=ticket_text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=TicketAnalysis,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    analysis = TicketAnalysis.model_validate_json(response.text)
    latency_ms = (time.perf_counter() - start) * 1000
    return analysis, latency_ms


def main() -> None:
    results = []

    print(f"Using Gemini model: {MODEL_NAME}")
    print(f"Classifying {len(SAMPLE_TICKETS)} sample tickets...\n")

    for idx, ticket in enumerate(SAMPLE_TICKETS, start=1):
        try:
            analysis, latency_ms = classify_ticket(ticket)
            results.append({
                "ticket_id": idx,
                **analysis.model_dump(),
                "latency_ms": round(latency_ms, 2),
            })
            print(
                f"#{idx:02d} | {analysis.category:<15} | "
                f"{analysis.urgency:<6} | {analysis.sentiment:<8} | "
                f"human={str(analysis.requires_human):<5} | {latency_ms:.0f}ms"
            )
            print(f"     Summary: {analysis.short_summary}")
        except Exception as exc:
            # Continue processing remaining tickets if one ticket fails.
            print(f"#{idx:02d} | ERROR: {type(exc).__name__}: {exc}")

    output_file = BASE_DIR / "classified_tickets.json"
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} successful classifications to:")
    print(output_file)


if __name__ == "__main__":
    main()
