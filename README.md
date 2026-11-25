# Ticket Feedback Classifier

A Python command-line application that classifies customer support tickets with the Google Gemini API and saves structured results to JSON.

## Requirements
- Python 3.10 or newer
- A Gemini API key from https://aistudio.google.com/apikey

## Windows PowerShell setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder with your real API key.

## Run

```powershell
python Ticket_Feedback.py
```

The results are written to `classified_tickets.json`.

## Important
- Keep `.env` private. Never commit or share your API key.
- `.gitignore` excludes `.env` and generated output.
- The classifier uses Gemini structured JSON output and validates it with Pydantic.
- API calls may consume quota and may incur charges depending on your Google AI plan.
