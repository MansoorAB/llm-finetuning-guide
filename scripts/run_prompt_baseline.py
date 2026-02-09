import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------
# Setup
# ----------------------------
load_dotenv()

import re

def extract_json(text: str):
    """
    Extract and parse JSON from model output, handling markdown fences.
    """
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    return json.loads(text.strip())


API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=API_KEY)

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.3  # we will discuss this below
DATA_FILE = "../data/synthetic_incidents_v2.json"
OUTPUT_FILE = "../data/baseline_outputs_v2.json"

# ----------------------------
# Frozen baseline prompt
# ----------------------------
BASELINE_PROMPT = """
You are a senior telecom incident triage assistant.

Given an incident ticket, your task is to identify the most likely root cause
based only on the information provided.

Rules:
- Choose exactly ONE root cause from the allowed list.
- If the information is insufficient, respond with "Unknown / Insufficient Data".
- Do not speculate or assume missing facts.
- Avoid generic advice.
- Follow the output format exactly.

Allowed Root Causes:
- Configuration Error
- Capacity / Load Issue
- Code Regression
- Upstream Dependency Failure
- Data Quality / Inconsistency
- Environmental / Infrastructure Issue
- Unknown / Insufficient Data

Output Format (JSON only):
{
  "root_cause": "",
  "confidence": "High | Medium | Low",
  "reasoning": "",
  "recommended_next_action": "",
  "similar_incidents": []
}
"""

def run_ticket(ticket, run_id):
    messages = [
        {"role": "system", "content": BASELINE_PROMPT},
        {
            "role": "user",
            "content": f"""
                    Incident Ticket:
                    Short Description: {ticket['short_description']}
                    Details: {ticket['detailed_notes']}
                    Severity: {ticket['severity']}
                """
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE
    )

    raw_output = response.choices[0].message.content

    try:
        parsed_output = extract_json(raw_output)
    except Exception:
        parsed_output = {
            "_parse_error": True,
            "raw_output": raw_output
        }

    return {
        "ticket_id": ticket["ticket_id"],
        "run_id": run_id,
        "model_output": parsed_output
    }


def main():
    with open(DATA_FILE) as f:
        tickets = json.load(f)

    results = []

    # Pick first 6 tickets; you can change this selection
    selected_tickets = tickets[:10]

    for ticket in selected_tickets:
        # Single run
        results.append(run_ticket(ticket, run_id=1))

        # Repeat runs for first two tickets
        if ticket["ticket_id"] in [selected_tickets[0]["ticket_id"],
                                   selected_tickets[1]["ticket_id"]]:
            results.append(run_ticket(ticket, run_id=2))
            results.append(run_ticket(ticket, run_id=3))

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Baseline runs completed")
    print(f"📁 Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
