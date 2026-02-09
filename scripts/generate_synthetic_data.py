import json
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()

def extract_json(text: str):
    """
    Extract JSON from a string that may contain markdown fences.
    """
    # Remove ```json or ``` wrappers if present
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    # Trim whitespace
    text = text.strip()

    return json.loads(text)

# ----------------------------
# Configuration
# ----------------------------
MODEL = "gpt-4o-mini"
OUTPUT_FILE = "../data/synthetic_incidents_v1.json"

ROOT_CAUSES = [
    "Configuration Error",
    "Capacity / Load Issue",
    "Code Regression",
    "Upstream Dependency Failure",
    "Data Quality / Inconsistency",
    "Environmental / Infrastructure Issue",
    "Unknown / Insufficient Data"
]

SYSTEM_PROMPT = """
You are generating synthetic telecom incident tickets for evaluating LLM reasoning quality.

Rules:
- Incidents must resemble real enterprise telecom tickets (imperfect, partial, messy).
- Do NOT explicitly state the root cause in the ticket text.
- Some tickets must be ambiguous or misleading.
- Some tickets must lack sufficient information to diagnose confidently.
- Use realistic telecom language (CDRs, mediation, probes, KPIs, signaling, etc.).

Generate incidents strictly as JSON.
"""

USER_PROMPT = f"""
Generate 12 synthetic telecom incident tickets.

For each ticket include:
- ticket_id
- short_description
- detailed_notes
- severity (1–4)
- true_root_cause (one of: {ROOT_CAUSES})
- ambiguity_level (low | medium | high)

Ensure:
- At least 3 tickets have ambiguity_level = high
- At least 2 tickets have true_root_cause = "Unknown / Insufficient Data"
- At least 2 tickets involve post-change / post-deployment symptoms
"""

def main():
    client = OpenAI()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ],
        temperature=0.7
    )

    content = response.choices[0].message.content

    try:
        incidents = extract_json(content)
    except Exception as e:
        raise ValueError(
            "Failed to extract valid JSON from model output.\n"
            f"Raw output was:\n{content}"
        ) from e

    with open(OUTPUT_FILE, "w") as f:
        json.dump(incidents, f, indent=2)

    print(f"✅ Generated {len(incidents)} synthetic incidents")
    print(f"📁 Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
