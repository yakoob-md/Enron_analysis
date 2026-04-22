# multiclass_pipeline/models/llm/llm_classifier.py
import json

SYSTEM_PROMPT = """
You are an expert analyst of the Enron email corpus.
Classify each email into exactly one of these disclosure categories:

NONE       - Routine business communication, no disclosure signal
STRATEGIC  - Strategic business decisions, mergers, acquisitions, restructuring
RELATIONAL - Personal/interpersonal communication, team coordination
LEGAL      - Legal proceedings, regulatory filings, SEC, FERC, attorney communication
FINANCIAL  - Financial figures, earnings, write-downs, balance sheet items

Respond ONLY with a JSON object in this exact format:
{"label": "LEGAL", "confidence": 0.87, "reason": "Email references SEC filing and attorney counsel"}

Do not include any other text.
"""

def classify_with_llm(email_text, client, model_name="claude-3-opus-20240229"):
    """
    Classifies email text using an LLM (Anthropic Claude example).
    Adapts based on the provided SYSTEM_PROMPT for 5-class classification.
    """
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=150,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": email_text[:2000]}]
        )
        return json.loads(response.content[0].text)
    except Exception as e:
        print(f"LLM Classification Error: {e}")
        return {"label": "NONE", "confidence": 0.0, "reason": "Error during classification"}
