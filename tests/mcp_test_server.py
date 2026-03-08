#!/usr/bin/env python3
"""
Custom MCP test server with deterministic mock tools.

Tools return fixed, predictable data for assertion-based testing.
Uses FastMCP for stdio transport (wrap with supergateway for HTTP).

Usage:
  # Direct stdio (for supergateway wrapping):
  python tests/mcp_test_server.py

  # Via supergateway:
  npx -y supergateway \
    --stdio "python tests/mcp_test_server.py" \
    --outputTransport streamableHttp --port 18322
"""

import json
import math
import re
import sys

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("ERROR: Install fastmcp: pip install 'mcp[cli]'", file=sys.stderr)
    sys.exit(1)

mcp = FastMCP("heartbit-test-tools")

# ── Deterministic mock data ──────────────────────────────────────

EMPLOYEES = {
    "alice": {
        "name": "Alice Johnson",
        "role": "Senior Engineer",
        "department": "Platform",
        "email": "alice@example.com",
        "employee_id": "EMP-1001",
    },
    "bob": {
        "name": "Bob Smith",
        "role": "Product Manager",
        "department": "Product",
        "email": "bob@example.com",
        "employee_id": "EMP-1002",
    },
    "carol": {
        "name": "Carol Davis",
        "role": "Data Scientist",
        "department": "Analytics",
        "email": "carol@example.com",
        "employee_id": "EMP-1003",
    },
}

WEATHER = {
    "london": {"city": "London", "temp_c": 12, "condition": "Cloudy", "humidity": 78},
    "tokyo": {"city": "Tokyo", "temp_c": 22, "condition": "Sunny", "humidity": 55},
    "new york": {"city": "New York", "temp_c": 18, "condition": "Partly Cloudy", "humidity": 62},
    "paris": {"city": "Paris", "temp_c": 15, "condition": "Rainy", "humidity": 85},
}

TRANSLATIONS = {
    "fr": {"hello": "bonjour", "goodbye": "au revoir", "thank you": "merci"},
    "es": {"hello": "hola", "goodbye": "adiós", "thank you": "gracias"},
    "de": {"hello": "hallo", "goodbye": "auf wiedersehen", "thank you": "danke"},
}

notification_log = []
report_store = {}
report_counter = 0

DATABASE = {
    "employees": [
        {"id": 1, "name": "Alice Johnson", "department": "Platform", "salary_band": "L5"},
        {"id": 2, "name": "Bob Smith", "department": "Product", "salary_band": "L4"},
        {"id": 3, "name": "Carol Davis", "department": "Analytics", "salary_band": "L5"},
        {"id": 4, "name": "Dan Lee", "department": "Platform", "salary_band": "L3"},
    ],
    "projects": [
        {"id": 101, "name": "Atlas", "status": "active", "lead": "Alice Johnson"},
        {"id": 102, "name": "Beacon", "status": "completed", "lead": "Bob Smith"},
        {"id": 103, "name": "Cipher", "status": "active", "lead": "Carol Davis"},
    ],
}

KNOWLEDGE_BASE = {
    "agent": [
        {"title": "Multi-Agent Orchestration", "content": "Orchestrators delegate tasks to specialized sub-agents.", "source": "docs/architecture.md"},
        {"title": "Agent Communication", "content": "Agents communicate via blackboard and delegation tools.", "source": "docs/patterns.md"},
    ],
    "guardrail": [
        {"title": "PII Detection", "content": "PII guardrails scan for SSNs, emails, and credit cards.", "source": "docs/security.md"},
        {"title": "Injection Prevention", "content": "Injection classifiers detect prompt injection attempts.", "source": "docs/security.md"},
    ],
    "tool": [
        {"title": "MCP Protocol", "content": "Model Context Protocol enables tool server connectivity.", "source": "docs/tools.md"},
        {"title": "Built-in Tools", "content": "14 built-in tools: read, write, bash, patch, etc.", "source": "docs/tools.md"},
    ],
}


# ── Tools ────────────────────────────────────────────────────────


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Supports basic arithmetic (+, -, *, /), powers (**), and common
    math functions (sqrt, sin, cos, tan, log, abs, pi, e).

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
    """
    # Safe math evaluation — only allow math operations
    allowed_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "abs": abs,
        "pi": math.pi,
        "e": math.e,
        "pow": pow,
        "round": round,
    }
    try:
        # Reject anything that isn't math
        if re.search(r"[a-zA-Z_]", re.sub(r"\b(" + "|".join(allowed_names) + r")\b", "", expression)):
            return json.dumps({"error": "Invalid expression: only math operations allowed"})

        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return json.dumps({"expression": expression, "result": result})
    except Exception as exc:
        return json.dumps({"error": f"Calculation failed: {exc}"})


@mcp.tool()
def lookup_employee(name: str) -> str:
    """Look up an employee by name in the company directory.

    Returns employee details including role, department, email, and ID.

    Args:
        name: Employee first name (case-insensitive)
    """
    key = name.strip().lower()
    if key in EMPLOYEES:
        return json.dumps(EMPLOYEES[key])
    # Fuzzy match
    for k, v in EMPLOYEES.items():
        if key in k or key in v["name"].lower():
            return json.dumps(v)
    return json.dumps({"error": f"Employee '{name}' not found", "available": list(EMPLOYEES.keys())})


@mcp.tool()
def send_notification(to: str, message: str) -> str:
    """Send a notification to a recipient (mock — logs but doesn't actually send).

    Args:
        to: Recipient identifier (email, name, or channel)
        message: Notification message content
    """
    entry = {"to": to, "message": message, "status": "sent"}
    notification_log.append(entry)
    return json.dumps(
        {
            "status": "success",
            "notification_id": f"NOTIF-{len(notification_log):04d}",
            "to": to,
            "message_preview": message[:50],
        }
    )


@mcp.tool()
def get_weather(city: str) -> str:
    """Get current weather for a city (mock data).

    Args:
        city: City name (e.g., "London", "Tokyo")
    """
    key = city.strip().lower()
    if key in WEATHER:
        return json.dumps(WEATHER[key])
    return json.dumps({"error": f"Weather data not available for '{city}'", "available_cities": list(WEATHER.keys())})


@mcp.tool()
def translate(text: str, target_lang: str) -> str:
    """Translate text to a target language (mock — uses a small dictionary).

    Args:
        text: Text to translate
        target_lang: Target language code (fr, es, de)
    """
    lang = target_lang.strip().lower()
    if lang not in TRANSLATIONS:
        return json.dumps({"error": f"Language '{target_lang}' not supported", "supported": list(TRANSLATIONS.keys())})

    dictionary = TRANSLATIONS[lang]
    key = text.strip().lower()
    if key in dictionary:
        return json.dumps({"original": text, "translated": dictionary[key], "language": lang})

    return json.dumps(
        {
            "original": text,
            "translated": f"[mock translation of '{text}' to {lang}]",
            "language": lang,
            "note": "exact translation not in dictionary",
        }
    )


@mcp.tool()
def get_notification_log() -> str:
    """Retrieve all sent notifications (for verification)."""
    return json.dumps({"notifications": notification_log, "total": len(notification_log)})


@mcp.tool()
def query_database(table: str, filter_field: str = "", filter_value: str = "") -> str:
    """Query a mock database table with optional filtering.

    Args:
        table: Table name (employees, projects)
        filter_field: Optional field name to filter on
        filter_value: Optional value to match (case-insensitive substring)
    """
    tbl = table.strip().lower()
    if tbl not in DATABASE:
        return json.dumps({"error": f"Table '{table}' not found", "available": list(DATABASE.keys())})

    rows = DATABASE[tbl]
    if filter_field and filter_value:
        field = filter_field.strip().lower()
        value = filter_value.strip().lower()
        rows = [r for r in rows if field in r and value in str(r[field]).lower()]

    return json.dumps({"table": tbl, "rows": rows, "count": len(rows)})


@mcp.tool()
def create_report(title: str, sections: str) -> str:
    """Create a report from title and sections. Returns a report ID.

    Args:
        title: Report title
        sections: Report sections as a JSON array of strings, or plain text
    """
    global report_counter
    report_counter += 1
    report_id = f"RPT-{report_counter:04d}"

    try:
        section_list = json.loads(sections) if sections.startswith("[") else [sections]
    except json.JSONDecodeError:
        section_list = [sections]

    report = {"id": report_id, "title": title, "sections": section_list}
    report_store[report_id] = report
    return json.dumps({"status": "created", "report_id": report_id, "title": title, "section_count": len(section_list)})


@mcp.tool()
def get_report(report_id: str) -> str:
    """Retrieve a previously created report by ID.

    Args:
        report_id: Report ID (e.g., RPT-0001)
    """
    rid = report_id.strip().upper()
    if rid in report_store:
        return json.dumps(report_store[rid])
    return json.dumps({"error": f"Report '{report_id}' not found"})


@mcp.tool()
def search_knowledge(query: str) -> str:
    """Search the knowledge base for relevant entries.

    Args:
        query: Search query (matches against titles and content)
    """
    query_lower = query.strip().lower()
    results = []
    for category, entries in KNOWLEDGE_BASE.items():
        for entry in entries:
            if query_lower in entry["title"].lower() or query_lower in entry["content"].lower():
                results.append({**entry, "category": category})

    # Fallback: match by category
    if not results and query_lower in KNOWLEDGE_BASE:
        results = [{**e, "category": query_lower} for e in KNOWLEDGE_BASE[query_lower]]

    return json.dumps({"query": query, "results": results, "count": len(results)})


if __name__ == "__main__":
    mcp.run(transport="stdio")
