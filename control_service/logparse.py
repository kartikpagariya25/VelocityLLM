import re

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
LEVEL = re.compile(r"\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]|\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b")
REQ = re.compile(r"\b(?:Request|request|completion|Active request)\s+([A-Za-z0-9_.:-]+)")
APP = re.compile(r"^\S+ \S+ \[\w+\] \[cid=[^\]]*\] [\w.]+: ")


def classify(raw: str):
    text = ANSI.sub("", raw).rstrip()
    m = LEVEL.search(text)
    level = (m.group(1) or m.group(2)) if m else ("WARNING" if "Warning" in text else "INFO")
    message = APP.sub("", text, count=1)
    low = message.lower()
    rid = None
    rm = REQ.search(message)
    if rm:
        rid = rm.group(1).strip(".,:")
    if "traceback" in low or level in ("ERROR", "CRITICAL"):
        category = "error"
    elif "adaptive batch controller" in low or "concurrency" in low and "->" in message:
        category = "controller"
    elif re.search(r"rejected|\bshed\b|shedding|queue full|admitted", low):
        category = "admission"
    elif re.search(r"cancel|disconnect|aborted", low):
        category = "cancel"
    elif rid and any(w in low for w in ("started", "completed", "executing", "scheduled")):
        category = "lifecycle"
    else:
        category = "system"
    if level == "WARN":
        level = "WARNING"
    return level, category, rid, message


NOISE = re.compile(r'"GET /(stats|health|smart/trace)\b')


def is_noise(raw: str) -> bool:
    return bool(NOISE.search(raw))
