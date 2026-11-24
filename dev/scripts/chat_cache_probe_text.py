"""Shared prompt text builder for the chat probes."""
import time

TOPICS = ["glaciers", "container ships", "the printing press", "chess openings", "volcanoes", "bread baking", "railway signalling", "tide tables", "coral reefs", "bridge engineering"]


def system_text(nonce, tokens):
    parts = [f"Session {nonce}. You are a careful assistant that answers from the reference notes below."]
    i = 0
    while sum(len(p) for p in parts) < tokens * 4.1:
        t = TOPICS[i % len(TOPICS)]
        parts.append(f"Note {i} on {t}: entry {nonce}-{i * 7919 % 1009} records that the {t} survey of block {i * 31 % 97} found {i * 13 % 89 + 3} items, "
                     f"with a margin of {i * 17 % 41} percent, and the next review of {t} is due in cycle {i * 5 % 23}.")
        i += 1
    return "\n".join(parts)
