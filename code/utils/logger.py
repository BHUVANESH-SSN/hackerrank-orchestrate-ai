"""
Logging utilities — structlog setup and ChatTranscriptLogger for log.txt.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import structlog


def setup_structlog() -> None:
    ### use of this function: setup structlog
    """Configure structlog for JSON file output + human-readable stdout."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20), 
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    ### use of this function: get logger
    """Get a named structlog logger."""
    setup_structlog()
    return structlog.get_logger(name)


class ChatTranscriptLogger:
    """
    Accumulates a readable conversation-style log per ticket.
    Writes the full log.txt on save().
    """

    def __init__(self) -> None:
        ### use of this function: init
        self._entries: list[str] = []

    def log_ticket_start(self, ticket_id: str, raw_text: str) -> None:
        ### use of this function: log ticket start
        """Log the start of a new ticket processing."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        from utils.safety import redact_pii
        redacted = redact_pii(raw_text)
        self._entries.append(
            f"\n{'=' * 80}\n"
            f"TICKET: {ticket_id} | {timestamp}\n"
            f"{'=' * 80}\n"
            f"INPUT: {redacted}\n"
        )

    def log_agent_step(
        self,
        agent_name: str,
        input_summary: str,
        output_summary: str,
        ticket_id: str,
    ) -> None:
        ### use of this function: log agent step
        """Log a single agent step in readable format."""
        self._entries.append(
            f"\n[{agent_name.upper()}]\n"
            f"  {output_summary}\n"
        )

    def log_final_output(
        self,
        ticket_id: str,
        action: str,
        response: str,
        sources: list[str],
    ) -> None:
        ### use of this function: log final output
        """Log the final output for a ticket."""
        from utils.safety import redact_pii
        redacted_response = redact_pii(response)
        sources_str = ", ".join(sources) if sources else "N/A"
        self._entries.append(
            f"\n[FINAL OUTPUT]\n"
            f"  Action: {action}\n"
            f"  Response: {redacted_response}\n"
            f"  Sources: {sources_str}\n"
            f"\n{'=' * 80}\n"
        )

    def save(self, path: str) -> None:
        ### use of this function: save
        """Write the full transcript log to disk."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("MULTI-DOMAIN SUPPORT TRIAGE AGENT — SESSION LOG\n")
            f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write("=" * 80 + "\n")
            for entry in self._entries:
                f.write(entry)
