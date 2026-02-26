import json
import sys
import time
from typing import Any, Dict


class EventEmitter:
    def __init__(self, source: str = "mods_worker") -> None:
        self.source = source
        self.sequence = 0

    def emit(self, event: Dict[str, Any]) -> None:
        self.sequence += 1
        payload = {
            "schema_version": "v1",
            "sequence": self.sequence,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": self.source,
            "event": event,
        }
        print(json.dumps(payload), flush=True)

    def info(self, message: str) -> None:
        self.emit({"type": "log", "level": "info", "message": message})

    def warning(self, code: str, message: str) -> None:
        self.emit({"type": "warning", "code": code, "message": message})

    def error(self, code: str, message: str, recoverable: bool = False, details: Dict[str, Any] | None = None) -> None:
        event: Dict[str, Any] = {
            "type": "error",
            "code": code,
            "message": message,
            "recoverable": recoverable,
        }
        if details:
            event["details"] = details
        self.emit(event)


def fatal(message: str, exit_code: int = 1) -> None:
    print(
        json.dumps(
            {
                "schema_version": "v1",
                "sequence": 1,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": "mods_worker",
                "event": {
                    "type": "error",
                    "code": "WORKER_INTERNAL_ERROR",
                    "message": message,
                    "recoverable": False,
                },
            }
        ),
        flush=True,
    )
    sys.exit(exit_code)
