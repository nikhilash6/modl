import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

from mods_worker.protocol import EventEmitter

_STEP_RE = re.compile(r"step\s*[:=]?\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_LOSS_RE = re.compile(r"loss\s*[:=]?\s*([0-9eE+\-.]+)", re.IGNORECASE)


def _build_train_command(config_path: Path) -> List[str]:
    env_cmd = os.getenv("MODS_AITOOLKIT_TRAIN_CMD", "").strip()
    if env_cmd:
        env_cmd = env_cmd.replace("{config}", str(config_path)).replace("{python}", sys.executable)
        return shlex.split(env_cmd)

    return [sys.executable, "-m", "toolkit.job", "--config", str(config_path)]


def run_train(config_path: Path, emitter: EventEmitter) -> int:
    if not config_path.exists():
        emitter.error(
            "SPEC_VALIDATION_FAILED",
            f"Training config not found: {config_path}",
            recoverable=False,
        )
        return 2

    cmd = _build_train_command(config_path)
    emitter.emit({"type": "job_started", "config": str(config_path), "command": cmd})

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        emitter.error(
            "AITOOLKIT_EXEC_NOT_FOUND",
            f"Could not execute ai-toolkit command: {exc}",
            recoverable=False,
        )
        return 127
    except Exception as exc:
        emitter.error(
            "AITOOLKIT_EXEC_FAILED",
            str(exc),
            recoverable=False,
        )
        return 1

    last_step = None
    for raw_line in process.stdout or []:
        line = raw_line.strip()
        if not line:
            continue

        emitter.info(line)

        step_match = _STEP_RE.search(line)
        if step_match:
            step = int(step_match.group(1))
            total_steps = int(step_match.group(2))
            if last_step != step:
                event = {
                    "type": "progress",
                    "stage": "train",
                    "step": step,
                    "total_steps": total_steps,
                }
                loss_match = _LOSS_RE.search(line)
                if loss_match:
                    try:
                        event["loss"] = float(loss_match.group(1))
                    except ValueError:
                        pass
                emitter.emit(event)
                last_step = step

    code = process.wait()
    if code == 0:
        emitter.emit({"type": "completed", "message": "ai-toolkit training command finished"})
    else:
        emitter.error(
            "TRAINING_FAILED",
            f"ai-toolkit process exited with code {code}",
            recoverable=False,
            details={"exit_code": code},
        )
    return code
