from __future__ import annotations

from pathlib import Path
import uuid

from . import constants as cs
from .context import app_context


def init_session_log(project_root: Path) -> Path:
    log_dir = project_root / cs.TMP_DIR
    log_dir.mkdir(exist_ok=True)
    app_context.session.log_file = (
        log_dir / f"{cs.SESSION_LOG_PREFIX}{uuid.uuid4().hex[:8]}{cs.SESSION_LOG_EXT}"
    )
    with open(app_context.session.log_file, "w") as f:
        f.write(cs.SESSION_LOG_HEADER)
    return app_context.session.log_file


def log_session_event(event: str) -> None:
    if app_context.session.log_file:
        with open(app_context.session.log_file, "a") as f:
            f.write(f"{event}\n")


def get_session_context() -> str:
    if app_context.session.log_file and app_context.session.log_file.exists():
        content = app_context.session.log_file.read_text(encoding="utf-8")
        return f"{cs.SESSION_CONTEXT_START}{content}{cs.SESSION_CONTEXT_END}"
    return ""

