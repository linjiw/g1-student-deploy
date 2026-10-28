"""The one token of this deploy PC's operator surfaces: the console panel, the hub and Studio's deploy link.

G1_PANEL_TOKEN when it is set; otherwise ~/.config/g1-deploy/panel_token (created once, mode 0600). A restarted hub
then still reaches the consoles it adopts, and Studio's gateway reads the same file instead of passing the token
through the browser.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

TOKEN_FILE = Path(os.environ.get("G1_PANEL_TOKEN_FILE", Path.home() / ".config/g1-deploy/panel_token"))


def panel_token(create: bool = True) -> str | None:
    """The token: the environment's, else the file's (created with a fresh token when `create` and missing)."""
    env = os.environ.get("G1_PANEL_TOKEN")
    if env:
        return env
    try:
        text = TOKEN_FILE.read_text().strip()
        if text:
            return text
    except OSError:
        pass
    if not create:
        return None
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(12)
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    return token
