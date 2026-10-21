"""Hosted entry point: `streamlit run app.py`, or a file path on a host that runs one.

A hosted run needs two things the local command does not, and this file does both:

1. `src` on the import path, because the host installs from `requirements.txt` and
   never installs this package.
2. The API key moved from `st.secrets`, where a hosted dashboard puts it, into the
   environment, where `Settings` looks for it.

The page itself is `legalrag.inspector`, unchanged. This is a wrapper, not a second
implementation, so the deployed app and `make inspect` cannot drift apart.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from legalrag.inspector import main  # noqa: E402  (must follow the path insert)

# Locally there is no secrets file at all, which raises rather than returning an
# empty mapping. An empty `secrets` is the normal case on a developer machine.
try:
    configured = dict(st.secrets)
except Exception:
    configured = {}

# `setdefault` and not assignment: a variable already set in the environment wins,
# so a local `.env` is not silently overridden by a stale dashboard secret.
for name, value in configured.items():
    if isinstance(value, str):
        os.environ.setdefault(name, value)

main()
