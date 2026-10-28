"""The "a newer release is available" notice under the language picker (app/update_check.py).

The check runs once per server process, in the background, so no page load waits on the network;
a found update shows on a later rerun. Nothing is checked while UPDATE_FEED_URL is empty."""
from __future__ import annotations

import streamlit as st

from app import __version__, update_check
from app.i18n import t


@st.cache_resource(show_spinner=False, max_entries=1)
def _update_watch() -> dict:
    """The dict gets "info" once the background check finds a newer release."""
    found: dict = {}
    update_check.check_in_background(__version__, lambda info: found.setdefault("info", info))
    return found


def render_update_notice() -> None:
    info = _update_watch().get("info")
    if info is not None:
        st.info(t("Photo Studio {0} is available.").format(info.version))
        st.link_button(t("Download the update"), info.url)
