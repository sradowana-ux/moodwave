"""
Spotify integration for Moodwave.

Uses the Client Credentials flow (app-only auth — no user login/redirect
needed) and the still-supported /v1/search endpoint to find tracks that
match a detected mood.

NOTE: Spotify deprecated the /recommendations and /audio-features endpoints
for any app created after 27 Nov 2024 (see Spotify's Nov 2024 developer
changelog). Only apps with "Extended Quota Mode" still have access to those.
This module deliberately avoids them and uses /search instead, which remains
available to every app.

Requires two environment variables (set as Space secrets, never hard-coded):
    SPOTIFY_CLIENT_ID
    SPOTIFY_CLIENT_SECRET
"""

import os
import random
import time

import requests

TOKEN_URL = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"

# Hand-picked free-text search phrases per mood. Plain free-text search
# against the track index is more reliable than Spotify's `genre:` field
# filter, which is documented as inconsistent for track-type search.
MOOD_QUERIES = {
    "joy": [
        "happy upbeat pop",
        "feel good dance hits",
        "uplifting summer anthems",
        "energetic feel-good songs",
    ],
    "sadness": [
        "sad acoustic ballads",
        "melancholy piano songs",
        "heartbreak slow songs",
        "rainy day sad songs",
    ],
    "anger": [
        "aggressive rock anthems",
        "angry metal songs",
        "rage rap",
        "intense hard rock",
    ],
    "fear": [
        "dark ambient tense",
        "eerie atmospheric soundtrack",
        "anxious moody electronic",
        "suspenseful score",
    ],
    "neutral": [
        "chill lofi beats",
        "calm instrumental",
        "relaxing background music",
        "easy listening chill",
    ],
}

MOOD_EMOJI = {
    "joy": "🟢",
    "sadness": "🔵",
    "anger": "🔴",
    "fear": "🟣",
    "neutral": "⚪",
}

_token_cache = {"access_token": None, "expires_at": 0.0}


def spotify_configured() -> bool:
    return bool(os.environ.get("SPOTIFY_CLIENT_ID") and os.environ.get("SPOTIFY_CLIENT_SECRET"))


def _get_app_token():
    """Fetch (and cache) an app-only access token via Client Credentials."""
    if _token_cache["access_token"] and time.time() < _token_cache["expires_at"] - 30:
        return _token_cache["access_token"]

    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    resp = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    _token_cache["access_token"] = data["access_token"]
    _token_cache["expires_at"] = time.time() + data.get("expires_in", 3600)
    return _token_cache["access_token"]


def search_tracks_for_mood(mood: str, n: int = 5):
    """
    Return up to `n` tracks matching the given mood via /v1/search.
    Each track is a dict: {name, artists, url, embed_url, image}.
    Raises RuntimeError with a human-readable message on failure
    (e.g. missing secrets, expired token).
    """
    mood = (mood or "neutral").lower()
    if mood not in MOOD_QUERIES:
        mood = "neutral"

    token = _get_app_token()
    if not token:
        raise RuntimeError(
            "Spotify isn't connected yet — add SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET in this Space's Settings → Variables and secrets."
        )

    query = random.choice(MOOD_QUERIES[mood])
    resp = requests.get(
        SEARCH_URL,
        headers={"Authorization": f"Bearer {token}"},
        params={"q": query, "type": "track", "limit": 20, "market": "US"},
        timeout=10,
    )
    if resp.status_code == 401:
        _token_cache["access_token"] = None
        raise RuntimeError("Spotify token expired — click ANALYZE + RECOMMEND again.")
    if resp.status_code == 403:
        raise RuntimeError("Spotify rejected the request (check the app's Client ID/Secret are correct).")
    resp.raise_for_status()

    items = resp.json().get("tracks", {}).get("items", []) or []
    if not items:
        return []

    random.shuffle(items)
    chosen = items[:n]

    tracks = []
    for t in chosen:
        images = t.get("album", {}).get("images", [])
        tracks.append(
            {
                "name": t.get("name", "Unknown"),
                "artists": ", ".join(a["name"] for a in t.get("artists", [])),
                "url": t.get("external_urls", {}).get("spotify", "#"),
                "embed_url": f"https://open.spotify.com/embed/track/{t['id']}",
                "image": images[-1]["url"] if images else "",
            }
        )
    return tracks


def tracks_to_html(tracks, mood: str) -> str:
    """Render a list of track dicts as a column of Spotify embed players."""
    if not tracks:
        return "<p>No tracks found — try again.</p>"

    cards = "\n".join(
        f'<iframe style="border-radius:12px" src="{t["embed_url"]}" '
        f'width="100%" height="152" frameBorder="0" allowfullscreen="" '
        f'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" '
        f'loading="lazy"></iframe>'
        for t in tracks
    )
    return f'<div style="display:flex;flex-direction:column;gap:10px;">{cards}</div>'
