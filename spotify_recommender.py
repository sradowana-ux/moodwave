"""
Spotify Mood Recommender
========================
Connects the MER pipeline to Spotify's API to recommend tracks matching
the predicted mood using Spotify's audio feature space (valence, energy, etc.).

Setup (one-time):
    1. Create a Spotify app at https://developer.spotify.com/dashboard
    2. Set Redirect URI to http://localhost:8888/callback
    3. Export your credentials:
         export SPOTIFY_CLIENT_ID="your_client_id"
         export SPOTIFY_CLIENT_SECRET="your_client_secret"
    4. pip install spotipy

Usage:
    from spotify_recommender import MoodRecommender
    rec = MoodRecommender()

    # From text
    tracks = rec.recommend_from_text("I feel exhausted and defeated today")
    for t in tracks:
        print(t["name"], "—", t["artist"], t["url"])

    # From audio file
    tracks = rec.recommend_from_audio("speech.wav")

    # From pre-computed mood
    tracks = rec.recommend_from_mood("joy", n=5)
"""

import os
from pathlib import Path
from typing import Optional

# ── Mood → Spotify audio feature targets ─────────────────────────────────
# Each mood maps to a range of Spotify "audio features":
#   valence  : musical positiveness (0 = sad/tense, 1 = happy/euphoric)
#   energy   : intensity and activity (0 = calm, 1 = loud/fast)
#   tempo    : approx BPM range [min, max]
#   mode     : 1 = major, 0 = minor, None = no constraint
#
# Sources: Spotify audio features documentation + MER literature on
# valence-arousal mappings (Russell, 1980; Thayer, 1989).

MOOD_AUDIO_FEATURES = {
    "joy": {
        "target_valence": 0.80,
        "target_energy":  0.75,
        "min_valence":    0.55,
        "min_energy":     0.45,
        "target_tempo":   120.0,
        "target_mode":    1,
    },
    "sadness": {
        "target_valence": 0.20,
        "target_energy":  0.25,
        "max_valence":    0.45,
        "max_energy":     0.50,
        "target_tempo":   75.0,
        "target_mode":    0,
    },
    "anger": {
        "target_valence": 0.25,
        "target_energy":  0.85,
        "max_valence":    0.50,
        "min_energy":     0.60,
        "target_tempo":   135.0,
        "target_mode":    0,
    },
    "fear": {
        "target_valence": 0.20,
        "target_energy":  0.55,
        "max_valence":    0.45,
        "min_energy":     0.30,
        "max_energy":     0.75,
        "target_tempo":   95.0,
        "target_mode":    0,
    },
    "neutral": {
        "target_valence": 0.50,
        "target_energy":  0.45,
        "min_valence":    0.35,
        "max_valence":    0.65,
        "target_tempo":   100.0,
    },
}

# Seed artists per mood (Spotify artist IDs) — used to bias recommendations
# toward genres that tend to match each mood.  All are mainstream / cross-genre.
MOOD_SEED_ARTISTS = {
    "joy":     ["06HL4z0CvFAxyc27GXpf02",  # Taylor Swift
                "1Xyo4u8uXC1ZmMpatF05PJ"],  # The Weeknd
    "sadness": ["4dpARuHxo51G3z768sgnrY",  # Adele
                "0du5cEVh5yTK9QJze8zA0C"],  # Bruno Mars (slow tracks)
    "anger":   ["36QJpDe2go2KgaRleHCDTp",  # Linkin Park
                "0L8ExT028jH3ddEcZwqJJ5"],  # Red Hot Chili Peppers
    "fear":    ["0LyfQWLjG2wT0CQbxcJXMo",  # Billie Eilish
                "4YRxDV8wJFPHPTeXepOstw"],  # Radiohead
    "neutral": ["53XhwfbYqKCa1cC15pYq2q",  # Imagine Dragons
                "3TVXtAsR1Inumwj472S9r4"],  # Drake
}


class MoodRecommender:
    """
    Recommends Spotify tracks based on predicted emotional mood.

    Parameters
    ----------
    client_id     : Spotify API client ID (defaults to SPOTIFY_CLIENT_ID env var)
    client_secret : Spotify API client secret (defaults to SPOTIFY_CLIENT_SECRET env var)
    redirect_uri  : OAuth redirect URI (default: http://localhost:8888/callback)
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: str = "http://localhost:8888/callback",
    ):
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyOAuth
        except ImportError:
            raise ImportError(
                "spotipy is required: pip install spotipy"
            )

        self._client_id     = client_id or os.environ["SPOTIFY_CLIENT_ID"]
        self._client_secret = client_secret or os.environ["SPOTIFY_CLIENT_SECRET"]

        self.sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=self._client_id,
                client_secret=self._client_secret,
                redirect_uri=redirect_uri,
                scope="user-read-playback-state user-modify-playback-state",
                cache_path=".spotify_token_cache",
                open_browser=True,
            )
        )
        # Lazy-load the MER models only when needed
        self._text_predictor  = None
        self._audio_predictor = None

    # ── MER model loading ─────────────────────────────────────────────────
    def _load_text_predictor(self):
        if self._text_predictor is not None:
            return
        from transformers import pipeline as hf_pipeline

        finetune_dir = Path("MER_Project/text_finetuned_model/best")
        zero_shot    = "SamLowe/roberta-base-go_emotions"

        if finetune_dir.exists():
            self._text_predictor = hf_pipeline(
                "text-classification", model=str(finetune_dir),
                tokenizer=str(finetune_dir), return_all_scores=True,
                truncation=True, max_length=128,
            )
            self._text_mode = "finetuned"
        else:
            self._text_predictor = hf_pipeline(
                "text-classification", model=zero_shot,
                return_all_scores=True, truncation=True, max_length=128,
            )
            self._text_mode = "zero-shot"

    def _load_audio_predictor(self):
        if self._audio_predictor is not None:
            return
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

        model_dir = Path("MER_Project/audio_final_model/best_model")
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Audio model not found at {model_dir}. "
                "Run the training notebook first."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._audio_fe    = AutoFeatureExtractor.from_pretrained(str(model_dir))
        self._audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(str(model_dir))
        self._audio_model.eval().to(device)
        self._audio_device = device

    # ── Mood prediction ───────────────────────────────────────────────────
    def _predict_text_mood(self, text: str) -> str:
        self._load_text_predictor()
        TARGET_LABELS = ["anger", "fear", "joy", "neutral", "sadness"]
        MOOD_MAP_LOCAL = {
            "admiration": "joy",   "amusement": "joy",   "anger": "anger",
            "annoyance": "anger",  "approval": "joy",    "caring": "joy",
            "confusion": "neutral","curiosity": "neutral","desire": "joy",
            "disappointment": "sadness","disapproval": "anger","disgust": "anger",
            "embarrassment": "sadness","excitement": "joy","fear": "fear",
            "gratitude": "joy",   "grief": "sadness",   "joy": "joy",
            "love": "joy",        "nervousness": "fear", "optimism": "joy",
            "pride": "joy",       "realization": "neutral","relief": "joy",
            "remorse": "sadness", "sadness": "sadness",  "surprise": "neutral",
            "neutral": "neutral",
        }
        raw = self._text_predictor(text)[0]
        if self._text_mode == "finetuned":
            scores = {item["label"]: item["score"] for item in raw}
        else:
            scores = {m: 0.0 for m in TARGET_LABELS}
            for item in raw:
                m = MOOD_MAP_LOCAL.get(item["label"])
                if m:
                    scores[m] += item["score"]
            total = sum(scores.values()) or 1.0
            scores = {m: v / total for m, v in scores.items()}
        return max(scores, key=scores.get)

    def _predict_audio_mood(self, audio_path: str) -> str:
        import torch, librosa, numpy as np
        self._load_audio_predictor()
        waveform, _ = librosa.load(audio_path, sr=16_000, mono=True)
        waveform = waveform[:96_000]
        inputs = self._audio_fe(
            waveform, sampling_rate=16_000, return_tensors="pt", padding=True
        ).to(self._audio_device)
        with torch.no_grad():
            logits = self._audio_model(**inputs).logits
        idx = int(torch.argmax(logits, dim=-1).item())
        id2label_local = {0: "anger", 1: "fear", 2: "joy", 3: "neutral", 4: "sadness"}
        return id2label_local[idx]

    # ── Spotify recommendation ────────────────────────────────────────────
    def recommend_from_mood(self, mood: str, n: int = 5) -> list[dict]:
        """
        Return n Spotify track recommendations for the given mood.

        Parameters
        ----------
        mood : one of anger / fear / joy / neutral / sadness
        n    : number of tracks to return (max 10 from a single recommendation call)

        Returns
        -------
        list of dicts with keys: name, artist, album, url, preview_url
        """
        if mood not in MOOD_AUDIO_FEATURES:
            raise ValueError(f"Unknown mood: {mood}. Choose from {list(MOOD_AUDIO_FEATURES)}")

        features = MOOD_AUDIO_FEATURES[mood]
        seed_artists = MOOD_SEED_ARTISTS.get(mood, [])[:2]

        recommendations = self.sp.recommendations(
            seed_artists=seed_artists,
            limit=min(n, 10),
            **features,
        )

        tracks = []
        for item in recommendations["tracks"]:
            tracks.append({
                "name":        item["name"],
                "artist":      ", ".join(a["name"] for a in item["artists"]),
                "album":       item["album"]["name"],
                "url":         item["external_urls"]["spotify"],
                "preview_url": item.get("preview_url"),
                "mood":        mood,
            })
        return tracks

    def recommend_from_text(self, text: str, n: int = 5) -> list[dict]:
        """Predict mood from text and return track recommendations."""
        mood = self._predict_text_mood(text)
        print(f"Predicted mood: {mood}")
        return self.recommend_from_mood(mood, n=n)

    def recommend_from_audio(self, audio_path: str, n: int = 5) -> list[dict]:
        """Predict mood from audio file and return track recommendations."""
        mood = self._predict_audio_mood(audio_path)
        print(f"Predicted mood: {mood}")
        return self.recommend_from_mood(mood, n=n)

    def recommend_fused(
        self,
        text: Optional[str] = None,
        audio_path: Optional[str] = None,
        text_weight: float = 0.2,
        audio_weight: float = 0.8,
        n: int = 5,
    ) -> list[dict]:
        """
        Predict mood from text + audio with late fusion, then recommend tracks.
        Falls back to single-modality if only one is provided.
        """
        TARGET_LABELS = ["anger", "fear", "joy", "neutral", "sadness"]

        if text is None and audio_path is None:
            raise ValueError("Provide at least one of text or audio_path.")

        # Build probability distributions for available modalities
        text_scores  = None
        audio_scores = None

        if text:
            self._load_text_predictor()
            text_mood = self._predict_text_mood(text)
            # Simple one-hot-ish: put all weight on predicted mood for fusion
            # (for a richer fusion, extract the full probability vector)
            text_scores = {m: (1.0 if m == text_mood else 0.0) for m in TARGET_LABELS}

        if audio_path:
            audio_mood = self._predict_audio_mood(audio_path)
            audio_scores = {m: (1.0 if m == audio_mood else 0.0) for m in TARGET_LABELS}

        if text_scores and audio_scores:
            fused = {m: text_weight * text_scores[m] + audio_weight * audio_scores[m]
                     for m in TARGET_LABELS}
            mood = max(fused, key=fused.get)
        elif text_scores:
            mood = max(text_scores, key=text_scores.get)
        else:
            mood = max(audio_scores, key=audio_scores.get)

        print(f"Fused mood: {mood}")
        return self.recommend_from_mood(mood, n=n)

    # ── Pretty printer ────────────────────────────────────────────────────
    @staticmethod
    def print_tracks(tracks: list[dict]):
        MOOD_EMOJI = {"anger": "😠", "fear": "😨", "joy": "😊",
                      "neutral": "😐", "sadness": "😢"}
        if not tracks:
            print("No tracks returned.")
            return
        mood = tracks[0].get("mood", "")
        print(f"\n🎵 Recommendations for mood: {MOOD_EMOJI.get(mood,'')} {mood.upper()}")
        print("─" * 55)
        for i, t in enumerate(tracks, 1):
            print(f"{i}. {t['name']}")
            print(f"   {t['artist']} · {t['album']}")
            print(f"   {t['url']}")
            if t.get("preview_url"):
                print(f"   Preview: {t['preview_url']}")
        print("─" * 55)


# ── CLI quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("Spotify Mood Recommender — quick test")
    print("Ensure SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET are set.\n")

    rec = MoodRecommender()

    text_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "I feel absolutely devastated, everything went wrong today."

    print(f"Input text: '{text_input}'")
    tracks = rec.recommend_from_text(text_input, n=5)
    MoodRecommender.print_tracks(tracks)
