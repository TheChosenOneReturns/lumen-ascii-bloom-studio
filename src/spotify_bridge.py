from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

try:
    from winrt.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionManager as MediaManager,
        GlobalSystemMediaTransportControlsSessionPlaybackStatus as PlaybackStatus,
    )

    _WINRT_AVAILABLE = True
except Exception:
    MediaManager = None
    PlaybackStatus = None
    _WINRT_AVAILABLE = False


@dataclass
class NowPlayingInfo:
    connected: bool = False
    is_spotify: bool = False
    app_name: str = "NONE"
    title: str = "SIN REPRODUCCION ACTIVA"
    artist: str = "ABRE SPOTIFY O REPRODUCE AUDIO"
    album: str = ""
    is_playing: bool = False
    status_text: str = "STANDBY"


class SpotifyWindowsBridge:
    """
    Zero-config Windows GSMTC bridge for Spotify & system media players.
    Reads live Track / Artist / Album / Play state and sends Play/Pause/Next/Prev commands
    directly to Spotify Desktop (or active Windows media session) without requiring API keys!
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._info = NowPlayingInfo(
            status_text="LISTENING" if _WINRT_AVAILABLE else "WINRT UNAVAILABLE"
        )
        self._cmd_queue: "queue.Queue[str]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if _WINRT_AVAILABLE:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    @property
    def available(self) -> bool:
        return _WINRT_AVAILABLE

    def get_info(self) -> NowPlayingInfo:
        with self._lock:
            return NowPlayingInfo(
                connected=self._info.connected,
                is_spotify=self._info.is_spotify,
                app_name=self._info.app_name,
                title=self._info.title,
                artist=self._info.artist,
                album=self._info.album,
                is_playing=self._info.is_playing,
                status_text=self._info.status_text,
            )

    def toggle_play_pause(self) -> None:
        if _WINRT_AVAILABLE:
            self._cmd_queue.put("toggle")

    def next_track(self) -> None:
        if _WINRT_AVAILABLE:
            self._cmd_queue.put("next")

    def prev_track(self) -> None:
        if _WINRT_AVAILABLE:
            self._cmd_queue.put("prev")

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_poll())
        except Exception:
            pass
        finally:
            try:
                loop.close()
            except Exception:
                pass

    @staticmethod
    def _pick_session(manager):
        sessions = manager.get_sessions()
        spotify_sess = None
        playing_sess = None
        for s in sessions:
            app_id = (s.source_app_user_model_id or "").lower()
            try:
                pb = s.get_playback_info()
                is_playing = pb is not None and pb.playback_status == PlaybackStatus.PLAYING
            except Exception:
                is_playing = False
            if "spotify" in app_id:
                spotify_sess = s
                if is_playing:
                    return s
            if is_playing and playing_sess is None:
                playing_sess = s

        if spotify_sess is not None:
            return spotify_sess
        if playing_sess is not None:
            return playing_sess
        return manager.get_current_session()

    @staticmethod
    def _clean_app_name(app_id: str) -> tuple[str, bool]:
        low = (app_id or "").lower()
        if "spotify" in low:
            return "SPOTIFY", True
        if "chrome" in low or "msedge" in low or "firefox" in low or "brave" in low or "opera" in low:
            return "BROWSER AUDIO", False
        if "music" in low or "apple" in low or "tidal" in low or "foobar" in low or "vlc" in low:
            return app_id.split("!")[0].split(".")[-1].upper()[:14], False
        if not app_id:
            return "PC AUDIO", False
        base = app_id.replace(".exe", "").split("!")[0].split(".")[-1]
        return (base.upper()[:14] or "PC AUDIO"), False

    async def _async_poll(self) -> None:
        manager = None
        last_poll = 0.0
        while not self._stop_event.is_set():
            try:
                if manager is None:
                    manager = await MediaManager.request_async()

                # Process any pending transport commands immediately
                while True:
                    try:
                        cmd = self._cmd_queue.get_nowait()
                    except queue.Empty:
                        break
                    sess = self._pick_session(manager)
                    if sess is not None:
                        try:
                            if cmd == "toggle":
                                await sess.try_toggle_play_pause_async()
                            elif cmd == "next":
                                await sess.try_skip_next_async()
                            elif cmd == "prev":
                                await sess.try_skip_previous_async()
                        except Exception:
                            pass
                        last_poll = 0.0  # Force immediate metadata refresh

                now = time.perf_counter()
                if now - last_poll >= 0.65:
                    last_poll = now
                    sess = self._pick_session(manager)
                    if sess is None:
                        new_info = NowPlayingInfo(
                            connected=False,
                            is_spotify=False,
                            app_name="PC LOOPBACK",
                            title="ESCUCHANDO AUDIO DEL SISTEMA",
                            artist="REPRODUCE SPOTIFY / YOUTUBE / MUSICA",
                            album="",
                            is_playing=False,
                            status_text="IDLE",
                        )
                    else:
                        app_id = sess.source_app_user_model_id or ""
                        app_name, is_spot = self._clean_app_name(app_id)
                        props = await sess.try_get_media_properties_async()
                        pb = sess.get_playback_info()
                        is_playing = bool(pb is not None and pb.playback_status == PlaybackStatus.PLAYING)
                        title = (props.title or "").strip() if props else ""
                        artist = (props.artist or "").strip() if props else ""
                        album = (props.album_title or "").strip() if props else ""
                        new_info = NowPlayingInfo(
                            connected=True,
                            is_spotify=is_spot,
                            app_name=app_name,
                            title=title or "AUDIO ACTIVO",
                            artist=artist or app_name,
                            album=album,
                            is_playing=is_playing,
                            status_text="PLAYING" if is_playing else "PAUSED",
                        )
                    with self._lock:
                        self._info = new_info
            except Exception as exc:
                with self._lock:
                    self._info.status_text = f"ERR: {str(exc)[:24]}"
                manager = None
                await asyncio.sleep(1.5)
                continue

            await asyncio.sleep(0.08)

    def close(self) -> None:
        self._stop_event.set()
