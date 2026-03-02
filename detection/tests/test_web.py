"""Tests for the Tornado web application (web.py).

Uses tornado.testing.AsyncHTTPTestCase with mocked dependencies so no camera,
model, or filesystem is required.

Run: python -m pytest detection/tests/test_web.py   (from repo root)
  or: cd detection && python -m pytest tests/test_web.py
"""

import json
import queue as _queue
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tornado.testing import AsyncHTTPTestCase

# Allow importing detection modules directly when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import web
from detector import EventQueue, FrameBuffer


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _make_jpeg_bytes() -> bytes:
    """Return the minimal valid JFIF JPEG header (2×2 grey image)."""
    # Tiny but valid JPEG
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
        b"\xff\xc0\x00\x0b\x08\x00\x02\x00\x02\x01\x01\x11\x00"
        b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf5\xff\xd9"
    )


class _FakeStatsMonitor:
    def get(self):
        return {"tile_fps": 10.0, "frame_ms": 5.0, "stream_fps": 30.0, "stream_res": "640x480", "n_tiles": 2}


def _make_app(tmp_path: Path, frame_buffer=None, event_queue=None):
    if frame_buffer is None:
        frame_buffer = FrameBuffer()
    if event_queue is None:
        event_queue = EventQueue()

    reviews_path = tmp_path / "reviews.json"
    reviews_lock = threading.Lock()
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    templates_dir = Path(__file__).parent.parent / "templates"
    stats_monitor = _FakeStatsMonitor()

    return web.make_app(
        frame_buffer, reviews_path, reviews_lock,
        clips_dir, templates_dir, stats_monitor, event_queue,
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestIndexHandler(AsyncHTTPTestCase):
    def get_app(self):
        import tempfile
        self._tmp = Path(tempfile.mkdtemp())
        return _make_app(self._tmp)

    def test_index_returns_200_html(self):
        resp = self.fetch("/")
        self.assertEqual(resp.code, 200)
        self.assertIn("text/html", resp.headers.get("Content-Type", ""))


class TestReviewHandler(AsyncHTTPTestCase):
    def get_app(self):
        import tempfile
        self._tmp = Path(tempfile.mkdtemp())
        return _make_app(self._tmp)

    def test_review_returns_200_html(self):
        resp = self.fetch("/review")
        self.assertEqual(resp.code, 200)
        self.assertIn("text/html", resp.headers.get("Content-Type", ""))


class TestStreamHandler(AsyncHTTPTestCase):
    def get_app(self):
        import tempfile
        self._tmp = Path(tempfile.mkdtemp())
        fb = FrameBuffer()
        # Pre-load a JPEG frame so the handler writes at least one chunk
        fb.frame = _make_jpeg_bytes()
        fb.event.set()
        self._fb = fb
        return _make_app(self._tmp, frame_buffer=fb)

    def test_stream_content_type(self):
        # Fetch with a short streaming window; connection will close after timeout
        resp = self.fetch("/stream", raise_error=False)
        ct = resp.headers.get("Content-Type", "")
        self.assertIn("multipart/x-mixed-replace", ct)

    def test_stream_contains_jpeg_frame(self):
        resp = self.fetch("/stream", raise_error=False)
        self.assertIn(b"--frame", resp.body)
        self.assertIn(b"image/jpeg", resp.body)


class TestClipsHandler(AsyncHTTPTestCase):
    def get_app(self):
        import tempfile
        self._tmp = Path(tempfile.mkdtemp())
        return _make_app(self._tmp)

    def test_clips_empty(self):
        resp = self.fetch("/clips")
        self.assertEqual(resp.code, 200)
        self.assertEqual(json.loads(resp.body), [])

    def test_clips_returns_entries(self):
        # Write a fake reviews.json
        reviews_path = self._tmp / "reviews.json"
        entry = {"id": "abc123", "path": "/tmp/clip.mp4", "reviewed": False}
        reviews_path.write_text(json.dumps([entry]))

        resp = self.fetch("/clips")
        self.assertEqual(resp.code, 200)
        data = json.loads(resp.body)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["id"], "abc123")


class TestClipReviewHandler(AsyncHTTPTestCase):
    def get_app(self):
        import tempfile
        self._tmp = Path(tempfile.mkdtemp())
        reviews_path = self._tmp / "reviews.json"
        entry = {"id": "abc123", "path": "/tmp/clip.mp4", "reviewed": False}
        reviews_path.write_text(json.dumps([entry]))
        return _make_app(self._tmp)

    def test_mark_reviewed_returns_204(self):
        resp = self.fetch("/clips/abc123/review", method="POST", body="")
        self.assertEqual(resp.code, 204)

    def test_mark_reviewed_persists(self):
        self.fetch("/clips/abc123/review", method="POST", body="")
        reviews_path = self._tmp / "reviews.json"
        data = json.loads(reviews_path.read_text())
        self.assertTrue(data[0]["reviewed"])


class TestStatsHandler(AsyncHTTPTestCase):
    def get_app(self):
        import tempfile
        self._tmp = Path(tempfile.mkdtemp())
        return _make_app(self._tmp)

    def test_stats_returns_200_json(self):
        resp = self.fetch("/stats")
        self.assertEqual(resp.code, 200)
        data = json.loads(resp.body)
        self.assertIn("tile_fps", data)
        self.assertIn("frame_ms", data)
        self.assertIn("stream_fps", data)


class TestEventsHandler(AsyncHTTPTestCase):
    def get_app(self):
        import tempfile
        self._tmp = Path(tempfile.mkdtemp())
        self._eq = EventQueue()
        return _make_app(self._tmp, event_queue=self._eq)

    def test_events_content_type(self):
        # Push an event before connecting so the handler has something to write
        clip = {"id": "evt1", "path": "/tmp/clip.mp4", "reviewed": False}
        # The handler polls every 0.3 s; we schedule the push so it arrives
        # while the fetch is streaming. For test simplicity we pre-enqueue by
        # directly populating a subscriber queue before the request is made.
        q = _queue.SimpleQueue()
        q.put(clip)
        with self._eq._lock:
            self._eq._subscribers.append(q)

        resp = self.fetch("/events", raise_error=False)
        ct = resp.headers.get("Content-Type", "")
        self.assertIn("text/event-stream", ct)

    def test_events_sends_data_line(self):
        clip = {"id": "evt2", "path": "/tmp/clip.mp4", "reviewed": False}
        q = _queue.SimpleQueue()
        q.put(clip)
        with self._eq._lock:
            self._eq._subscribers.append(q)

        resp = self.fetch("/events", raise_error=False)
        self.assertIn(b"data:", resp.body)
        # Verify the embedded JSON contains the expected id
        for line in resp.body.decode().splitlines():
            if line.startswith("data:"):
                payload = json.loads(line[len("data:"):].strip())
                self.assertEqual(payload["id"], "evt2")
                break


if __name__ == "__main__":
    unittest.main()
