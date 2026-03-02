"""Tornado web application: MJPEG streaming, detection review UI, and SSE events."""

import asyncio
import json
import threading
from pathlib import Path

import tornado.iostream
import tornado.web

import reviews as reviews_mod


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


class StreamHandler(tornado.web.RequestHandler):
    def initialize(self, frame_buffer):
        self._frame_buffer = frame_buffer

    async def get(self):
        self.set_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        try:
            while True:
                frame = self._frame_buffer.get_or_none()
                if frame is not None:
                    self.write(b"--frame\r\n")
                    self.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.write(frame)
                    self.write(b"\r\n")
                    await self.flush()
                await asyncio.sleep(0.033)
        except tornado.iostream.StreamClosedError:
            pass


class ReviewHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("review.html")


class ClipsHandler(tornado.web.RequestHandler):
    def initialize(self, reviews_path, reviews_lock):
        self._reviews_path = reviews_path
        self._reviews_lock = reviews_lock

    def get(self):
        with self._reviews_lock:
            data = reviews_mod.load(self._reviews_path)
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(data))


class StatsHandler(tornado.web.RequestHandler):
    def initialize(self, stats_monitor):
        self._monitor = stats_monitor

    def get(self):
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(self._monitor.get()))


class ClipReviewHandler(tornado.web.RequestHandler):
    def initialize(self, reviews_path, reviews_lock):
        self._reviews_path = reviews_path
        self._reviews_lock = reviews_lock

    def post(self, clip_id):
        with self._reviews_lock:
            entries = reviews_mod.load(self._reviews_path)
            for entry in entries:
                if entry["id"] == clip_id:
                    entry["reviewed"] = True
                    break
            reviews_mod.save(self._reviews_path, entries)
        self.set_status(204)


class EventsHandler(tornado.web.RequestHandler):
    """Server-Sent Events endpoint. Streams clip-saved events to connected clients."""

    def initialize(self, event_queue):
        self._event_queue = event_queue

    async def get(self):
        self.set_header("Content-Type", "text/event-stream")
        self.set_header("Cache-Control", "no-cache")
        self.set_header("X-Accel-Buffering", "no")  # disable nginx buffering
        q = self._event_queue.subscribe()
        try:
            while True:
                try:
                    event = q.get_nowait()
                    self.write(f"data: {json.dumps(event)}\n\n")
                    await self.flush()
                except Exception:
                    pass
                await asyncio.sleep(0.3)
        except tornado.iostream.StreamClosedError:
            pass
        finally:
            self._event_queue.unsubscribe(q)


def make_app(frame_buffer, reviews_path: Path, reviews_lock: threading.Lock,
             clips_dir: Path, templates_dir: Path, stats_monitor,
             event_queue) -> tornado.web.Application:
    shared = dict(reviews_path=reviews_path, reviews_lock=reviews_lock)
    return tornado.web.Application(
        [
            (r"/", IndexHandler),
            (r"/stream", StreamHandler, {"frame_buffer": frame_buffer}),
            (r"/review", ReviewHandler),
            (r"/clips", ClipsHandler, shared),
            (r"/clips/(.+)/review", ClipReviewHandler, shared),
            (r"/clips/files/(.*)", tornado.web.StaticFileHandler, {"path": str(clips_dir)}),
            (r"/stats", StatsHandler, {"stats_monitor": stats_monitor}),
            (r"/events", EventsHandler, {"event_queue": event_queue}),
        ],
        template_path=str(templates_dir),
    )
