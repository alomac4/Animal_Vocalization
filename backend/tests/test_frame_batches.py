import threading
import unittest
from types import SimpleNamespace

from backend.frame_batches import bounded_batch_size, decoded_batches


class Capture:
    def __init__(self, frames, failure=None):
        self.frames = iter(frames)
        self.failure = failure

    def read(self):
        if self.failure:
            raise self.failure
        frame = next(self.frames, None)
        return frame is not None, frame


class FrameBatchTests(unittest.TestCase):
    def test_prefetch_preserves_order_including_partial_final_batch(self):
        frames = [SimpleNamespace(shape=(100, 100, 3), index=i) for i in range(9)]
        timings = {"decode_seconds": 0}
        batches = list(decoded_batches(Capture(frames[1:]), frames[0], 4, timings))
        self.assertEqual([len(batch) for batch in batches], [4, 4, 1])
        self.assertEqual([frame.index for batch in batches for frame in batch], list(range(9)))
        self.assertGreaterEqual(timings["decode_seconds"], 0)

    def test_cpu_memory_budget_and_invalid_batch(self):
        frame = SimpleNamespace(shape=(4096, 4096, 3))
        self.assertEqual(bounded_batch_size(frame, 8), 1)
        self.assertEqual(bounded_batch_size(SimpleNamespace(shape=(480, 640, 3)), 4), 4)
        with self.assertRaises(ValueError):
            bounded_batch_size(frame, 9)

    def test_decode_failure_propagates_and_thread_exits(self):
        frame = SimpleNamespace(shape=(100, 100, 3))
        with self.assertRaisesRegex(ValueError, "broken decoder"):
            list(decoded_batches(Capture([], ValueError("broken decoder")), frame, 4, {"decode_seconds": 0}))
        self.assertFalse(any(thread.name == "video-decode" for thread in threading.enumerate()))

    def test_changing_dimensions_fail(self):
        first = SimpleNamespace(shape=(100, 100, 3))
        with self.assertRaisesRegex(ValueError, "changing frame dimensions"):
            list(decoded_batches(Capture([SimpleNamespace(shape=(110, 100, 3))]), first, 4, {"decode_seconds": 0}))

    def test_early_close_stops_prefetch_thread(self):
        frames = [SimpleNamespace(shape=(100, 100, 3)) for _ in range(100)]
        batches = decoded_batches(Capture(frames[1:]), frames[0], 4, {"decode_seconds": 0})
        next(batches)
        batches.close()
        self.assertFalse(any(thread.name == "video-decode" for thread in threading.enumerate()))