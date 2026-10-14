"""Bounded CPU decode prefetch; one thread owns the capture's read calls."""
from __future__ import annotations

import queue
import threading


def bounded_batch_size(frame, requested, budget_bytes=128 * 1024 * 1024):
    if not 1 <= requested <= 8:
        raise ValueError("Dog frame batch must be 1–8")
    height, width, channels = frame.shape
    # Consumer batch, two queued batches and producer batch can coexist.
    return max(1, min(requested, budget_bytes // max(1, height * width * channels * 4)))


def decoded_batches(capture, first, requested, timings):
    size = bounded_batch_size(first, requested)
    pending = queue.Queue(maxsize=2)
    stop = threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                pending.put(item, timeout=.1)
                return True
            except queue.Full:
                pass
        return False

    def decode():
        import time
        batch = [first]
        try:
            while not stop.is_set():
                if len(batch) == size:
                    if not put((batch, None)):
                        return
                    batch = []
                start = time.perf_counter()
                ok, frame = capture.read()
                timings["decode_seconds"] += time.perf_counter() - start
                if not ok:
                    if batch:
                        put((batch, None))
                    put((None, None))
                    return
                if frame.shape != first.shape:
                    raise ValueError("Videos with changing frame dimensions are unsupported.")
                batch.append(frame)
        except Exception as exc:
            put((None, exc))

    producer = threading.Thread(target=decode, name="video-decode", daemon=True)
    producer.start()
    try:
        while True:
            batch, error = pending.get()
            if error:
                raise error
            if batch is None:
                return
            yield batch
    finally:
        stop.set()
        producer.join(timeout=10)
        if producer.is_alive():
            raise RuntimeError("Video decoder did not stop in time")