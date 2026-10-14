"""Depth Pro dog-region sampling and isolated bounded-process client."""
from __future__ import annotations

import json
import math
import queue
import subprocess
import threading
from pathlib import Path


def metadata():
    return {"method": "apple_depth_pro", "approximate": True, "units": ["m", "ft"],
            "focal_source": "model_estimated_per_frame", "sampling": "central_box_median_axial_depth",
            "warning": "AI-estimated axial depth, not measured range. Box background/occlusion and camera estimation can cause errors; no dog masks or ground-truth calibration."}


def sample_box_depth(depth, box):
    """Robust axial-depth sample, avoiding box edges, with spatial validity gates.

    Not a segmentation mask: background may dominate. No false reliability score.
    Reject excessively mixed regions rather than report precise-looking garbage.
    """
    import numpy as np

    if depth.ndim != 2 or len(box) != 4 or not all(math.isfinite(v) for v in box):
        return None
    height, width = depth.shape
    x1,y1,x2,y2 = box
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height or min(x2-x1,y2-y1)<8:
        return None
    # Inner 40% width/height: lessen legs/ground/tail and neighboring dog leakage.
    a,b = math.floor(x1+.3*(x2-x1)),math.floor(y1+.3*(y2-y1))
    c,d = math.ceil(x2-.3*(x2-x1)),math.ceil(y2-.3*(y2-y1))
    region=depth[b:d,a:c]
    valid=np.isfinite(region)&(region>=.1)&(region<=200)
    if region.size<16 or valid.sum()<max(16, math.ceil(region.size*.7)):
        return None
    pixels=region[valid]
    q25,median,q75=np.percentile(pixels,[25,50,75])
    if (q75-q25)/median>.5:
        return None
    metres=float(median)
    return {"metres": metres,"feet":metres/.3048,"axial_metres":metres,"approximate":True}


class DepthProClient:
    """Binary frame pipe; only sampled distances return to the main worker."""
    def __init__(self, python, weights, device="cuda:0", timeout=180):
        if not Path(python).is_file() or not Path(weights).is_file():
            raise RuntimeError("Depth Pro installation missing; run backend/install_depth_pro.py")
        self.timeout=timeout
        self.events=queue.Queue()
        self.process=subprocess.Popen([str(python),str(Path(__file__).with_name('depth_service.py')),
            '--weights',str(weights),'--device',device],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=None,bufsize=0)
        self.reader=threading.Thread(target=self._read,daemon=True)
        self.reader.start()
        self.frames=0
        self.total_seconds=0.
        self.peak_allocated_mib=0.
        try:
            ready=self._receive()
            if not ready.get('ready'): raise RuntimeError('Depth Pro failed to initialize')
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                try: self.events.put(json.loads(line))
                except (ValueError,UnicodeError): pass
        finally: self.events.put(None)

    def _receive(self):
        try: event=self.events.get(timeout=self.timeout)
        except queue.Empty: raise RuntimeError('Depth Pro GPU request timed out') from None
        if event is None: raise RuntimeError('Depth Pro process exited unexpectedly')
        if event.get('error'): raise RuntimeError('Depth Pro: '+event['error'])
        return event

    def estimate(self, frame, detections):
        import numpy as np
        frame=np.ascontiguousarray(frame,dtype=np.uint8)
        height,width=frame.shape[:2]
        header=json.dumps(dict(width=width,height=height,boxes=[list(d['box']) for d in detections])).encode()+b'\n'
        self._write(header)
        # BufferedWriter makes full writes; Popen's unbuffered pipe may short-write.
        self._write(memoryview(frame).cast('B'))
        event=self._receive()
        if len(event.get('distances',[]))!=len(detections):
            raise RuntimeError('Depth Pro result ownership mismatch')
        self.frames+=1
        self.total_seconds+=event.get('seconds',0)
        self.peak_allocated_mib=max(self.peak_allocated_mib,event.get('peak_allocated_mib',0))
        return event['distances']

    def _write(self, data):
        view=memoryview(data).cast('B')
        while view:
            written=self.process.stdin.write(view)
            if not written: raise RuntimeError('Depth Pro frame transfer failed')
            view=view[written:]

    def notice(self):
        return 'Depth Pro AI estimate | axial depth in m / ft | not calibrated measurement'

    def metadata(self):
        return metadata()

    def close(self):
        if self.process.stdin:
            try: self.process.stdin.close()
            except OSError: pass
        if self.process.poll() is None:
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
        self.reader.join(timeout=2)
        if self.process.stdout: self.process.stdout.close()