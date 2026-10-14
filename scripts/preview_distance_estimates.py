"""Render distance-only diagnostic using original frames and existing track cache."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from backend.distance import DistancePreset
from backend.tracking import apply_track_breeds
from backend.worker import detection_label


def main():
    source = next((ROOT/'test_video').glob('*.mp4'))
    cached=ROOT/'outputs/benchmarks/original_video_tracking'
    winners={int(key):value for key,value in json.loads((cached/'ocsort-winners.json').read_text()).items()}
    output=ROOT/'outputs/benchmarks/distance_estimates'
    output.mkdir(parents=True,exist_ok=True)
    capture=cv2.VideoCapture(str(source))
    if not capture.isOpened(): raise RuntimeError('Cannot open original video')
    preset=DistancePreset()
    samples={100,1575,3150,4705}
    panels=[]
    records=[]
    try:
        with (cached/'ocsort-observations.jsonl').open(encoding='utf-8') as spool:
            for line in spool:
                record=json.loads(line)
                if record['frame'] not in samples: continue
                capture.set(cv2.CAP_PROP_POS_FRAMES,record['frame'])
                ok,frame=capture.read()
                if not ok: raise RuntimeError('Sample frame unavailable')
                height,width=frame.shape[:2]
                values=[]
                for item in apply_track_breeds(record['detections'],winners):
                    item['distance']=preset.estimate(item['box'],width,height)
                    x1,y1,x2,y2=item['box']
                    cv2.rectangle(frame,(x1,y1),(x2-1,y2-1),(29,76,185),2)
                    label=detection_label(item)
                    text_width=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.5,1)[0][0]
                    position=(max(2,min(x1,width-text_width-4)),max(20,y1-6))
                    cv2.putText(frame,label,position,cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),3)
                    cv2.putText(frame,label,position,cv2.FONT_HERSHEY_SIMPLEX,.5,(255,248,236),1)
                    values.append(dict(dog_id=item['track_id'],box=item['box'],distance=item['distance']))
                cv2.putText(frame,preset.notice(),(4,height-8),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),3)
                cv2.putText(frame,preset.notice(),(4,height-8),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,248,236),1)
                cv2.imwrite(str(output/f"frame-{record['frame']}.jpg"),frame)
                panels.append(cv2.resize(frame,(960,540)))
                records.append(dict(frame=record['frame'],dogs=values))
    finally: capture.release()
    assert len(panels)==4
    cv2.imwrite(str(output/'comparison.jpg'),np.concatenate([np.concatenate(panels[:2],axis=1),np.concatenate(panels[2:],axis=1)],axis=0))
    (output/'report.json').write_text(json.dumps(dict(preset=preset.metadata(),frames=records,
        note='Illustrative assumed-camera/assumed-size calculation. No known real camera calibration or ground-truth distances. Not accuracy validation; tracking unchanged.'),indent=2),encoding='utf-8')
    print('Distance samples:',output,flush=True)


if __name__=='__main__': main()