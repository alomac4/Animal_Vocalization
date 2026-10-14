"""Real original-video Depth Pro samples and timings, no ground-truth accuracy claim."""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from backend.depth_distance import DepthProClient
from backend.distance import DistancePreset
from backend.tracking import apply_track_breeds
from backend.worker import detection_label


def main():
    output=ROOT/'outputs/benchmarks/depth_pro'
    output.mkdir(parents=True,exist_ok=True)
    cached=ROOT/'outputs/benchmarks/original_video_tracking'
    winners={int(key):value for key,value in json.loads((cached/'ocsort-winners.json').read_text()).items()}
    numbers={100,1575,3150,4705}
    observations={}
    with (cached/'ocsort-observations.jsonl').open(encoding='utf-8') as spool:
        for line in spool:
            record=json.loads(line)
            if record['frame'] in numbers: observations[record['frame']]=record['detections']
    source=next((ROOT/'test_video').glob('*.mp4'))
    capture=cv2.VideoCapture(str(source))
    start=time.perf_counter()
    client=DepthProClient(ROOT/'backend/.depth-venv/Scripts/python.exe',ROOT/'backend/assets/depth_pro.pt')
    load=time.perf_counter()-start
    panels,records=[],[]
    try:
        for number in sorted(numbers):
            capture.set(cv2.CAP_PROP_POS_FRAMES,number)
            ok,frame=capture.read()
            if not ok: raise RuntimeError('Unable to decode original sample')
            detections=apply_track_breeds(observations[number],winners)
            times=[]
            for trial in range(3):
                start=time.perf_counter()
                values=client.estimate(frame,detections)
                times.append(time.perf_counter()-start)
            comparisons=[]
            for item,value in zip(detections,values):
                item['distance']=value
                x1,y1,x2,y2=item['box']
                label=detection_label(item)
                text_width=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.5,1)[0][0]
                position=(max(2,min(x1,frame.shape[1]-text_width-4)),max(20,y1-6))
                cv2.rectangle(frame,(x1,y1),(x2-1,y2-1),(29,76,185),2)
                cv2.putText(frame,label,position,cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),3)
                cv2.putText(frame,label,position,cv2.FONT_HERSHEY_SIMPLEX,.5,(255,248,236),1)
                comparisons.append(dict(dog_id=item['track_id'],depth_pro=value,
                    old_formula=DistancePreset().estimate(item['box'],frame.shape[1],frame.shape[0])))
            cv2.putText(frame,client.notice(),(4,frame.shape[0]-8),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),3)
            cv2.putText(frame,client.notice(),(4,frame.shape[0]-8),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,248,236),1)
            cv2.imwrite(str(output/f'frame-{number}.jpg'),frame)
            panels.append(cv2.resize(frame,(960,540)))
            record=dict(frame=number,trials_seconds=times,median_seconds=statistics.median(times),dogs=comparisons)
            records.append(record)
            print(json.dumps(record),flush=True)
    finally:
        client.close(); capture.release()
    cv2.imwrite(str(output/'comparison.jpg'),np.concatenate([np.concatenate(panels[:2],axis=1),np.concatenate(panels[2:],axis=1)],axis=0))
    (output/'report.json').write_text(json.dumps(dict(load_seconds=load,samples=records,
        metadata=client.metadata(),notes='Same original frame samples, official pretrained model, predicted focal length. No measured ground-truth ranges. These are integration/timing tests, NOT accuracy evidence. Original-frame depth, center-box axial median. Three repeats per sample; first inference includes warm-up.'),indent=2),encoding='utf-8')
    print('Depth Pro samples complete',flush=True)


if __name__=='__main__': main()