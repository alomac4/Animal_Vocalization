"""Official Depth Pro CUDA model process with length-delimited raw frame input."""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path


def emit(**event):
    print(json.dumps(event),flush=True)


def read_exact(stream,count):
    parts=[]
    remaining=count
    while remaining:
        chunk=stream.read(remaining)
        if not chunk: raise RuntimeError('Incomplete depth frame input')
        parts.append(chunk); remaining-=len(chunk)
    return b''.join(parts)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
    from depth_distance import sample_box_depth

    if not args.device.startswith('cuda:') or not torch.cuda.is_available():
        raise RuntimeError('Depth Pro requires CUDA; no CPU/formula fallback')
    manifest=json.loads(args.weights.with_name('depth-pro-manifest.json').read_text())
    digest=hashlib.sha256()
    with args.weights.open('rb') as file:
        while chunk:=file.read(8*1024**2): digest.update(chunk)
    if digest.hexdigest()!=manifest['sha256']: raise RuntimeError('Depth Pro checkpoint integrity mismatch')
    torch.set_num_threads(2)
    config=replace(DEFAULT_MONODEPTH_CONFIG_DICT,checkpoint_uri=str(args.weights))
    model,transform=depth_pro.create_model_and_transforms(config=config,device=torch.device(args.device),precision=torch.float16)
    model.eval()
    emit(ready=True)
    stream=sys.stdin.buffer
    while line:=stream.readline(1024*1024):
        request=json.loads(line)
        width,height=request['width'],request['height']
        boxes=request['boxes']
        if (not isinstance(width,int) or not isinstance(height,int) or min(width,height)<16
                or max(width,height)>4096 or len(boxes)>1000):
            raise ValueError('Depth request exceeds bounded frame/box limits')
        data=read_exact(stream,width*height*3)
        bgr=np.frombuffer(data,dtype=np.uint8).reshape(height,width,3)
        start=time.perf_counter()
        image=transform(Image.fromarray(bgr[:,:,::-1]))
        with torch.inference_mode():
            prediction=model.infer(image,f_px=None)
        depth=prediction['depth'].float().cpu().numpy().squeeze()
        if depth.shape!=(height,width): raise RuntimeError('Depth output does not match original frame')
        focal=float(prediction['focallength_px'].item())
        if not np.isfinite(focal) or focal<=0: raise RuntimeError('Invalid predicted camera focal length')
        distances=[sample_box_depth(depth,box) for box in boxes]
        emit(distances=distances,seconds=round(time.perf_counter()-start,6),focal_px=focal,
             peak_allocated_mib=round(torch.cuda.max_memory_allocated()/1024**2,1))
        del depth,prediction,image,data,bgr


if __name__=='__main__':
    try: main()
    except Exception as exc:
        emit(error=str(exc)[:500]); sys.exit(1)