# Dog identity research and proposed experiments

Research only. No new models installed and no production settings changed.

## Existing local evidence

Original clip: 4,725 frames, approximately 158 seconds, 1280 x 720. The user
reports five dogs. Identical cached detections and the same duration filtering
produced 31 final identities for color/motion, 28 for OC-SORT, 31 for BoT-SORT
without neural ReID, and 32 for FastTracker. Short-gap optical flow worsened
OC-SORT to 30 identities and increased association/decode time from 13.617 to
60.672 seconds. Counts alone do not establish correct identity association.
The reviewed frames also show a false detector box around a tree stump.

## Recommendation

Add learned appearance embeddings plus clip-wide constrained tracklet association.
Keep local motion tracking for short gaps. Separate transient tracklet IDs from
physical-dog identity hypotheses. Breed and super-resolution must not define
identity. Keep ambiguous associations unresolved rather than force five dogs.
This is a proposed architecture, not an established result on the user's video.

### Candidate embedding models

1. MiewID-msv2, Conservation X Labs: pretrained multi-species individual ReID.
   Model card specifies ~51M parameters and 440 x 440 inputs. It reports 87.38%
   rank-1 retrieval on DogFaceNet. This is dog-face retrieval evidence, NOT a
   result on whole-body backyard video tracking or on this clip. Review custom
   model code before loading (published usage requests trust_remote_code), pin
   revisions, and verify redistribution terms before shipping.
2. MegaDescriptor-L-384, BVRA/WildlifeDatasets: animal ReID feature extractor,
   ~229M parameters, 384 x 384, public weights. Its model card is CC-BY-NC-4.0;
   do not assume it is suitable for a commercial app. Useful research comparator.
3. open-noodle/pet-recognition-base: whole-animal dog/cat embeddings, frozen
   DINOv2-base plus learned 512D projection, dynamic-batch ONNX, 224 x 224.
   Publisher declares Apache-2.0 and reports 61.2% top-1 on whole-animal Dogs-World
   and 94.3% on DogFaceNet. These are publisher-reported results, not independent
   validation; small/community model needs extra provenance and licensing review.
4. Official DINOv2 features: sensible generic baseline, not dog-identity-trained.
   Official model card describes retrieval use and Apache-2.0 licensing for these
   backbones. Do not confuse with separately licensed Cell-DINO variants.

Run a small labelled crop verification test on this clip BEFORE running every
candidate through all video frames. Label same/different pairs for difficult
poses, two similar black/white dogs, occlusions, blurry/distant dogs and stump
false positives. Sample at separated times; do not report adjacent-frame leakage
as generalization. Compare original crops against masked crops if necessary.
Do not start by feeding neural-super-resolved identity crops: that is an untested
preprocessing change; compare separately rather than assume enhancement helps.

### Proposed per-video identity gallery and offline association

- Store several high-quality original views per identity, rather than one mean
  descriptor or color histogram; retain front/side/rear pose diversity.
- Initially test embeddings every 0.5-1 second plus uncertain recovery events,
  in GPU batches. This is a proposed cadence, not a measured speed guarantee.
- Never update a trusted gallery from severely occluded, overlapping, tiny or
  blurry crops. Do not propagate an uncertain identity into all later memories.
- For a candidate association, combine embedding similarity and calibrated margin
  over alternatives with short-range geometry, temporal overlap and visibility.
- Offline pass pools consistent evidence from multiple frames and both ends of
  a tracklet. Prohibit simultaneous distinct dogs sharing an identity; suppress
  same-dog duplicate detector boxes before imposing that constraint.
- Distinguish missing detection, occlusion and exit from a newly discovered dog.
- Pool known breed votes only after identity association is finalized.
- Keep these galleries within the current temporary job lifecycle; no DB needed.

### Learned sequential video tracking

SAM 2.1 uses streaming visual memory for prompted video masks. SAM 3.1 Object
Multiplex, released March 27, 2026, jointly tracks multiple objects and offers
publicly documented checkpoints. It is a relevant alternative to linear-motion
and Lucas-Kanade flow, but mask propagation is not independently validated
whole-animal ReID after long exits. Detector/re-identification anchors still
need integration and testing. Derive output boxes from masks; masks need not be
shown in the app. Start on short difficult video sections, not all 4,725 frames.
SAM 3 checkpoints require owner access/authentication; README lists Python >=3.12,
PyTorch >=2.7 and CUDA >=12.6. Benchmarks on H100/128 objects are not runtime
predictions for five dogs on an RTX 4090. Measure actual peak memory and runtime.
SAM 2 installation recommends WSL on Windows. Use an isolated experiment
environment rather than upgrading the serving/training environment blindly.

CoTracker3 tracks points, with pretrained online/offline checkpoints. It is a
possible gap-motion experiment, not a dog identity embedding model. Lower
priority after the failed classical flow experiment.

PetFace and DogFaceNet are face-centric identity resources: helpful when a face
is visible, insufficient as the only cue for rear/side/distant dogs. PetFace's
official repo restricts data/code/models to noncommercial research.

## Evaluation and deployment gates

1. Annotate evaluation sections with true dog IDs, occlusion/missing visibility,
   duplicate boxes and non-dog boxes; retain held-out sections for final checks.
2. Measure same/different retrieval errors first, especially false merges.
3. Then run the chosen embedding hybrid on identical detector cache and assess
   ID switches, fragmentation, false merges and visibility coverage. TrackEval
   supports IDF1/HOTA and detection/association components for labelled evaluation.
4. Render comparison videos and per-dog timelines. Five output IDs alone is not
   success if different animals share IDs or genuine appearances disappear.
5. Measure added ReID time, GPU memory and total runtime; deployment is conditional
   on actual improvement. No guaranteed outcome or unmeasured FPS estimate.

Duration filtering is only a heuristic. It can drop genuine brief visitors and
retain long-lasting false objects; equal frame counts are not proof of identity.
If automatic pretrained ReID fails, try a separate semi-assisted benchmark:
manually select one or more reference views per dog and use them with video-memory
tracking. This requires an optional reference/correction UI if shipped, and is
different from fully automatic upload inference. Fine-tuning should use multiple
videos/dogs and identity labels, not breed labels; evaluate on held-out clips.

## Primary sources consulted

- https://huggingface.co/conservationxlabs/miewid-msv2
- https://arxiv.org/abs/2412.05602
- https://huggingface.co/BVRA/MegaDescriptor-L-384
- https://github.com/WildlifeDatasets/wildlife-tools
- https://openaccess.thecvf.com/content/WACV2024/papers/Cermak_WildlifeDatasets_An_Open-Source_Toolkit_for_Animal_Re-Identification_WACV_2024_paper.pdf
- https://huggingface.co/open-noodle/pet-recognition-base
- https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
- https://github.com/facebookresearch/sam2
- https://github.com/facebookresearch/sam2/blob/main/INSTALL.md
- https://github.com/facebookresearch/sam3/blob/main/README.md
- https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md
- https://github.com/facebookresearch/co-tracker
- https://github.com/mapooon/PetFace
- https://github.com/GuillaumeMougeot/DogFaceNet
- https://github.com/JonathonLuiten/TrackEval

Sources describe general datasets/benchmarks, not accuracy on the user's video.
Model-card declarations and access requirements should be verified at installation.