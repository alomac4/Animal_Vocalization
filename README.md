# Dog detection and breed recognition

## Android app and private PC video server

The Warm Companion interface now uses the centered original beagle logo without
adjacent BARK text. App development and serving are separate from training:

- React Native / Expo app and build instructions:
  `C:\Work\Classes\Animal_Vocalization\mobile\README.md`
- Authenticated CPU inference server and private phone connection setup:
  `C:\Work\Classes\Animal_Vocalization\backend\README.md`
- HTML/CSS design showcase:
  `C:\Work\Classes\Animal_Vocalization\design-showcase\index.html`

The app uploads one video, shows processing progress, and exports an MP4 with
boxes already rendered into its frames. No database or permanent result history.
Temporary server files expire. Tailscale sign-in on the PC and phone is required
for remote access; no public server exposure is enabled. While training continues,
the service uses a copied pretrained dog-detection baseline on CPU, not the
checkpoint currently being written by training.

Stage 1 trains a COCO-pretrained **YOLO12 small** detector with **130 original
Tsinghua Dogs breed categories**, using **whole-body** boxes, **100 epochs**, and
**960 x 960** input at **batch 4**. Historical smoke/benchmark runs used 992 pixels.
Image resizing preserves aspect ratio using padding. Age estimation is a future,
separate model. User pet names are backend metadata, never renamed breed classes.

## Full dataset: custom 70/15/15 split

Run the complete download/preparation/training pipeline:

```powershell
& 'C:\Work\Classes\Animal_Vocalization\scripts\start_full_training.ps1'
```

Full images and archives go to `E:\Datasets\TsinghuaDogs\original`; annotations
and original manifests go to `E:\Datasets\TsinghuaDogs\metadata`. The pipeline
expects all 70,428 original image paths and 130 breeds before preparing anything.
Missing images fail preparation instead of silently training on a partial download.

Preparation fully decodes each image, validates body boxes and image/XML sizes,
and excludes invalid annotations/images (every exclusion is recorded). SHA-256
hashes of decoded RGB pixels identify exact-content duplicates even when image
encodings differ. Consistently annotated duplicates retain one image; conflicting
duplicate annotations exclude the entire group. **Splitting happens after these
checks**, not before, so excluded images do not distort the final ratios.

The custom split is deterministic (seed 42), breed-stratified, globally rounded
by largest remainder, and uses all unique valid images. With no exclusions, the
counts would be **49,300 train / 10,564 validation / 10,564 test**. Actual counts
are printed and recorded after preparation. Each split must represent every breed.
Images are hardlinked on E: rather than duplicated. Outputs are:

```text
E:\Datasets\TsinghuaDogs\yolo_70_15_15_seed42\
  data.yaml                  Separate train, val and test manifests
  train.txt / val.txt / test.txt
  images\ / labels\         Validated images and whole-body detection labels
  preparation_report.json   Exclusions, deduplication and breed counts per split
  image_inventory.json      Split membership and decoded-pixel hashes
```

Training uses **batch 4, 960 x 960, 100 epochs**, fresh pretrained YOLO12-small
weights, and only the validation split for epoch metrics/best-checkpoint selection.
After training, the pipeline evaluates `best.pt` once on the held-out test split
and writes `test_metrics.json`. New checkpoints and run outputs stay beneath
`C:\Work\Classes\Animal_Vocalization\weights\yolo12s_960_full_701515_<timestamp>`.
Download progress, preparation counts, batch progress and automatic epoch summaries
appear in the terminal and a persistent `.pipeline.log` under the workspace's
`logs` directory; `.pipeline.json` tracks the current stage and child PID.

This is a **custom split**, not the official benchmark split. Exact duplicate
content cannot cross splits, but near-duplicates and photos of the same individual
dog cannot be reliably grouped without identity metadata. Interrupted downloads
resume from `.part` files. Preparation refuses to overwrite a partial conversion;
inspect/move the incomplete converted directory before retrying. Keep the launched
terminal open while the pipeline runs.

## Earlier small existing-dataset run

The largest ready local dataset is currently the benchmark subset: **192 train /
48 validation images**, a 130-class mapping, and only 48 breeds represented in
validation. This is not full-dataset training and does not establish production
breed accuracy. No new dataset download is needed for this run.

Launch from PowerShell to see live batch progress and automatic epoch summaries:

```powershell
& 'C:\Work\Classes\Animal_Vocalization\scripts\start_existing_training.ps1'
```

The launcher uses batch 4 and the current 960-pixel configuration. It saves
everything beneath `C:\Work\Classes\Animal_Vocalization\weights\yolo12s_960_existing_<timestamp>`:

- `weights\best.pt` and `weights\last.pt`: trained checkpoints.
- `training_state.json`: latest epoch/status, losses, validation metrics, learning
  rates, elapsed time and estimated remaining time (updated automatically).
- `epoch_state.jsonl`: one state record per completed training epoch.
- `results.csv`, plots and `training_summary.json`: training outputs.

A terminal transcript is saved beneath `C:\Work\Classes\Animal_Vocalization\logs`.
The training program itself prints each epoch's start and post-validation summary;
no manual status narration or polling is required. Keep the launched terminal open;
press Ctrl+C to interrupt. Resume an interrupted run using its actual `last.pt`
with `train_breeds.py --resume <absolute-checkpoint-path>`.

## Historical workspace layout and full-dataset workflow

All project-owned files are organized under
`C:\Work\Classes\Animal_Vocalization`:

```text
C:\Work\Classes\Animal_Vocalization\
  scripts\                    Python/PowerShell entry points and path utilities
  configs\                    Training configuration
  datasets\tsinghua_dogs\
    metadata\                 Original annotation ZIPs, XMLs, official splits
    original\                 Full image archives/extraction when downloaded
    annotation_audit\         Annotation checks and class mapping
    smoke_original\           Six original integration-test images
    smoke_yolo\               Converted integration-test images/labels
    smoke_subsets\            Generated short-run manifests
    benchmark\                Stratified original images/labels for batch testing
    yolo_breeds\              Full converted dataset when prepared
  weights\                    New training runs/checkpoints and live state
  models\
    pretrained\               Downloaded base/AMP-check checkpoints
    breed_detection\runs\     Training checkpoints, metrics, run configuration
    benchmarks\               Benchmark run configuration/metrics (no weights)
  outputs\                    Inference JSONL and visualization files
  logs\                       Installation/audit logs and local library settings
  tests\                      Automated tests
  .venv\                      Python environment
  README.md
  requirements.txt
```

Paths are derived from the workspace, not the caller's current directory.
The default pretrained model is
`C:\Work\Classes\Animal_Vocalization\models\pretrained\yolo12s.pt`.
Both `python <absolute-script-path>` and `python -m scripts.train_breeds`
(from the workspace) are supported. Model and dataset CLI paths remain configurable.
The earlier `data`, `runs`, and `weights` root folders were consolidated; generated
text/JSON/YAML references were relocated and old-path image caches invalidated.
Existing binary checkpoints and historical logs are preserved unchanged and may
contain their original run paths. Use them for inference; the completed smoke
checkpoints are not resumable full training runs.

For datasets kept in the working folder, the launcher defaults to
`C:\Work\Classes\Animal_Vocalization\datasets\tsinghua_dogs`:

```powershell
& 'C:\Work\Classes\Animal_Vocalization\scripts\run_training.ps1' -Batch 4 -SkipInvalidAnnotations
```

**Do not launch a full download there with the current C: free space.** The
storage section below shows an explicit external-drive override. Alternatively,
the `datasets\tsinghua_dogs` directory can be moved to a larger drive and replaced
with a Windows directory junction, retaining the same workspace-facing layout.
No external relocation/junction or full download has been performed automatically.

## Status and limitations

- A Python 3.12 virtual environment and CUDA-enabled PyTorch have been installed.
- Original high-resolution annotations and official train/validation lists are
  in `C:\Work\Classes\Animal_Vocalization\datasets\tsinghua_dogs\metadata`.
- Six original high-resolution Shiba images were fetched for a small integration
  test. They are **not** the full training dataset and are not representative.
- A one-epoch GPU smoke run is not a useful trained breed model. Never deploy
  weights from a run named `smoke_*` as the final model.
- The original split has 65,228 train and 5,200 validation images, and 130 breeds.
- An inspection of every official high-resolution XML found **one annotated
  dog per image**. Unannotated secondary dogs may exist. Mosaic augmentation is
  not a substitute for real crowded scenes. Evaluate on independently collected,
  fully annotated multi-dog images before claiming multi-dog performance.
- Breed frequencies are imbalanced. Inspect per-breed results/confusion matrices,
  not only overall mAP. The validation set is used during training; it is not an
  independent final test set.
- Mixed breeds and breeds absent from the 130 categories are not explicitly
  modeled. A detector can confidently misclassify them; score thresholds alone
  do not guarantee out-of-distribution rejection.
- Review visual box overlays before a full run. XML coordinates are interpreted
  as continuous image-boundary coordinates, clipped to `[0,W]` and `[0,H]`.
- The full XML audit found **one unusable training box**, in
  `233-n000071-malinois/n120915.jpg`, entirely outside its declared image, and
  **1,215 other boxes requiring boundary clipping**. Review these in
  `C:\Work\Classes\Animal_Vocalization\datasets\tsinghua_dogs\annotation_audit\preparation_report.json`.
  The commands below explicitly opt into skipping invalid records. With the
  annotation audit's single exclusion, counts are 65,227 train / 5,200 validation;
  full image validation may identify further issues, recorded in its own report.
- The **130-class GPU smoke test** passed at batch 4 / 992: peak PyTorch allocated
  memory **4.65 GiB**, peak reserved **5.68 GiB**. These measurements cover a tiny
  six-image test, not a guaranteed worst case for the full dataset.
- Review Ultralytics' AGPL/Enterprise licensing before commercial backend use.

Official sources:

- Dataset and downloads: https://cg.cs.tsinghua.edu.cn/ThuDogs/
- Training options: https://docs.ultralytics.com/modes/train/
- YOLO12 support/caveats: https://docs.ultralytics.com/models/yolo12/
- PyTorch wheels: https://pytorch.org/get-started/previous-versions/
- License: https://www.ultralytics.com/license

## Environment (already created in this workspace)

Use the project interpreter, **not** the system's Python 3.14 interpreter.
From PowerShell:

```powershell
py -3.12 -m venv 'C:\Work\Classes\Animal_Vocalization\.venv'
$Python = 'C:\Work\Classes\Animal_Vocalization\.venv\Scripts\python.exe'
& $Python -m pip install --upgrade pip
& $Python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
& $Python -m pip install -r 'C:\Work\Classes\Animal_Vocalization\requirements.txt'
& $Python -m pip check
& $Python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

CUDA 12.8 wheels are pinned to a tested PyTorch/torchvision pair. A separate local
CUDA toolkit or FlashAttention installation is not required by this tested setup.
7-Zip is required to extract the original nine-part high-resolution ZIP; the
downloader defaults to `C:\Program Files\7-Zip\7z.exe`.

## Storage

The publisher lists a 38.8 GB high-resolution image download. Allow **at least
90 GiB** for archives plus extraction, metadata, and a little headroom; training
outputs need further space. The downloader enforces this conservative reserve.
C: did not have sufficient room when inspected. E: and G: had enough free space.
Prefer the faster local SSD; the following commands use **E: only as an example**.

YOLO images are **hardlinked**, not copied, by default. Put original images and
converted output on the same NTFS volume. Hardlinks avoid another full image
copy. To intentionally copy across volumes, pass `--image-mode copy` and allow
additional disk space. Do not edit a hardlinked image in place: both paths refer
to the same file.

## One-command full pipeline

This explicitly starts a large download, extraction, conversion, GPU smoke test,
then the full 100-epoch run. It has **not** been started automatically.

```powershell
& 'C:\Work\Classes\Animal_Vocalization\scripts\run_training.ps1' -DataRoot 'E:\Datasets\TsinghuaDogs' -Batch 4 -SkipInvalidAnnotations
```

PowerShell execution policy may require calling this local script through an
approved process-specific policy in your environment; no machine policy is
changed by this project. Each step stops if the previous one fails. Archive
downloads resume from `.part` files. Dataset conversion refuses nonempty outputs
to prevent stale labels. If a conversion fails, inspect its report, move/delete
that **converted output only**, resolve the issue, and rerun. An explicit
`-SkipInvalidAnnotations` opt-in excludes invalid records and logs every exclusion;
it does not silently invent labels. Breeds must remain present in both splits.

## Manual workflow

### 1. Download and extract original images

```powershell
$Python = 'C:\Work\Classes\Animal_Vocalization\.venv\Scripts\python.exe'
& $Python 'C:\Work\Classes\Animal_Vocalization\scripts\download_dataset.py' --metadata 'C:\Work\Classes\Animal_Vocalization\datasets\tsinghua_dogs\metadata' --images 'E:\Datasets\TsinghuaDogs\original' --extract
```

### 2. Convert to YOLO detection format

```powershell
& $Python 'C:\Work\Classes\Animal_Vocalization\scripts\prepare_dataset.py' --annotations 'C:\Work\Classes\Animal_Vocalization\datasets\tsinghua_dogs\metadata\High-Annotations' --splits 'C:\Work\Classes\Animal_Vocalization\datasets\tsinghua_dogs\metadata\TrainAndValList' --images 'E:\Datasets\TsinghuaDogs\original\high-resolution' --output 'E:\Datasets\TsinghuaDogs\yolo_breeds' --skip-invalid
```

The converter reads `bodybndbox` for **every** XML object, never substitutes a
head/full-image box, checks unknown breeds, malformed boxes, split overlap,
duplicate entries, EXIF rotation, and image/XML dimension agreement. Invalid boxes
fail by default; out-of-bound coordinates are clipped and counted in the report.
For an annotation-only audit, omit `--images`. An audit deliberately does not
create a training YAML.

Outputs:

```text
E:\Datasets\TsinghuaDogs\yolo_breeds\
  data.yaml
  classes.json
  preparation_report.json
  train.txt
  val.txt
  images\train\<original-breed-folder>\<image>
  images\val\<original-breed-folder>\<image>
  labels\train\<original-breed-folder>\<image-stem>.txt
  labels\val\<original-breed-folder>\<image-stem>.txt
```

Each label row is `breed_class_id center_x center_y width height`, normalized to
the matching image dimensions. Class order is stable by original breed code,
not by directory iteration or breed population. The entire 130-class mapping
is stored in `classes.json` and embedded in trained checkpoints.

### 3. Validate and smoke-test on your GPU

```powershell
& $Python 'C:\Work\Classes\Animal_Vocalization\scripts\train_breeds.py' --data 'E:\Datasets\TsinghuaDogs\yolo_breeds\data.yaml' --check-only
& $Python 'C:\Work\Classes\Animal_Vocalization\scripts\train_breeds.py' --data 'E:\Datasets\TsinghuaDogs\yolo_breeds\data.yaml' --smoke-test --batch 4
```

Smoke testing uses at most eight images per split, one epoch, and the full class
head at the configured resolution (currently 960 pixels). It logs peak allocated/reserved PyTorch VRAM in
`training_summary.json`. This is not total system GPU usage and not a worst-case
guarantee. Full-data validation/augmentation and larger batches can use more.

### 4. Train for 100 epochs

```powershell
& $Python 'C:\Work\Classes\Animal_Vocalization\scripts\train_breeds.py' --data 'E:\Datasets\TsinghuaDogs\yolo_breeds\data.yaml' --batch 4
```

The config is `C:\Work\Classes\Animal_Vocalization\configs\breed_train.yaml`.
It enables mixed precision and validation, disables early stopping, and disables
RAM/disk image caching. Checkpoints are saved each epoch and every tenth epoch.
To compare larger batches, use the full-batch benchmark below, not the tiny
smoke test (an eight-image subset cannot test an actual batch of sixteen).
Alternatively use `--batch 0.65` for AutoBatch. The full 100-epoch run will be considerably
longer than the tiny smoke test; no duration estimate is inferred from it.

Default final weights (a repeated run may receive an incremented directory):

- `C:\Work\Classes\Animal_Vocalization\weights\yolo12s_960\weights\best.pt`
- `C:\Work\Classes\Animal_Vocalization\weights\yolo12s_960\weights\last.pt`

Resume an **interrupted** training run from its actual checkpoint:

```powershell
& $Python 'C:\Work\Classes\Animal_Vocalization\scripts\train_breeds.py' --resume 'C:\Work\Classes\Animal_Vocalization\weights\yolo12s_960\weights\last.pt'
```

## Batch-size benchmark

```powershell
$Python = 'C:\Work\Classes\Animal_Vocalization\.venv\Scripts\python.exe'
& $Python 'C:\Work\Classes\Animal_Vocalization\scripts\prepare_benchmark.py'
& $Python 'C:\Work\Classes\Animal_Vocalization\scripts\benchmark_batches.py'
```

The preparation script fetches only selected ZIP member byte ranges from the
official high-resolution archives, verifies image CRCs and annotations, and
selects 192 train images across all 130 breeds plus 48 separate validation images.
Invalid original dimensions/EXIF rotation are logged and replaced by validated
images of the same breed. Selection manifests remain in the benchmark dataset
folder. Validation includes fewer than 130 breeds: this is **not an accuracy
evaluation**. The converter's normal full-dataset coverage checks remain enabled;
only this programmatic benchmark conversion opts out of per-split coverage.

The benchmark runs batch sizes 4, 8, 12, 16 in separate processes, twice each,
reversing order on the second pass. It verifies actual full training batches,
uses 992 x 992 with AMP and the existing augmentation settings, and excludes
the first of three epochs from throughput timing. CUDA-synchronized epoch wall
time includes data loading and augmentation, but excludes validation/setup.
Validation runs every epoch and its peak VRAM is recorded separately. Automatic
OOM batch reduction is disabled: failures must not masquerade as a larger batch.

For fair short-run timing the benchmark uses fixed AdamW, gradient accumulation
with nominal batch 64, and no learning-rate warmup. Production retains its own
`optimizer: auto` and warmup settings. Benchmark runs save metrics/configuration
under `C:\Work\Classes\Animal_Vocalization\models\benchmarks`, not trained weights.
Reports go to `C:\Work\Classes\Animal_Vocalization\outputs\benchmarks` and detailed
logs to `C:\Work\Classes\Animal_Vocalization\logs\benchmarks`.

PyTorch reserved VRAM is **not total GPU memory consumption**. Results depend on
background GPU use, image sizes/storage, object counts and long-run thermals.
The benchmark never changes the production configuration or launches 100 epochs.

### Measured RTX 4090 results

Two verified repetitions per batch (three epochs each, first epoch excluded):

| Actual batch | Training images/second | Peak training allocated GiB | Peak reserved GiB (train/val) |
| --- | ---: | ---: | ---: |
| 4 | 32.99 | 4.83 | 7.12 |
| 8 | 31.11 | 9.36 | 12.86 |
| 12 | 30.43 | 13.93 | 16.68 |
| 16 | 30.11 | 18.46 | 22.17 |

All eight verified runs passed with exact requested batches and finite losses.
For this setup **retain batch 4**: it was fastest and has the most memory headroom.
Batch 16 completed but leaves little reserved-memory headroom and gave no speed
benefit. These results do not compare breed accuracy or guarantee full-dataset
memory use. The verified reports are in
`C:\Work\Classes\Animal_Vocalization\outputs\benchmarks\batch_comparison_992_verified`.
The earlier `batch_comparison_992` reports preserve an initial benchmark-wrapper
failure (checkpoint reload expected despite intentionally disabled saving); use
only the `verified` directory for the final comparison.

## Backend-friendly inference

Use full-training `best.pt`, not smoke-test weights:

```powershell
& $Python 'C:\Work\Classes\Animal_Vocalization\scripts\predict_breeds.py' --weights 'C:\Work\Classes\Animal_Vocalization\weights\yolo12s_960\weights\best.pt' --source 'E:\dog_photos' --output 'C:\Work\Classes\Animal_Vocalization\outputs\detections.jsonl'
```

The output is JSON Lines, one record per image/frame. Each detection contains
`bbox_xyxy`, `breed.class_id`, `breed.label`, `breed.confidence`, and separate
unassigned `identity` fields. Confidence is a detection score, not a calibrated
probability of breed correctness and not pet-identity confidence. The API leaves
age absent until a separate estimator exists. `results_to_payload()` is reusable
in an API service; no web framework is assumed or installed.

Inference uses class-agnostic NMS to reduce duplicate breed hypotheses on one dog.
Evaluate/tune its IoU threshold for heavily overlapping dogs; it can suppress
distinct dogs whose boxes overlap strongly. Add `--track` only for a single video
or stream, not unrelated photographs. Track IDs are session-local, not registered
pet identities. A backend must scope name assignments by session and handle
identity switches.

## Tests

```powershell
& 'C:\Work\Classes\Animal_Vocalization\.venv\Scripts\python.exe' -m unittest discover -s 'C:\Work\Classes\Animal_Vocalization\tests' -v
```

Tests and direct script invocations work from any current directory.
They cover conversion, normalization, multi-object support, safety checks, image
size mismatches, and separation of breed predictions from identity metadata.