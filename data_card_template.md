
## Dataset 2 (actual): eashankaushik/car-damage-detection (VIA format)
- Source: kaggle.com/datasets/eashankaushik/car-damage-detection
- Annotation format: VIA (VGG Image Annotator) JSON, polygon-level
- Train: 217 regions / 160 images — Scratch 84, Dent 46, Dislocation 45, Shatter 42
- Val: 17 regions / 8 images — Scratch 7, Shatter 6, Dislocation 2, Dent 2
- Class imbalance: 2.0x (train), 3.5x (val, small-sample noise)
- Adds a "Dislocation" class not present in CarDD
- Known limitation: small dataset, best used as a CarDD supplement not primary source

## Dataset 1 (actual): CarDD (via Hugging Face FiftyOne mirror)
- Source: huggingface.co/datasets/harpreetsahota/CarDD
- Format: FiftyOne export (samples.json) — NOT raw COCO despite official docs describing it that way; contains normalized bounding boxes + base64 segmentation masks (masks not used in this pipeline)
- This export = train split only: 2,816 images, 6,211 detections
- Class distribution: scratch 2560, dent 1806, crack 651, broken_lamp 494, shattered_glass 475, flat_tire 225
- Class imbalance: 11.4x (scratch vs. flat_tire) — weighted sampler required in Week 2 training
- Verified against official CarDD paper's published train-split counts — exact match, confirms correct/legitimate data
- Known gap: val/test splits not included in this particular mirror export; may need separate download or a manual re-split of train for Week 2 if a true held-out test set is required
