
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

## Combined Assessment

- **Total images across both datasets:** ~2,976 (2,816 CarDD + 160 Kaggle)
- **Total labeled detections:** ~6,428 (6,211 CarDD + 217 Kaggle)
- **Unique damage classes covered:** 7 total — dent, scratch, crack, shattered_glass, broken_lamp, flat_tire (from CarDD), plus dislocation (Kaggle-only, not in CarDD)
- **Class balance:** CarDD alone is significantly imbalanced (11.4x, scratch vs. flat_tire); combining datasets doesn't fully fix this — Week 2's training will need the weighted sampler (already built into the plan) to avoid the model just learning to predict "scratch" most of the time.
- **Vehicle make/model coverage:** NOT filtered or tracked by make/model in either dataset — CarDD is a general mixed-vehicle dataset. This is a gap against the PRD's "3-5 pilot vehicle makes" MVP scope; revisit if make-specific behavior becomes important later.
- **Part-level labels:** Neither dataset provides reliable part identification (bumper, headlight, etc.) — both only label damage TYPE. This is the same "part": "unknown" gap already documented in the Week 2/3 master docs — a known, deliberate scope limitation, not an oversight.
- **Ethical/compliance notes:** CarDD images sourced from Flickr under license terms accepted at download; Kaggle dataset sourced from a public Kaggle dataset. No PII (faces, plates) manually reviewed yet — worth a quick visual spot-check before Week 2 training if time allows. Both are used here as prototyping/portfolio data, not production-validated insurer data (per the roadmap's honest scope notes).

## Decision for Week 2

- Combine both datasets into one manifest via `prepare_dataset.py` (already built for exactly this).
- Train on all 7 classes; expect flat_tire and dislocation (smallest classes) to be the hardest to get good recall on given limited examples — flag this specifically in Day 7's evaluation/error analysis rather than being surprised by it.
- Use the weighted sampler (`--use_weighted_sampler True`, already the default) to counter the 11.4x imbalance.
- Do NOT attempt part-level classification this pass — explicitly deferred, as documented.
