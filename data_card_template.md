
## Dataset 2 (actual): eashankaushik/car-damage-detection (VIA format)
- Source: kaggle.com/datasets/eashankaushik/car-damage-detection
- Annotation format: VIA (VGG Image Annotator) JSON, polygon-level
- Train: 217 regions / 160 images — Scratch 84, Dent 46, Dislocation 45, Shatter 42
- Val: 17 regions / 8 images — Scratch 7, Shatter 6, Dislocation 2, Dent 2
- Class imbalance: 2.0x (train), 3.5x (val, small-sample noise)
- Adds a "Dislocation" class not present in CarDD
- Known limitation: small dataset, best used as a CarDD supplement not primary source
