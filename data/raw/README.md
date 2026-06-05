# How to generate training data

Run this script to create train/val/test CSVs from the pattern database.

```bash
python scripts/generate_training_data.py --output-dir data/raw
```

Each CSV has columns: `text`, `label
    - 0 = safe
    - 1 = toxic
    - 2 = jailbreak
