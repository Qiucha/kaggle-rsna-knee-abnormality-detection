# Data

Nothing in this folder is in git. The competition rules (section 4.b) forbid redistributing the
competition data, so download it yourself after accepting the rules.

| File | Source |
|---|---|
| `train.csv`, `train_series.csv` | [Competition data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) |
| `gold_label.csv` | The 58 rows of `train.csv` that carry labels (see below) |
| `gold_unlabeled.csv` | The same 58 rows with the label columns blanked |
| `lm_label/lm_labels_kaggle.csv` | @pilkwang's CC0 dataset [`rsna-knee-llm-labels`](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels) |
| `processed/contrastive-cohort-gold-holdout-v1.jsonl` | Not public; see the README's Reproducing the Results |

Build the two gold files from `train.csv` (this reproduces ours byte for byte):

```bash
.venv/bin/python -c "
import pandas as pd
train = pd.read_csv('data/train.csv')
gold = train.dropna(subset=['ACL'])
gold.to_csv('data/gold_label.csv')
gold.assign(**{c: None for c in gold.columns[2:]}).to_csv('data/gold_unlabeled.csv')
"
```

The DICOMs go wherever `data_root` in `paths.json` points.

Committed result files (under `digest/`, in notebooks and in scripts) never contain real study
UIDs. They refer to studies by a pseudonym, `study-<16 hex>`: the first 16 hex characters of the
SHA-256 of the `StudyInstanceUID`. If you hold the data, map them back with
`pd.read_csv('data/train.csv').assign(pseudonym=lambda d: d.StudyInstanceUID.map(lambda u: 'study-' + hashlib.sha256(u.encode()).hexdigest()[:16]))`,
or call `contrastive_learning.manifest.study_pseudonym`.
