# Dataset

This project uses the **Industry Biscuit (Cookie) dataset**, a third-party dataset
hosted on Kaggle. It is **not committed to this repository** — the image files are
kept out of Git (see the root `.gitignore`), so you need to download the dataset
yourself before running the notebook.

- **Source:** https://www.kaggle.com/datasets/imonbilk/industry-biscuit-cookie-dataset
- **License:** GNU Lesser General Public License v3.0 (LGPL-3.0), as stated on the
  Kaggle dataset page. The dataset is the property of its original authors; this
  repository neither redistributes it nor relicenses it.
- **Approximate size:** ~44 MB (about 4,900 JPEG images plus an annotations CSV).

## What the data looks like

After extraction, the dataset has this structure:

```
biscuit_dataset/
└── IndustryBiscuit/
    ├── Images/            # ~4,900 .jpg images named 0001.jpg … 4900.jpg
    ├── Annotations.csv    # one row per image
    └── DatasetFolder.py   # helper shipped with the Kaggle dataset
```

`Annotations.csv` has three columns:

| column             | meaning                                                            |
| ------------------ | ------------------------------------------------------------------ |
| `file`             | image file name, e.g. `0908.jpg`                                   |
| `classDescription` | one of `Defect_No`, `Defect_Shape`, `Defect_Object`, `Defect_Color`|
| `classCode`        | integer code: `0`, `1`, `2`, `3`                                   |

The notebook collapses these four fine-grained classes into a **binary** label:
`Defect_No` → `0` (normal), any defect type → `1` (defective).

## Option A — Download via the Kaggle API (what the notebook does)

1. Install the Kaggle CLI (already in `requirements.txt`):
   ```bash
   pip install kaggle
   ```
2. Create a Kaggle API token (Kaggle → *Account* → *Create New API Token*). Do **not**
   paste the token into the notebook or commit it anywhere. Instead, either place the
   downloaded `kaggle.json` in `~/.kaggle/kaggle.json`, or export the token as an
   environment variable before launching the notebook:
   ```bash
   export KAGGLE_TOKEN="your_token_here"
   ```
   The notebook reads the token from `os.environ["KAGGLE_TOKEN"]` — it contains no
   hardcoded credentials.
3. Download and unzip:
   ```bash
   kaggle datasets download -d imonbilk/industry-biscuit-cookie-dataset -p . --unzip
   ```

## Option B — Download manually

1. Open the dataset page and click **Download** (requires a free Kaggle account).
2. Unzip it so that the `IndustryBiscuit/` folder sits inside a `biscuit_dataset/`
   folder, matching the structure shown above.

## Telling the notebook where the data is

The notebook's **Environment setup** cell defines `DATASET_DIR`:

- **On Google Colab** it mounts Google Drive and expects the data at
  `MyDrive/Colab_Notebooks/vision_for_insdusrty_project/biscuit_dataset`.
- **Locally** it defaults to a `biscuit_dataset/` folder next to the notebook. To use
  a different location, set an environment variable before launching Jupyter:
  ```bash
  export BISCUIT_DATA_DIR="/absolute/path/to/biscuit_dataset"
  ```

Everything downstream (`DATA_ROOT`, `IMAGES_DIR`, `CSV_PATH`) is derived from
`DATASET_DIR`, so this is the only place you need to set the path.
