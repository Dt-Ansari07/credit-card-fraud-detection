# Data

This project uses the **Credit Card Transactions Fraud Detection Dataset** from Kaggle:

**Source:** https://www.kaggle.com/datasets/kartik2112/fraud-detection

The raw file (`fraudTrain.csv`, ~1.3M rows / ~350 MB) is **not committed to this repository** — it's too large for Git and easy to re-download.

## Download steps

1. Create a free Kaggle account if you don't have one, and set up the Kaggle API (`~/.kaggle/kaggle.json`) — see [Kaggle API docs](https://www.kaggle.com/docs/api).
2. Download the dataset:
   ```bash
   kaggle datasets download -d kartik2112/fraud-detection -p data/ --unzip
   ```
   (or download `fraudTrain.csv` manually from the Kaggle page above and place it in this `data/` folder).
3. Confirm the file is at `data/fraudTrain.csv`, then run the notebook in `notebooks/`.

The notebook expects exactly this filename and location: `data/fraudTrain.csv`.
