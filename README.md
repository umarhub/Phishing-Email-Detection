# Phishing Email Detector

A simple Scikit-learn-based phishing email detection script.

## Features

- Trains on a dataset of phishing and legitimate emails
- Extracts email features such as URLs, suspicious keywords, uppercase words, and punctuation
- Classifies emails as `Phishing` or `Safe`
- Displays model accuracy, classification report, and confusion matrix
- Supports an optional CSV dataset input

## Requirements

- Python 3.14+
- numpy
- pandas
- scikit-learn
- matplotlib

Install dependencies with:

```bash
python -m pip install numpy pandas scikit-learn matplotlib
```

## Usage

Run the detector using the built-in sample dataset:

```bash
python phishing-emial-detector.py
```

Use a CSV dataset with `email` and `label` columns:

```bash
python phishing-emial-detector.py --dataset path/to/dataset.csv
```

Show a confusion matrix plot after training:

```bash
python phishing-emial-detector.py --show-plot
```

## Dataset Format

The CSV dataset must include:

- `email`: the email body text
- `label`: the classification label (`Phishing` / `Safe`, `malicious` / `legitimate`, or numeric `1`/`0`)

Example CSV rows:

```csv
email,label
"Please verify your account now.",Phishing
"Your order has shipped.",Safe
```

## Notes

- The current implementation uses a random forest classifier with balanced class weights.
- If no dataset is provided, the script uses a built-in sample dataset.
- The model can be improved with a larger labeled dataset and more advanced feature extraction.
