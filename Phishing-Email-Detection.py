import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt

SUSPICIOUS_KEYWORDS = [
    'account', 'verify', 'login', 'password', 'urgent', 'security',
    'bank', 'paypal', 'confirm', 'click', 'update', 'billing', 'refund',
    'invoice', 'suspend', 'limited', 'alert', 'unauthorized', 'wire transfer',
    'social security', 'ssn', 'tax', 'winner', 'prize', 'lottery'
]
URL_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
EXCESSIVE_PUNCTUATION_PATTERN = re.compile(r"[!?]{2,}")
UPPERCASE_WORD_PATTERN = re.compile(r"\b[A-Z]{2,}\b")


def build_sample_dataset() -> pd.DataFrame:
    emails = [
        "Dear customer, your account has been suspended. Click http://secure-example.com to verify now.",
        "Your PayPal account has limited access. Please login at https://paypal.com.security-verify.example to update your payment information.",
        "Final notice: your mailbox is full. Visit http://mailbox.example.com to avoid service interruption.",
        "We detected suspicious activity on your bank account. Confirm your identity immediately.",
        "Important: refund request pending. Click here to resolve your billing issue.",
        "Your package delivery failed. Provide your address and tracking information to avoid return.",
        "Congratulations! You are a winner. Claim your prize by visiting the secure link below.",
        "Verify your password to continue using our secure service.",
        "Update required: your software license will expire unless you confirm today.",
        "Unauthorized login attempt detected. Review your recent activity now.",
        "Hello team, attached is the meeting agenda for next week.",
        "Please find the quarterly report attached and let me know if you have any questions.",
        "The invoice for your recent purchase is attached. Thank you for your business.",
        "Can you review the document and share your feedback by Friday?",
        "Lunch tomorrow at 12pm? Let me know if that works for you.",
        "Our support team has processed your request and will update you shortly.",
        "Reminder: the project kickoff is scheduled for Monday at 9am.",
        "Here is the login information for the new internal portal.",
        "Your subscription renewal has been processed successfully.",
        "Thanks for registering. Please save this confirmation for future reference.",
        "Security update completed. No further action is required.",
        "Your recent order has shipped and tracking details are available.",
        "We have received your application and will contact you within 3 business days.",
        "Thank you for attending the webinar. A recording is available on our site.",
        "Could you please approve the attached budget worksheet?",
        "Friendly reminder: complete the compliance training module by the end of the week.",
        "Your delivery is scheduled for tomorrow between 2pm and 5pm.",
        "Meeting rescheduled to Thursday. Please see the updated calendar invite.",
        "I appreciate your help with the customer escalation this morning.",
        "Please verify the attached time sheet before we submit payroll.",
    ]

    labels = [
        'Phishing', 'Phishing', 'Phishing', 'Phishing', 'Phishing',
        'Phishing', 'Phishing', 'Phishing', 'Phishing', 'Phishing',
        'Safe', 'Safe', 'Safe', 'Safe', 'Safe',
        'Safe', 'Safe', 'Safe', 'Safe', 'Safe',
        'Safe', 'Safe', 'Safe', 'Safe', 'Safe',
        'Safe', 'Safe', 'Safe', 'Safe', 'Safe',
    ]

    return pd.DataFrame({'email': emails, 'label': labels})


def normalize_label(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {'phishing', 'malicious', 'spam', '1', 'true', 'yes'}:
            return 'Phishing'
        if value in {'safe', 'legitimate', 'ham', '0', 'false', 'no'}:
            return 'Safe'
    try:
        numeric = int(value)
        return 'Phishing' if numeric == 1 else 'Safe'
    except (TypeError, ValueError):
        pass
    raise ValueError(f"Unknown label value: {value}")


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'email' not in df.columns and 'text' in df.columns:
        df = df.rename(columns={'text': 'email'})
    if 'email' not in df.columns or 'label' not in df.columns:
        raise ValueError('Dataset file must contain columns named "email" and "label".')

    df = df[['email', 'label']].dropna()
    df['label'] = df['label'].apply(normalize_label)
    return df


def count_urls(text: str) -> int:
    return len(URL_PATTERN.findall(text))


def count_suspicious_keywords(text: str) -> int:
    lowered = text.lower()
    return sum(int(keyword in lowered) for keyword in SUSPICIOUS_KEYWORDS)


def count_uppercase_words(text: str) -> int:
    return len(UPPERCASE_WORD_PATTERN.findall(text))


def count_excessive_punctuation(text: str) -> int:
    return len(EXCESSIVE_PUNCTUATION_PATTERN.findall(text))


def extract_meta_features(X):
    if isinstance(X, np.ndarray):
        email_texts = [str(item) for item in X.ravel()]
    else:
        email_texts = [str(item) for item in np.array(X).ravel()]

    features = []
    for email in email_texts:
        content = email.strip()
        features.append([
            count_urls(content),
            count_suspicious_keywords(content),
            len(re.findall(r"\d", content)),
            len(content.split()),
            count_uppercase_words(content),
            count_excessive_punctuation(content),
            int('urgent' in content.lower()),
            int('bank' in content.lower()),
            int('password' in content.lower()),
        ])

    return np.asarray(features, dtype=float)


def build_model_pipeline() -> Pipeline:
    transformer = ColumnTransformer(
        transformers=[
            (
                'text',
                TfidfVectorizer(max_features=800, ngram_range=(1, 2), stop_words='english'),
                'email',
            ),
            (
                'meta',
                FunctionTransformer(extract_meta_features, validate=False),
                ['email'],
            ),
        ],
        remainder='drop',
    )

    return Pipeline([
        ('features', transformer),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
    ])


def evaluate_model(model: Pipeline, X_train, X_test, y_train, y_test, show_plot: bool = False) -> None:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy:.2%}')
    print('\nClassification report:')
    print(classification_report(y_test, predictions, target_names=['Phishing', 'Safe'], zero_division=0))

    cm = confusion_matrix(y_test, predictions, labels=['Phishing', 'Safe'])
    print('Confusion matrix:')
    print(cm)

    if show_plot:
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xticks([0, 1], ['Phishing', 'Safe'])
        plt.yticks([0, 1], ['Phishing', 'Safe'])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train and evaluate a phishing email detector with Scikit-learn.'
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        help='Optional CSV dataset file containing columns "email" and "label".',
    )
    parser.add_argument(
        '--show-plot',
        action='store_true',
        help='Display a confusion matrix plot after training.',
    )
    args = parser.parse_args()

    if args.dataset:
        print(f'Loading dataset from: {args.dataset}')
        dataset = load_dataset(args.dataset)
    else:
        print('No dataset provided. Using built-in sample dataset.')
        dataset = build_sample_dataset()

    X = dataset[['email']]
    y = dataset['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    model = build_model_pipeline()
    evaluate_model(model, X_train, X_test, y_train, y_test, show_plot=args.show_plot)


if __name__ == '__main__':
    main()
