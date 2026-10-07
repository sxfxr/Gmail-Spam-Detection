# 📧 Gmail Spam Detection using Machine Learning

A machine learning-based email spam detection system that trains and evaluates classification models using the **UCI Spambase dataset** and integrates the trained classifier with the **Gmail API** for live inbox classification.

## 🚀 Project Overview

This project implements an end-to-end spam detection pipeline:

```text
UCI Spambase Dataset
        ↓
Data Preprocessing
        ↓
Train/Test Split
        ↓
Model Training
        ↓
Logistic Regression vs Random Forest
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Gmail API Integration
        ↓
Live Email Classification
```

The system compares multiple machine learning algorithms and selects the model with the best spam-class F1 score.

## 🧠 Machine Learning Models

Two classification algorithms were evaluated:

- Logistic Regression
- Random Forest

Random Forest produced the strongest overall spam-classification performance and was selected as the final model.

## 📊 Model Performance

Evaluation was performed on a stratified 20% test split of the UCI Spambase dataset.

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 92.18% | 89.01% | 91.46% | 90.22% |
| **Random Forest** | **94.68%** | **95.11%** | 91.18% | **93.11%** |

### Final Model

**Random Forest**

- Accuracy: **94.68%**
- Spam Precision: **95.11%**
- Spam Recall: **91.18%**
- Spam F1 Score: **93.11%**

The model was selected primarily using spam-class F1 rather than accuracy alone, providing a better balance between detecting spam and limiting false positives.

## 📂 Dataset

The model is trained using the **UCI Spambase dataset**.

The dataset contains:

- **4,601 email samples**
- **57 numerical features**
- **2 classes:** Ham and Spam
- **2,788 ham emails**
- **1,813 spam emails**

Features include word frequencies, character frequencies and capital-letter statistics commonly associated with spam messages.

The dataset is loaded programmatically using the `ucimlrepo` Python package.

## 📬 Gmail Integration

The project integrates with Gmail using the Gmail API and OAuth 2.0.

The Gmail component:

1. Authenticates securely using OAuth 2.0
2. Retrieves recent inbox messages
3. Extracts the email subject and body
4. Generates Spambase-style features
5. Passes the features to the trained Random Forest classifier
6. Predicts whether the message is **SPAM** or **NOT SPAM**
7. Displays the estimated spam probability

Example:

```text
EMAIL 1
Subject: Security alert
Prediction: NOT SPAM
Spam probability: 30.67%
```

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest
- Logistic Regression
- UCI Machine Learning Repository
- Gmail API
- Google OAuth 2.0
- BeautifulSoup
- Joblib

## 📁 Project Structure

```text
gmail-spam-detector/
│
├── train_model.py
├── gmail_spam_checker.py
├── extract_emails_to_csv.py
├── requirements.txt
├── .gitignore
└── README.md
```

Sensitive credentials, OAuth tokens, private email data and locally generated model artifacts are excluded from version control.

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/sxfxr/Gmail-Spam-Detection.git
cd Gmail-Spam-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🧪 Train the Model

Run:

```bash
python train_model.py
```

The script:

- downloads the UCI Spambase dataset
- performs a stratified train/test split
- trains Logistic Regression and Random Forest classifiers
- evaluates Accuracy, Precision, Recall and F1
- selects the model with the highest spam F1 score
- saves the trained model locally

## 🔐 Gmail API Setup

To use live Gmail classification:

1. Create a project in Google Cloud Console.
2. Enable the Gmail API.
3. Configure an OAuth consent screen.
4. Create OAuth credentials for a Desktop application.
5. Download the OAuth file and rename it to:

```text
credentials.json
```

6. Keep `credentials.json` inside the project directory.
7. Authenticate through Google when prompted.

OAuth credentials and tokens are excluded through `.gitignore` and must never be committed to the repository.

## ▶️ Run Gmail Spam Detection

After training the model and configuring Gmail OAuth:

```bash
python gmail_spam_checker.py
```

The program retrieves recent inbox messages and displays the predicted class and spam probability.

## ⚠️ Limitations

The reported evaluation metrics are measured on the **UCI Spambase test split**, not on a user's live Gmail inbox.

The Gmail integration converts live email content into Spambase-compatible features for inference. Because real-world Gmail messages can differ from the historical Spambase dataset, live predictions may experience dataset/domain shift.

The Gmail component should therefore be considered a demonstration of real-world model integration rather than a claim of 94.68% accuracy on live Gmail traffic.

## 🔒 Privacy & Security

This repository does **not** contain:

- Gmail passwords
- OAuth client credentials
- OAuth access/refresh tokens
- private email contents
- personal email datasets

The Gmail integration requests read-only access.

## 🎯 Key Learning Outcomes

This project demonstrates:

- Binary classification
- Feature-based spam detection
- Model comparison and evaluation
- Handling class imbalance using appropriate metrics
- Random Forest classification
- Logistic Regression
- Precision, Recall and F1 analysis
- REST API integration
- OAuth 2.0 authentication
- Applying an ML model to real-world data

## 👨‍💻 Author

**Mohammed Safar**  
B.Tech Computer Science & Engineering  
Rajagiri School of Engineering & Technology

GitHub: [@sxfxr](https://github.com/sxfxr)