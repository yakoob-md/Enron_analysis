# Enron Email Disclosure Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced NLP pipeline designed to detect and categorize sensitive corporate disclosures within the Enron Email Corpus. This project implements a two-stage classification system using both traditional Machine Learning and State-of-the-Art Deep Learning architectures.

## 🚀 Overview

During the Enron scandal, critical evidence was hidden within millions of internal emails. This project builds an automated system to identify **disclosures**—emails revealing sensitive, non-public, or legally significant information—in real-time

### Two-Stage Pipeline
1.  **Binary Classification**: A high-recall filter that separates "Disclosure" emails from routine "Non-Disclosure" traffic.
2.  **Multiclass Classification**: Categorizes flagged emails into five functional domains:
    *   **NONE**: Routine communication.
    *   **STRATEGIC**: M&A, restructuring, and high-level decisions.
    *   **RELATIONAL**: Interpersonal coordination.
    *   **LEGAL**: SEC filings, attorney-client communication, and regulatory matters.
    *   **FINANCIAL**: Earnings, balance sheets, and write-downs.

---

## 🛠️ Key Technical Features

### 1. Hybrid Feature Fusion
Combines statistical lexical patterns with domain expertise:
*   **TF-IDF (ngram_range 1-2)**: 7,000 features capturing word and bigram frequencies.
*   **12 Handcrafted Domain Features**: Engineered signals including:
    *   `disclosure_hits`: Frequency of specific investigative keywords.
    *   `modal_count`: Density of legalistic verbs (*must, shall, should*).
    *   `uncertainty_count`: Risk-related language (*likely, possibly, risk*).
    *   `caps_ratio` & `has_dollar`: Formatting and financial content indicators.  

### 2. Model Zoo 
*   **Classical ML**: Logistic Regression, Random Forest, SVM, and **XGBoost** (tuned for high recall).
*   **Deep Learning**:  
    *   **BERT (base-uncased)**: Fine-tuned with a custom classification head and warm-up scheduling.
    *   **BiLSTM**: Bidirectional sequence modeling with Layer Normalization and Dropout.
ggg
### 3. Advanced Optimization 

*   **Imbalance Mitigation**: Implements **Balanced Class Weights** and **Multiclass Focal Loss** to handle skewed distributions.
*   **Dynamic Thresholding**: Rejects the default 0.5 boundary in favor of:
    *   **Youden’s J Statistic**: Equalizing error rates.
    *   **F1-Optimization**: Maximizing the harmonic mean of precision and recall.
    *   **Cost-Sensitive Analysis**: Minimizing False Negatives (critical for fraud detection).

---

## 📂 Repository Structure 

```text
NLP3/
├── binary_pipeline/          # 1st Stage: Disclosure vs. None
│   ├── src/                  # Core logic (Preprocessing, Features, Training)
│   ├── models/               # Model definitions (ML, BERT, BiLSTM)
│   └── results/              # ROC/PR curves and metrics
├── multiclass_pipeline/      # 2nd Stage: 5-Class Categorization
├── requirements.txt          # Dependency list
├── generate_report.py        # Automated PDF report generation
├── run_diagnostics.py        # Post-training error analysis
└── prompt.txt                # Technical project requirements
```

---

## 📊 Results Summary

| Model | Accuracy | F1-Score | Recall | Note |
| :--- | :--- | :--- | :--- | :--- |
| **BERT** | **0.8421** | **0.8924** | 0.8873 | Best overall semantic understanding |
| **XGBoost** | 0.8145 | 0.8801 | **0.9801** | Best for high-recall auditing |
| **RF** | 0.8210 | 0.8812 | 0.9145 | Solid ML baseline |
| **BiLSTM**| 0.7912 | 0.8654 | 0.8521 | Limited by random embeddings |

---

## ⚙️ Installation & Usage

### 1. Setup Environment
```bash
python -m venv pipeline_venv
source pipeline_venv/bin/activate  # Windows: .\pipeline_venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Pipeline
To run the full binary pipeline with ML models:
```bash
python binary_pipeline/src/pipeline.py
```
*Modify `binary_pipeline/src/config.py` to switch between `ml` and `dl` modes.*

---

## ⚠️ Known Limitations & Future Work

*   **Temporal Leakage**: The current train-test split is random; temporal cross-validation is required to test crisis-period generalization.
*   **BiLSTM Embeddings**: Currently uses random initialization. Integrating **GloVe/FastText** embeddings is a priority.
*   **Silver Labels**: The ground truth is programmatically generated; a human-verified "Gold Standard" subset is needed for final validation.
*   **LLM Benchmarking**: Zero-shot comparison against Large Language Models (LLMs) like GPT-4 or Claude-3 is planned.
