# Enron Email Disclosure Analysis 📧⚖️

A research-grade NLP pipeline designed to detect "disclosure" in the Enron email corpus. This project implements a multi-phase workflow ranging from data validation and feature engineering to evaluation using both traditional Machine Learning (ML) and modern Deep Learning (DL) architectures.

## 🚀 Overview

The Enron Email Dataset contains over 500,000 emails from the early 2000s, reflecting one of the largest corporate collapses in history. This project specifically focuses on identifying emails that contain significant disclosures, helping to map out the timeline of the Enron crisis.

### Key Features
*   **Dual Mode Support:** Switch seamlessly between **ML Mode** (Logistic Regression, Random Forest, SVM, XGBoost) and **DL Mode** (BERT, BiLSTM).
*   **Comprehensive Pipeline:** Structured phases from validation to error analysis.
*   **Feature Engineering:** Advanced feature extraction including metadata processing and word count analysis.
*   **Evaluation Suite:** Detailed metrics including Precision, Recall, F1-Score, and Confusion Matrices.

---

## 📂 Project Structure

```text
NLP3/
├── data/               # Parquet/CSV datasets (Git Ignored)
├── models/             # Saved model weights/binaries (Git Ignored)
├── results/            # Performance plots and comparison CSVs (Git Ignored)
├── src/                # Project Source Code
│   ├── models/         # ML and DL model definitions
│   ├── vectorizers/    # Text vectorization (TF-IDF, BERT Tokenization)
│   ├── config.py       # Centralized configuration
│   ├── pipeline.py     # Main execution entry point
│   └── phase*_*.py     # Modular pipeline phases
├── .gitignore          # Excludes large binaries and envs
├── requirements.txt    # Python dependencies
└── README.md           # This file!
```

---

## 🛠️ Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd NLP3
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv pipeline_venv
    source pipeline_venv/bin/activate  # Windows: .\pipeline_venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚙️ Configuration

You can configure the pipeline's behavior in `src/config.py`:

*   `MODE`: Set to `'ml'` or `'dl'`.
*   `ML_MODELS`: List of models to run in ML mode (`lr`, `rf`, `svm`, `xgb`).
*   `DL_MODEL`: Current model for DL mode (`bert`, `bilstm`).
*   `DATA_PATH`: Path to the silver-labeled dataset.

---

## 🏃 Running the Pipeline

To execute the entire analysis workflow:

```bash
python src/pipeline.py
```

This will run all selected phases:
1.  **Phase 1 (Validate):** Data integrity checks.
2.  **Phase 2 (Preprocess):** Text cleaning and deduplication.
3.  **Phase 2b (Features):** Metadata feature extraction (ML Mode).
4.  **Phase 3 (Vectorize):** TF-IDF or BERT Tokenization.
5.  **Phase 4 (Training):** Model training and persistence.
6.  **Phase 5 (Evaluate):** Metric generation.
7.  **Phase 6 (Error Analysis):** Sample inspection of False Positives/Negatives.

---

## 📊 Research Notes: The Enron Crisis

Based on our analysis, the email volume shows significant spikes during key events:
*   **May 2001:** Peak pre-crisis communication.
*   **Aug 14, 2001:** Jeffrey Skilling resigns.
*   **Oct 2001:** SEC Investigation begins (**Massive Spike in Volume**).
*   **Dec 2, 2001:** Enron files for bankruptcy.

The models are trained to differentiate routine business correspondence from critical disclosures occurring during these panic periods.

---

## 📝 License
This project is developed for research purposes. Data source: Enron Email Dataset.
