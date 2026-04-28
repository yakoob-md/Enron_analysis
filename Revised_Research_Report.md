# Project Report: Enron Email Disclosure Analysis

**SUBMITTED IN THE PARTIAL FULFILLMENT REQUIREMENT FOR THE AWARD OF DEGREE OF**
Bachelor of Technology
(Computer Science and Engineering)

**SUBMITTED BY**
Shruti Keshri-230862
Ayush Pratap singh-230881
Vanshik Soni-230
Yakub-230477

**UNDER THE SUPERVISION OF**
Dr. Atul Mishra
School of Engineering and Technology  

BML MUNJAL UNIVERSITY Gurugram, 
Haryana - 122413
MAY-2026

---

## 1. Introduction 

In today's corporate landscape, identifying sensitive or classified information within vast archives of daily email communication poses a significant challenge. The sheer volume of correspondence makes manual auditing nearly impossible. Consequently, there is a growing need for intelligent systems capable of distinguishing routine conversations from critical legal or financial disclosures. By leveraging Natural Language Processing (NLP), we can transform unstructured email data into structured, actionable insights that help safeguard organizations.

This project investigates the effectiveness of various machine learning and deep learning models in detecting corporate disclosures. We propose a hybrid text-processing approach that combines traditional frequency-based lexical patterns with advanced contextual embeddings. Over the course of this study, we developed two distinct classification frameworks: a binary system to detect the presence of a disclosure, and a multiclass system to categorize it into specific domains such as Legal, Strategic, or Financial. We evaluate traditional algorithms like Logistic Regression and Random Forest alongside modern transformer-based architectures, specifically BERT, to determine the optimal approach for this task.

Automating this detection process carries substantial implications for legal compliance and internal auditing. Overlooking a single critical email can expose companies to severe regulatory penalties and litigation. By treating critical disclosures as a "needle in a haystack," this research highlights how AI can serve as a continuous monitoring mechanism, ensuring that high-priority communications are flagged immediately.

The Enron Email Corpus, which serves as our primary dataset, presents several distinct challenges. First, true disclosures are exceptionally rare; out of a thousand emails, only a handful may contain sensitive information, creating a severe class imbalance. Furthermore, corporate language is inherently ambiguous—words like "lead" or "strike" carry different meanings depending on context. Finally, because the dataset stems from the 2001 Enron scandal, the models must be trained to recognize underlying semantic patterns rather than simply memorizing specific names and dates from that era.

To address these issues, we implemented a comprehensive preprocessing and classification pipeline. The text was rigorously cleaned, with hyperlinks removed and numerical values replaced by generic tags to prevent overfitting. Feature extraction relied on a combination of TF-IDF and BERT embeddings. Furthermore, we integrated cost-sensitive learning to ensure the models heavily penalize false negatives, acknowledging that missing a genuine disclosure is far more detrimental than a false alarm.

The remainder of this report is structured as follows: Section 2 reviews the existing literature on computational text analysis. Section 3 details our methodology, covering data preprocessing, feature engineering, and model architecture. Section 4 presents our experimental results and visual analysis. Finally, Section 5 offers concluding remarks and directions for future research.

## 2. Literature Review 

A rich body of literature has explored the use of computational methods to analyze communication archives, detect organizational crises, and uncover deceptive behavior. Reviewing these studies provides a critical technical foundation for understanding how high-stakes information can be extracted from unstructured text.

Previous research has successfully applied various machine learning techniques to derive meaning from massive digital datasets. Notable approaches include Support Vector Machines (SVM) for classifying deceptive identities, Latent Dirichlet Allocation (LDA) for topic modeling in health crises, and Sentiment Analysis for tracking emotional shifts in corporate emails. Learning-to-Rank models have also been utilized to prioritize critical messages in overloaded inboxes. These algorithms are frequently evaluated on three primary datasets: the Enron Email Corpus, Twitter streams, and specialized Identity Deception Datasets.

The relationship between these methods and datasets is highly specialized. For example, network centrality measures applied to the Enron corpus have shown that communication networks tend to centralize during a crisis. Similarly, SVMs have achieved up to 88% accuracy in detecting identity deception, while LDA on Twitter data has been instrumental in building early-warning systems for public health threats. Building upon these foundations, our study seeks to combine traditional feature engineering with modern deep learning to advance disclosure detection in corporate communications.

## 3. Methodology 

Our proposed methodology follows a structured, end-to-end NLP pipeline designed to identify and categorize corporate disclosures. This approach balances the interpretability of traditional linguistic features with the deep contextual understanding of state-of-the-art transformer models.

### 3.1 Data Ingestion and Cleaning
The pipeline begins with raw text files from the Enron Email Corpus. Given the high degree of noise—such as headers, timestamps, and signature blocks—the text undergoes rigorous cleaning. We convert all text to lowercase, strip special characters, and remove standard stopwords. A critical preprocessing step is 'Digit Replacement,' where specific numerical values are replaced with a generic tag. This forces the model to learn the financial context of a sentence rather than overfitting to specific monetary figures.

### 3.2 Tokenization and Normalization
Cleaned text is subsequently tokenized into discrete units. We apply morphological normalization to group related terms (e.g., 'disclose' and 'disclosure'). For the deep learning models, we employ the BERT WordPiece Tokenizer, which adeptly handles out-of-vocabulary words by breaking them down into known sub-word units, thereby preserving crucial semantic information.

### 3.3 Feature Extraction and Modeling Architecture
To capture both surface-level patterns and deep semantic intent, the system generates numerical representations through three concurrent paths:
1. **TF-IDF with Bigrams**: Captures the frequency and co-occurrence of word pairs.
2. **Hand-crafted Features**: Injects domain-specific knowledge, such as disclosure hit ratios and modal verb frequencies.
3. **Contextual Embeddings (BERT)**: Captures the nuanced intent behind professional language.

These features feed into a hierarchical classification system. Initially, a binary classifier filters out routine, non-disclosure emails. If an email is flagged, a multiclass classifier categorizes it into one of five domains: Strategic, Legal, Financial, Relational, or None. We employ cost-sensitive threshold tuning to prioritize recall, ensuring critical disclosures are captured.

The core of our deep learning approach utilizes the `BERT-base-uncased` architecture. BERT’s bidirectional self-attention mechanism is uniquely equipped to process the complex syntax of corporate language. We constructed a specialized classification head atop the 12-layer transformer, incorporating dropout layers to prevent overfitting and a final Softmax activation for the multiclass probability distribution.

## 4. Performance and Results

We evaluated our models across both binary and multiclass tasks using standard classification metrics: accuracy, precision, recall, and F1-score, supplemented by ROC-AUC and Precision-Recall (PR) curves. This section presents the visual and metric highlights of the project, focusing on the highest-performing architectures and the specific techniques used to overcome dataset challenges.

### 4.1 Binary Classification: Traditional ML vs. Deep Learning

The binary task aimed to distinguish disclosures from routine emails. In highly imbalanced datasets like Enron, standard accuracy is misleading. We relied on Precision-Recall metrics and cost-sensitive evaluations.

#### 4.1.1 Model Performance Overview
Our experiments revealed a stark contrast between baseline models and advanced architectures. XGBoost consistently outperformed SVM and Logistic Regression by effectively modeling the non-linear relationships in the sparse TF-IDF feature space. However, deep learning models, specifically BERT, provided the absolute highest contextual understanding, proving that pre-trained attention mechanisms are vastly superior at parsing the evasive language of corporate disclosures.

![Binary Model Comparison](c:/Users/dabaa/OneDrive/Desktop/dektop_content/NLP3/binary_pipeline/results/tables/binary/binary_model_comparison.png)
*Figure 1: Binary Model Comparison. This table explicitly highlights the dramatic improvements gained by shifting from standard thresholding (Youden) to Cost-Sensitive thresholds. By heavily penalizing False Negatives, we optimized for Recall (the critical metric for fraud detection), with XGBoost achieving an exceptional 0.9801 Recall.*

#### 4.1.2 The Power of Precision-Recall
For highly skewed datasets where routine emails heavily outnumber true disclosures, the Precision-Recall (PR) curve is the most rigorous diagnostic tool.

![PR Curve - XGBoost](c:/Users/dabaa/OneDrive/Desktop/dektop_content/NLP3/binary_pipeline/results/pr_xgb.png)
*Figure 2: Precision-Recall Curve for XGBoost. The large Area Under the Curve (Average Precision = 0.941) indicates a highly robust model. Unlike weaker models whose precision collapses as recall increases, XGBoost maintains strong predictive confidence across the entire threshold spectrum, effectively isolating the minority 'Disclosure' class without causing a flood of false alarms.*

![ROC Curve - BERT](c:/Users/dabaa/OneDrive/Desktop/dektop_content/NLP3/binary_pipeline/results/roc_bert.png)
*Figure 3: ROC Curve for BERT. The curve bows tightly into the top-left quadrant, yielding an impressive AUC. This confirms BERT's unparalleled discriminative power; it successfully utilizes contextual embeddings to separate nuanced disclosure discussions from benign corporate chatter.*

### 4.2 Optimizing Decision Thresholds

One of the primary strengths of this project is the explicit rejection of the default `0.5` decision threshold. A 0.5 threshold yields high accuracy but catastrophic recall for the critical minority class. By mathematically isolating the optimal threshold, we vastly improved the operational viability of the pipeline.

![Threshold Analysis - XGBoost](c:/Users/dabaa/OneDrive/Desktop/dektop_content/NLP3/binary_pipeline/results/threshold_xgb.png)
*Figure 4: Threshold vs. Metrics Curve (XGBoost). This graph maps how Precision (orange) and Recall (blue) react as the decision boundary shifts. We can visually pinpoint the exact optimal operating point—where the green F1-score curve reaches its global maximum (around threshold 0.35). This proves that lowering the threshold captures a massive influx of true disclosures before precision suffers.*

![F1 Threshold - XGBoost](c:/Users/dabaa/OneDrive/Desktop/dektop_content/NLP3/binary_pipeline/results/f1_threshold_xgb.png)
*Figure 5: F1-Score Optimization (XGBoost). Isolating the F1-score across the entire probability spectrum clearly demonstrates the mathematical peak (marked in red). Implementing this precise threshold ensures the best possible harmonic mean of positive predictive value and sensitivity.*

### 4.3 Multiclass Categorization and Class-Weight Tuning

The multiclass pipeline categorizes flagged emails into five distinct functional domains: NONE, STRATEGIC, RELATIONAL, LEGAL, and FINANCIAL. This task is inherently complex due to overlapping semantic boundaries (e.g., an email discussing the financial implications of a legal settlement).

#### 4.3.1 Resolving Minority Class Failure
Initially, our models suffered from the natural statistical bias of the training data, heavily favoring majority classes ('None' and 'Relational') while missing critical minority categories ('Strategic' and 'Legal'). We successfully mitigated this by implementing algorithmic class-weight tuning.

![Confusion Matrix - BiLSTM Argmax](c:/Users/dabaa/OneDrive/Desktop/dektop_content/NLP3/multiclass_pipeline/results/cm_bilstm_argmax.png)
*Figure 6: Baseline Confusion Matrix (BiLSTM). Using standard probability selection, the model performs well on dominant classes but suffers from noticeable false negatives on the 'Strategic' and 'Legal' categories (evidenced by the weaker diagonal blocks).*

![Confusion Matrix - BiLSTM Tuned](c:/Users/dabaa/OneDrive/Desktop/dektop_content/NLP3/multiclass_pipeline/results/cm_bilstm_tuned.png)
*Figure 7: Class-Weight Tuned Confusion Matrix (BiLSTM). After applying class-weight penalties for misclassifying minority categories, the diagonal alignment becomes significantly stronger. The tuned model successfully redistributes its predictive confidence, drastically reducing false negatives for crucial categories. This visual comparison proves the success of our class-balancing methodology.*

#### 4.3.2 Multiclass Discriminative Power
![Multiclass ROC - BERT](c:/Users/dabaa/OneDrive/Desktop/dektop_content/NLP3/multiclass_pipeline/results/plots/roc_multiclass_bert.png)
*Figure 8: One-vs-Rest ROC Curves for BERT. Each curve represents the model's ability to isolate one specific topic against all others. 'Legal' and 'Financial' consistently show unique, highly accurate profiles, indicating that regulatory and monetary terminology forms highly distinct semantic clusters that BERT effectively recognizes.*

### 4.4 Model Training, Generalization, and Feature Interpretability

A successful NLP pipeline must not only achieve high scores but also demonstrate that it is learning generalized linguistic patterns rather than simply memorizing the training dataset.

![Learning Curve - XGBoost](c:/Users/dabaa/OneDrive/Desktop/dektop_content/NLP3/binary_pipeline/results/learning_xgb.png)
*Figure 9: Learning Curve (XGBoost). By plotting training accuracy versus cross-validation accuracy across an increasing number of samples, we diagnose model variance. The converging gap between the two lines confirms that XGBoost is generalizing effectively to unseen data, rather than overfitting.*

![Deep Learning Training - BiLSTM](c:/Users/dabaa/OneDrive/Desktop/dektop_content/NLP3/binary_pipeline/results/training_bilstm.png)
*Figure 10: Deep Learning Training Trajectory (BiLSTM). The concurrent, steady decrease in both training and validation loss indicates healthy optimization. The point where validation loss stabilizes visually validates our use of Early Stopping to prevent premature overfitting on the Enron corpus.*

![Feature Importance - XGBoost](c:/Users/dabaa/OneDrive/Desktop/dektop_content/NLP3/multiclass_pipeline/results/plots/features_xgb.png)
*Figure 11: Feature Importance (XGBoost). This plot highlights the specific TF-IDF tokens and handcrafted linguistic features (such as modal verb frequency) that drive the model's decision-making process. This provides crucial transparency, proving the model is leveraging logical corporate terminology rather than anomalous noise.*

### 4.5 Benchmark Comparisons
Our results establish a formidable benchmark for the Enron corpus. Previous SVM-based topic classification efforts typically yielded F1 scores between 0.70 and 0.82. By incorporating robust feature engineering, our SVM baseline reached an F1 of 0.8571, validating the inclusion of domain-specific features like modal verb density. Furthermore, our fine-tuned BERT model sits at the upper echelon of contemporary transformer benchmarks for corporate email analysis. Ablation studies confirm that our holistic pipeline—combining stratified sampling, handcrafted features, and cost-sensitive tuning—drives this strong performance.

## 5. Conclusion

This study successfully implemented a comprehensive, multi-stage NLP pipeline to detect and categorize corporate disclosures within the Enron email corpus. By comparing distinct model architectures across binary and multiclass settings, we demonstrated that automated disclosure detection is highly feasible when supported by robust preprocessing, domain-informed feature engineering, and rigorous class imbalance mitigation.

Our findings unequivocally highlight BERT as the superior architecture for this task, primarily due to its ability to capture nuanced contextual semantics. However, traditional models remain highly relevant; XGBoost's exceptional recall makes it an ideal candidate for low-risk-tolerance investigative tasks, while Logistic Regression offers excellent interpretability.

While our approach yielded strong results, several limitations remain. The Enron dataset suffers from severe temporal imbalance, with a disproportionate number of emails clustered around late 2001, potentially limiting the model's generalization to pre-crisis communication. Additionally, the BiLSTM model's reliance on random weight initialization hindered its performance, reaffirming the necessity of pretrained embeddings for recurrent architectures.

Future research should focus on integrating robust pretrained word vectors (e.g., FastText or GloVe) into recurrent models, conducting temporal cross-validation, and testing the pipeline on external corporate datasets. Finally, establishing a zero-shot or few-shot classification baseline using Large Language Models (LLMs) would provide valuable insights into the trade-offs between task-specific fine-tuning and general foundational models.
