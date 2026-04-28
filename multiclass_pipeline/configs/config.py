# multiclass_pipeline/configs/config.py
import os

MODE = 'all' # 'ml', 'dl', 'bert', or 'all'
ML_MODELS = ['lr', 'rf', 'xgb']
DL_MODELS = ['bilstm', 'bert']
DATA_PATH = 'multiclass_pipeline/data/emails_labeled_silver_tenK.parquet'
MODEL_DIR = 'multiclass_pipeline/saved_models'
RESULTS_DIR = 'multiclass_pipeline/results'

NUM_CLASSES = 5
CLASS_NAMES = ['NONE', 'STRATEGIC', 'RELATIONAL', 'LEGAL', 'FINANCIAL']

# Training params
BATCH_SIZE = 32
VOCAB_SIZE = 20000
MAX_LEN = 128

# BERT specific
EPOCHS_BERT = 5
LR_BERT = 2e-5

# BiLSTM specific
EPOCHS_DL = 15
LR_DL = 1e-3
