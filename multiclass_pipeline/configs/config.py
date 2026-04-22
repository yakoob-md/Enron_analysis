# multiclass_pipeline/configs/config.py
import os

MODE = 'ml' # 'ml', 'dl', 'llm'
DATA_PATH = 'multiclass_pipeline/data/emails_labeled_silver_tenK.parquet'
MODEL_DIR = 'multiclass_pipeline/saved_models'
RESULTS_DIR = 'multiclass_pipeline/results'

NUM_CLASSES = 5
CLASS_NAMES = ['NONE', 'STRATEGIC', 'RELATIONAL', 'LEGAL', 'FINANCIAL']

# Training params
EPOCHS = 5
BATCH_SIZE = 32
LR = 2e-5
MAX_LEN = 200
VOCAB_SIZE = 10000
