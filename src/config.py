MODE = 'dl'

ML_MODELS = ['lr', 'rf', 'svm', 'xgb']

DL_MODEL = 'bilstm'   # 🔥 change here

DATA_PATH = 'data/emails_labeled_silver_tenK.parquet'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'

EPOCHS = 5
BATCH_SIZE = 16
MAX_LEN = 200
VOCAB_SIZE = 10000