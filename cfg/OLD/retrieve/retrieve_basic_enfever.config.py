from datetime import datetime
from pathlib import Path

def config():
    FEVER_ROOT = "/mnt/data/factcheck/fever/data-en-lrev/fever-data"
    SPLIT = "paper_test"
    FEVER_CLAIMS = Path(FEVER_ROOT, f"{SPLIT}.jsonl")

    # TRAIN_DATA= "fever"
    # TRAIN_DATA= "qacg"
    TRAIN_DATA= "fever+qacg"

    COLBERT_ROOT = "/mnt/data/factcheck/fever/data-en-lrev/colbertv2"

    MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = MODEL_NAME.replace("/", "_")
    # CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini+minilm_230424_120755/colbert") # FEVER
    # CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini+minilm_230424_111951/colbert-20000") # QACG
    CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini+minilm_230426_180605/colbert") # FEVER + QACG

    INDEX_BITS = 2
    INDEX_ROOT = Path(COLBERT_ROOT, TRAIN_DATA, "indices")
    INDEX_NAME = Path(CHECKPOINT_NAME, f"{INDEX_BITS}bits")
    INDEX_PATH = Path(INDEX_ROOT, INDEX_NAME)

    ID2PID = Path(COLBERT_ROOT, "original_id2pid.json")

    PREDICTIONS_ROOT = Path(COLBERT_ROOT, TRAIN_DATA, "predictions", INDEX_NAME)

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Predictions on LREV EnFEVER for FEVER+QACG trained ColBERT."

    return {
        "root": PREDICTIONS_ROOT,
        "index_path": INDEX_PATH,
        "id2pid": ID2PID,
        "split": SPLIT,
        "claims": FEVER_CLAIMS,
        "k": 500,
        "note": NOTE,
        "date": DATESTR,
    }
