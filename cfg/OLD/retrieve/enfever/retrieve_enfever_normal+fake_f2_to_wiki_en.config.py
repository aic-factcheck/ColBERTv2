from datetime import datetime
from pathlib import Path

def config():
    # LEVEL = "document"
    LEVEL = "block"

    TEST_NAME = "enfever"
    SPLIT_ROOT = "/mnt/data/factcheck/wiki/en/20230801/qacg/splits/stanza/mt5-large_all-cp126k/mt5-large_all-cp156k"
    SPLIT = "test_balanced"
    FEVER_CLAIMS = Path(SPLIT_ROOT, f"{SPLIT}.jsonl")

    COLBERT_ROOT = Path("/mnt/data/factcheck/wiki/en/20230801/colbertv2/qacg")

    MODEL_NAME = "bert-base-multilingual-cased"
    MODEL_NAME = MODEL_NAME.replace("/", "_")
    CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_231025_195145/colbert")

    NBITS = 2
    INDEX_ROOT = Path(COLBERT_ROOT, "indices", TEST_NAME)
    INDEX_NAME = Path(CHECKPOINT_NAME, f"{NBITS}bits")
    INDEX_PATH = Path(INDEX_ROOT, INDEX_NAME)

    ID2PID = Path(COLBERT_ROOT, "original_id2pid.json")

    PREDICTIONS_ROOT = Path(COLBERT_ROOT, "predictions", TEST_NAME, INDEX_NAME)

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Predictions on QACG EN for EnFEVER LREV trained ColBERTv2."

    return {
        "root": PREDICTIONS_ROOT,
        "index_path": INDEX_PATH,
        "id2pid": ID2PID,
        "split": SPLIT,
        "claims": FEVER_CLAIMS,
        "k": 500,
        "level": LEVEL,
        "replace_underscore": " ", 
        "note": NOTE,
        "date": DATESTR,
    }
