from datetime import datetime
from pathlib import Path

def config():
    # LEVEL = "document"
    LEVEL = "block"

    TEST_NAME = "feverfever_over_qacg"
    SPLIT = "test_deepl"
    FEVER_ROOT = "/mnt/data/factcheck/fever/data-cs-lrev/fever-data"
    FEVER_CLAIMS = Path(FEVER_ROOT, f"{SPLIT}.jsonl")

    COLBERT_ROOT = "/mnt/data/factcheck/wiki/cs/20230801/colbertv2/qacg"

    MODEL_ROOT = Path("/mnt/data/factcheck/fever/data-cs-lrev/colbertv2")
    MODEL_NAME = "bert-base-multilingual-cased"
    CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_231028_093104/colbert-best_eval")

    INDEX_BITS = 2
    INDEX_ROOT = Path(COLBERT_ROOT, "indices", "csfever")
    INDEX_NAME = Path(CHECKPOINT_NAME, f"{INDEX_BITS}bits")
    INDEX_PATH = Path(INDEX_ROOT, INDEX_NAME)

    ID2PID = Path(COLBERT_ROOT, "original_id2pid.json")

    PREDICTIONS_ROOT = Path(COLBERT_ROOT, "predictions", TEST_NAME, INDEX_NAME)

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Predictions on CsFEVER test data claims on WIKI-CS corpus for CsFEVER ColBERTv2. Aimed for human annotations."
    
    return {
        "root": PREDICTIONS_ROOT,
        "index_path": INDEX_PATH,
        "id2pid": ID2PID,
        "split": SPLIT,
        "claims": FEVER_CLAIMS,
        "k": 500,
        "level": LEVEL,
        "note": NOTE,
        "date": DATESTR,
    }
