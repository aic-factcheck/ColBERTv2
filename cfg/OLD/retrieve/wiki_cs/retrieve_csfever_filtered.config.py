from datetime import datetime
from pathlib import Path

def config():
    LEVEL = "document"
    # LEVEL = "block"

    TEST_NAME = "csfever_filtered"
    FEVER_ROOT = "/mnt/data/factcheck/fever/data_full_nli-filtered-cs/fever-data/F1_titles_anserininew_threshold"
    SPLIT = "paper_test_fb_cs_nli_split_F1_titles_anserininew_CS-EVIDENCE"
    FEVER_CLAIMS = Path(FEVER_ROOT, f"{SPLIT}.jsonl")

    COLBERT_ROOT = "/mnt/data/factcheck/wiki/cs/20230801/colbertv2/qacg"

    MODEL_NAME = "bert-base-multilingual-cased"
    MODEL_NAME = MODEL_NAME.replace("/", "_")
    CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_balanced_230909_220224/colbert") # QACG

    INDEX_BITS = 2
    INDEX_ROOT = Path(COLBERT_ROOT, "indices")
    INDEX_NAME = Path(CHECKPOINT_NAME, f"{INDEX_BITS}bits")
    INDEX_PATH = Path(INDEX_ROOT, INDEX_NAME)

    ID2PID = Path(COLBERT_ROOT, "original_id2pid.json")

    PREDICTIONS_ROOT = Path(COLBERT_ROOT, "predictions", TEST_NAME, INDEX_NAME)

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Predictions on CsFEVER (NLI filtered) for QACG trained ColBERTv2."

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
