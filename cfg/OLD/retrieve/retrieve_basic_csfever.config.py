from datetime import datetime
from pathlib import Path

def config():
    FEVER_ROOT = "/mnt/data/factcheck/fever/data_full_nli-filtered-cs/fever-data/F1_titles_anserininew_threshold"
    SPLIT = "paper_test_fb_cs_nli_split_F1_titles_anserininew"
    FEVER_CLAIMS = Path(FEVER_ROOT, f"{SPLIT}.jsonl")

    # TRAIN_DATA= "fever"
    # TRAIN_DATA= "qacg"
    # TRAIN_DATA= "fever+qacg"
    TRAIN_DATA= "qacg-r"

    COLBERT_ROOT = "/mnt/data/factcheck/fever/data_full_nli-filtered-cs/colbertv2"

    MODEL_NAME = "bert-base-multilingual-cased"
    MODEL_NAME = MODEL_NAME.replace("/", "_")
    # CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_230528_094536/colbert") # FEVER
    # CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_230525_131126/colbert") # QACG
    # CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_230530_153224/colbert") # QACG, NFC corrected
    # CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_230605_112607/colbert") # FEVER+QACG
    CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_230608_192647/colbert") # QACG-R

    INDEX_BITS = 2
    INDEX_ROOT = Path(COLBERT_ROOT, TRAIN_DATA, "indices")
    INDEX_NAME = Path(CHECKPOINT_NAME, f"{INDEX_BITS}bits")
    INDEX_PATH = Path(INDEX_ROOT, INDEX_NAME)

    ID2PID = Path(COLBERT_ROOT, "original_id2pid.json")

    PREDICTIONS_ROOT = Path(COLBERT_ROOT, TRAIN_DATA, "predictions", INDEX_NAME)

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Predictions on CsFEVER (NLI filtered) for FEVER trained ColBERTv2."

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
