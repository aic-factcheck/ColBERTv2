from datetime import datetime
from pathlib import Path

def config():
    K_VALUES = [1, 2, 5, 10, 15, 20, 50] # top-k Anserini (or other retrieval) filtering source predictions (original order is preserved)

    LANG = "en"
    DATE = "20230801"
    CORPUS_ROOT = f"/mnt/data/factcheck/wiki/{LANG}/{DATE}"

    SRC_PREDICTIONS_ROOT = Path(CORPUS_ROOT, "colbertv2/qacg/predictions/enfever/bert-base-multilingual-cased/nway32_anserini_231028_093101/colbert-best_eval/2bits")
    FILTER_PREDICTIONS_ROOT = Path(CORPUS_ROOT, "anserini/predictions/qacg/test_balanced_shuf")
    DST_PREDICTIONS_ROOT = Path(SRC_PREDICTIONS_ROOT, "filter_anserini")

    SRC_PREDICTIONS = Path(SRC_PREDICTIONS_ROOT, "predictions.jsonl")
    FILTER_PREDICTIONS = Path(FILTER_PREDICTIONS_ROOT, "predictions.jsonl")
    DST_PREDICTIONS_PREFIX = Path(DST_PREDICTIONS_ROOT, "predictions")

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Predictions on QACG test data for EnFEVER trained ColBERTv2 filtered by Anserini."

    return {
        "src_predictions": SRC_PREDICTIONS,
        "filter_predictions": FILTER_PREDICTIONS,
        "dst_predictions_root": DST_PREDICTIONS_ROOT,
        "dst_predictions_prefix": DST_PREDICTIONS_PREFIX,
        "k_values": K_VALUES,
        "note": NOTE,
        "date": DATESTR,
    }
