from datetime import datetime
from pathlib import Path

def config():
    MAX_K = 20
    K_VALUES = [1, 2, 3, 5, 10, 20]

    TEST_NAME = "qacg"
    LANG = "cs"
    DATE = "20230801"

    CORPUS_ROOT = f"/mnt/data/factcheck/wiki/{LANG}/{DATE}"
    CORPUS_FILE = f"{CORPUS_ROOT}/paragraphs/{LANG}wiki-{DATE}-paragraphs.jsonl"

    ER_MODEL_NAME = "bert-base-multilingual-cased"
    ER_CHECKPOINT_NAME = Path(ER_MODEL_NAME, "nway32_anserini_balanced_230909_220224/colbert")

    NLI_MODEL_ROOT = "/home/drchajan/devel/python/FC/Zero-shot-Fact-Verification/experiments/nli/"
    NLI_MODEL_NAME = "deepset/xlm-roberta-large-squad2_sum_cs_en_pl_sk-20230801_balanced_lr1e-6/checkpoint-321184_calibrated"
    NLI_MODEL_PATH = Path(NLI_MODEL_ROOT, NLI_MODEL_NAME)
    NLI_MAX_LENGTH = 512

    COLBERT_ROOT = Path(CORPUS_ROOT, f"colbertv2/qacg")
    INDEX_BITS = 2
    INDEX_NAME = Path(ER_CHECKPOINT_NAME, f"{INDEX_BITS}bits")
    SRC_PREDICTIONS_ROOT = Path(COLBERT_ROOT, "predictions", TEST_NAME, INDEX_NAME)
    DST_PREDICTIONS_ROOT = Path(COLBERT_ROOT, "predictions", TEST_NAME, INDEX_NAME, "nli_reranked", NLI_MODEL_NAME)

    SRC_PREDICTIONS = Path(SRC_PREDICTIONS_ROOT, "predictions.jsonl")
    CLAIM2EVIDENCE = Path(DST_PREDICTIONS_ROOT, "claim2evidence.jsonl")
    DST_PREDICTIONS_PREFIX = Path(DST_PREDICTIONS_ROOT, "predictions")

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Predictions on QACG test data for QACG-CS trained ColBERTv2 reranked by NLI model."

    return {
        "corpus_file": CORPUS_FILE,
        "src_predictions": SRC_PREDICTIONS,
        "claim2evidence": CLAIM2EVIDENCE,
        "dst_predictions_root": DST_PREDICTIONS_ROOT,
        "dst_predictions_prefix": DST_PREDICTIONS_PREFIX,
        "nli_model_path": NLI_MODEL_PATH,
        "nli_max_length": NLI_MAX_LENGTH,
        "max_k": MAX_K,
        "k_values": K_VALUES,
        "note": NOTE,
        "date": DATESTR,
    }
