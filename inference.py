"""Faithful raw-review preprocessing and inference for the Chapter 4 model."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import string
import tempfile
import unicodedata
import warnings
from pathlib import Path

import contractions
import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from scipy import sparse


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "lgb_model_bundle.joblib"
MODEL_CHECK_PATH = APP_DIR / "models" / "lgb_model_bundle.verification.json"
EXPECTED_STOPWORDS_SHA256 = "47608d511aa4fec95139d41e487109ab4a260313745d397210dbf966b1d3c225"

FEATURES_14 = [
    "char_count",
    "word_count",
    "mean_word_len",
    "ttr",
    "rttr",
    "sentence_count",
    "mean_sentence_len",
    "std_sentence_len",
    "punct_count",
    "punct_density",
    "exclamation_count",
    "question_count",
    "caps_ratio",
    "stopword_ratio",
]
PUNCT_SET = set(string.punctuation)
URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", flags=re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]*>")
CONTROL_RE = re.compile(r"[\u0000-\u001f\u007f-\u009f]")
NON_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
MULTISPACE_RE = re.compile(r"\s+")
WORD_CLEAN_RE = re.compile(r"[a-zA-Z0-9]+")

# Kept consistent with the Chapter 2 text preprocessing notebook.
NEGATIONS = {
    "not", "no", "nor", "cannot", "can't", "couldn't", "didn't", "doesn't",
    "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
    "needn't", "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn me",
}


def ensure_nltk_resources() -> set[str]:
    """Download the same named NLTK resources used by the research notebooks."""
    nltk_dir = Path(tempfile.gettempdir()) / "amazon_fake_reviews_nltk"
    nltk_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NLTK_DATA"] = str(nltk_dir)
    if str(nltk_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_dir))

    resources = {
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet.zip/wordnet/",
        "omw-1.4": "corpora/omw-1.4.zip/",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    }
    for resource, lookup_path in resources.items():
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            if not nltk.download(resource, download_dir=str(nltk_dir), quiet=True):
                raise RuntimeError(f"Không tải được tài nguyên NLTK bắt buộc: {resource}")

    words = set(stopwords.words("english"))
    digest = hashlib.sha256("\n".join(sorted(words)).encode("utf-8")).hexdigest()
    if digest != EXPECTED_STOPWORDS_SHA256:
        raise RuntimeError(
            "Bộ stopwords không khớp với bộ dùng khi tạo đặc trưng Chương 2; "
            "dừng suy luận để tránh thay đổi kết quả."
        )
    return words


def clean_text(text: str) -> str:
    """Mirror preprocess_text_clean from Notebooks/02_TienXuLy_NLP.ipynb."""
    if text is None or not str(text).strip():
        return "empty"
    value = html.unescape(str(text))
    value = URL_RE.sub("", value)
    value = HTML_RE.sub("", value)
    value = contractions.fix(value)
    value = unicodedata.normalize("NFC", value)
    value = CONTROL_RE.sub("", value)
    value = NON_WORD_RE.sub(" ", value)
    value = MULTISPACE_RE.sub(" ", value).strip().lower()
    return value if value else "empty"


def penn_to_wordnet(tag: str):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_text(cleaned: str, stop_words: set[str]) -> str:
    """Mirror preprocess_text_lemmatized from Notebooks/02_TienXuLy_NLP.ipynb."""
    if cleaned == "empty" or not cleaned:
        return "empty"
    kept = [token for token in cleaned.split() if token not in (stop_words - NEGATIONS) and not token.isnumeric()]
    if not kept:
        return "empty"
    lemmatizer = WordNetLemmatizer()
    tagged = pos_tag(kept)
    result = " ".join(lemmatizer.lemmatize(token, penn_to_wordnet(tag)) for token, tag in tagged)
    return result if result.strip() else "empty"


def extract_stylometry(text: str, stop_words: set[str]) -> dict[str, float]:
    """Mirror extract_comprehensive_stylometry from Notebooks/01_KhamPha_va_Stylometry.ipynb."""
    value = str(text)
    char_count = len(value)
    raw_words = value.split()
    word_count = len(raw_words)
    clean_words = WORD_CLEAN_RE.findall(value.lower())
    clean_word_count = len(clean_words) if clean_words else 1
    total_word_chars = sum(map(len, clean_words))
    mean_word_len = total_word_chars / clean_word_count
    vocab_size = len(set(clean_words))
    ttr = vocab_size / (clean_word_count + 1e-5)
    rttr = vocab_size / (np.sqrt(clean_word_count) + 1e-5)

    raw_sentences = [part.strip() for part in re.split(r"[.!?]+", value) if part.strip()]
    sentence_count = len(raw_sentences)
    if sentence_count:
        sentence_lengths = [len(sentence.split()) for sentence in raw_sentences]
        mean_sentence_len = float(np.mean(sentence_lengths))
        std_sentence_len = float(np.std(sentence_lengths)) if sentence_count > 1 else 0.0
    else:
        sentence_count = 1
        mean_sentence_len = float(word_count)
        std_sentence_len = 0.0

    punct_count = sum(char in PUNCT_SET for char in value)
    return {
        "char_count": char_count,
        "word_count": word_count,
        "mean_word_len": mean_word_len,
        "ttr": ttr,
        "rttr": rttr,
        "sentence_count": sentence_count,
        "mean_sentence_len": mean_sentence_len,
        "std_sentence_len": std_sentence_len,
        "punct_count": punct_count,
        "punct_density": punct_count / (char_count + 1e-5),
        "exclamation_count": value.count("!"),
        "question_count": value.count("?"),
        "caps_ratio": sum(char.isupper() for char in value) / (char_count + 1e-5),
        "stopword_ratio": sum(token in stop_words for token in clean_words) / (clean_word_count + 1e-5),
    }


def make_raw14_frame(reviews: pd.DataFrame, stop_words: set[str]) -> pd.DataFrame:
    required = {"text", "rating", "category"}
    missing = required - set(reviews.columns)
    if missing:
        raise ValueError(f"Thiếu cột đầu vào: {', '.join(sorted(missing))}")

    rows = []
    for record in reviews.to_dict(orient="records"):
        text = str(record["text"])
        cleaned = clean_text(text)
        row = {
            "text": text,
            "text_clean": cleaned,
            "text_lemmatized": lemmatize_text(cleaned, stop_words),
            "rating": float(record["rating"]),
            "category": str(record["category"]).strip(),
        }
        row.update(extract_stylometry(text, stop_words))
        rows.append(row)
    frame = pd.DataFrame(rows)
    for feature in FEATURES_14:
        frame[feature] = pd.to_numeric(frame[feature], errors="raise")
    return frame


def verify_preprocessing_reference(stop_words: set[str]) -> None:
    """Fail closed if runtime NLP resources no longer reproduce source records."""
    presets = json.loads((APP_DIR / "preset_reviews.json").read_text(encoding="utf-8"))
    references = json.loads(
        (APP_DIR / "preset_preprocessing_reference.json").read_text(encoding="utf-8-sig")
    )
    expected_by_id = {int(item["review_id"]): item for item in references}
    for preset in presets:
        review_id = int(preset["source_review_id"])
        if review_id not in expected_by_id:
            raise RuntimeError(f"Thiếu preprocessing reference cho preset review_id={review_id}.")
        actual = make_raw14_frame(
            pd.DataFrame(
                [{"text": preset["text"], "rating": preset["rating"], "category": preset["category"]}]
            ),
            stop_words,
        ).iloc[0]
        expected = expected_by_id[review_id]
        if str(actual["text_clean"]) != str(expected["text_clean"]):
            raise RuntimeError(f"text_clean không khớp dữ liệu tham chiếu cho review_id={review_id}.")
        if str(actual["text_lemmatized"]) != str(expected["text_lemmatized"]):
            raise RuntimeError(f"text_lemmatized không khớp dữ liệu tham chiếu cho review_id={review_id}.")
        for feature, expected_value in expected["stylometry"].items():
            if not np.isclose(float(actual[feature]), float(expected_value), atol=1e-8, rtol=1e-7):
                raise RuntimeError(
                    f"Đặc trưng {feature} không khớp dữ liệu tham chiếu cho review_id={review_id}."
                )


def load_bundle():
    import sys
    import joblib

    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    import chapter3_transformers  # noqa: F401 - needed for joblib class resolution

    if not MODEL_PATH.exists() or not MODEL_CHECK_PATH.exists():
        raise FileNotFoundError("Model bundle hoặc tệp kiểm chứng không tồn tại.")
    import json

    check = json.loads(MODEL_CHECK_PATH.read_text(encoding="utf-8"))
    if not all(check.get(key) is True for key in ("prediction_match", "probability_match", "preprocessor_match")):
        raise RuntimeError("Tệp kiểm chứng model bundle chưa đạt.")
    bundle = joblib.load(MODEL_PATH)
    if bundle.get("label_mapping") != {"OR": 0, "CG": 1}:
        raise RuntimeError("Label mapping trong model bundle không đúng hợp đồng.")
    if len(bundle.get("feature_names", [])) != 5024:
        raise RuntimeError("Số đặc trưng trong model bundle không khớp 5.024.")
    return bundle


def infer_reviews(reviews: pd.DataFrame, bundle: dict, stop_words: set[str], explain: bool = False):
    raw14 = make_raw14_frame(reviews, stop_words)
    matrix = bundle["preprocessor"].transform(raw14)
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    if matrix.shape[1] != len(bundle["feature_names"]):
        raise RuntimeError("Số chiều suy luận không khớp schema của model.")
    model = bundle["model"]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        predictions = model.predict(matrix).astype(int)
        classes = list(model.classes_)
        positive_index = classes.index(1)
        probabilities = model.predict_proba(matrix)[:, positive_index]
    contributions = None
    if explain:
        contributions = model.booster_.predict(matrix, pred_contrib=True)
        if sparse.issparse(contributions):
            contributions = contributions.toarray()
        contributions = np.asarray(contributions)[:, :-1]
        if contributions.shape[1] != len(bundle["feature_names"]):
            raise RuntimeError("SHAP contributions không khớp schema đặc trưng.")
    return raw14, predictions, probabilities, contributions
