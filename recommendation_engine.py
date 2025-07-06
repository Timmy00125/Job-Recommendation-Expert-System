import os
import re
import zipfile
from functools import lru_cache
from typing import Dict, List, Any, Set, Tuple

import nltk  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore


# ---------- Data Setup Helpers ----------
def ensure_data_unzipped(
    zip_path: str = "archive.zip", extract_dir: str = "data"
) -> str:
    """
    If archive.zip exists and our CSV isn’t already in data/,
    unzip it there.
    """
    # Peek inside archive.zip to find the CSV
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No CSV found inside archive.zip")
        csv_name = csv_files[0]

        target_path = os.path.join(extract_dir, os.path.basename(csv_name))
        if not os.path.exists(target_path):
            os.makedirs(extract_dir, exist_ok=True)
            print(f"Unzipping {zip_path} → {extract_dir}/")
            z.extract(csv_name, extract_dir)
            extracted = os.path.join(extract_dir, csv_name)
            if extracted != target_path:
                os.replace(extracted, target_path)
                # clean up empty dirs
                subdir = os.path.dirname(extracted)
                if subdir and os.path.isdir(subdir):
                    try:
                        os.rmdir(subdir)
                    except OSError:
                        pass
        else:
            print(f"{target_path} already exists, skipping unzip.")
    return target_path


# ---------- Data & Model Setup ----------
# 1. Load and preprocess data
def load_data(zip_path: str = "archive.zip") -> pd.DataFrame:  # type: ignore
    # Ensure data folder and CSV exist
    csv_path = ensure_data_unzipped(zip_path=zip_path, extract_dir="data")
    df = pd.read_csv(csv_path)  # type: ignore
    df.dropna(  # type: ignore
        subset=["Title", "JobDescription", "JobRequirment", "RequiredQual"],
        inplace=True,
    )
    df["Location"] = df["Location"].fillna("").astype(str)  # type: ignore
    df["IT"] = df["IT"].fillna(0).astype(int)  # type: ignore
    # Reset index to ensure continuous indices after dropna
    df.reset_index(drop=True, inplace=True)  # type: ignore
    return df


@lru_cache(maxsize=None)
def get_stopwords() -> Set[str]:
    nltk.download("stopwords", quiet=True)  # type: ignore
    return set(stopwords.words("english"))  # type: ignore


def clean_text(text: Any) -> str:
    stop_words = get_stopwords()
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join([w for w in text.split() if w not in stop_words])


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore
    df["text"] = (  # type: ignore
        df["Title"]
        + " "
        + df["JobDescription"]
        + " "
        + df["JobRequirment"]
        + " "
        + df["RequiredQual"]
    ).apply(clean_text)
    return df


# 2. TF-IDF
def create_vectorizer() -> TfidfVectorizer:  # type: ignore
    return TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # type: ignore


def vectorize_jobs(vectorizer: TfidfVectorizer, job_texts: pd.Series) -> Any:  # type: ignore
    return vectorizer.fit_transform(job_texts.values.astype("U"))  # type: ignore


# 3. Cosine Similarity
def compute_cosine_scores(  # type: ignore
    query: str, vectorizer: TfidfVectorizer, job_tfidf: Any
) -> np.ndarray:  # type: ignore
    vec = vectorizer.transform([clean_text(query)])  # type: ignore
    return cosine_similarity(vec, job_tfidf).flatten()  # type: ignore


# 4. Recommendation Logic (formerly Expert Rules & Scoring)
def recommend_jobs(
    query: str,
    prefs: Dict[str, Any],
    df: pd.DataFrame,  # type: ignore
    vectorizer: TfidfVectorizer,  # type: ignore
    job_tfidf: Any,
    alpha: float = 0.7,
    beta: float = 0.3,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    print("--- New Recommendation Request ---")
    print(f"Query: {query}, Prefs: {prefs}")

    print("1. Computing cosine scores...")
    cos: np.ndarray = compute_cosine_scores(query, vectorizer, job_tfidf)  # type: ignore

    print("2. Computing rule scores (vectorized)...")
    rules = np.zeros(len(df))  # type: ignore
    # Location rule
    loc = prefs.get("location", "").lower()  # type: ignore
    if loc:
        # Vectorized string comparison
        rules += df["Location"].str.lower().eq(loc).astype(float)  # type: ignore
    # IT preference rule
    if int(prefs.get("it_preference", 0)) == 1:  # type: ignore
        rules += df["IT"].astype(float)  # type: ignore

    print("3. Combining scores...")
    if rules.max() > 0:  # type: ignore
        rules = rules / rules.max()  # type: ignore # Normalize
    final = alpha * cos + beta * rules  # type: ignore

    print("4. Sorting and filtering results...")
    idx = np.argsort(final)[::-1]  # type: ignore

    # Ensure location match is prioritized if specified
    if loc:
        matched: List[int] = [  # type: ignore
            i
            for i in idx  # type: ignore
            if i < len(df) and loc in df.iloc[i]["Location"].lower()  # type: ignore
        ]
        others: List[int] = [i for i in idx if i < len(df) and i not in matched]  # type: ignore
        ordered = (matched + others)[:top_n]  # type: ignore
    else:
        ordered = idx[:top_n]  # type: ignore

    # Ensure all indices are valid
    ordered = [i for i in ordered if i < len(df)][:top_n]  # type: ignore

    result = df.iloc[ordered][["Title", "Company", "Location"]].copy()  # type: ignore
    result["score"] = final[ordered]  # type: ignore
    print("--- Recommendation Complete ---")
    return result.to_dict(orient="records")  # type: ignore


# Initialize model and data
def initialize_recommendation_system(
    zip_path: str = "archive.zip",
) -> Tuple[pd.DataFrame, TfidfVectorizer, Any]:  # type: ignore
    print("Initializing recommendation system...")
    # No longer need setup_nlp(), it's handled by get_stopwords cache
    print("1. Loading data...")
    df = load_data(zip_path=zip_path)
    print("2. Preparing data (cleaning text)...")
    df = prepare_data(df)
    print("3. Creating TF-IDF vectorizer...")
    vectorizer = create_vectorizer()
    print("4. Vectorizing job data...")
    job_tfidf = vectorize_jobs(vectorizer, df["text"])  # type: ignore
    print("Initialization complete.")
    return df, vectorizer, job_tfidf
