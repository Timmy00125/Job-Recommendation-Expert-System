
import os
import re
import zipfile
from typing import Any, Dict, Generator, List, Tuple

import nltk  # type: ignore
import numpy as np
import pandas as pd
from experta import MATCH, DefFacts, Fact, Field, KnowledgeEngine, Rule  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore


# ---------- Data Setup Helpers ----------
def ensure_data_unzipped(
    zip_path: str = "archive.zip", extract_dir: str = "data"
) -> str:
    """
    If archive.zip exists and our CSV isn't already in data/,
    unzip it there.
    """
    # Peek inside archive.zip to find the CSV
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_files: List[str] = [f for f in z.namelist() if f.lower().endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No CSV found inside archive.zip")
        csv_name: str = csv_files[0]

        target_path: str = os.path.join(extract_dir, os.path.basename(csv_name))
        if not os.path.exists(target_path):
            os.makedirs(extract_dir, exist_ok=True)
            print(f"Unzipping {zip_path} → {extract_dir}/")
            z.extract(csv_name, extract_dir)
            extracted: str = os.path.join(extract_dir, csv_name)
            if extracted != target_path:
                os.replace(extracted, target_path)
                # clean up empty dirs
                subdir: str = os.path.dirname(extracted)
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
def load_data(zip_path: str = "archive.zip") -> pd.DataFrame:
    # Ensure data folder and CSV exist
    csv_path: str = ensure_data_unzipped(zip_path=zip_path, extract_dir="data")
    df: pd.DataFrame = pd.read_csv(csv_path)  # type: ignore
    df.dropna(  # type: ignore
        subset=["Title", "JobDescription", "JobRequirment", "RequiredQual"],
        inplace=True,
    )
    df["Location"] = df["Location"].fillna("").astype(str)  # type: ignore
    df["IT"] = df["IT"].fillna(0).astype(int)  # type: ignore
    return df


def setup_nlp() -> None:
    nltk.download("stopwords", quiet=True)  # type: ignore


def clean_text(text: Any) -> str:
    from nltk.corpus import stopwords  # type: ignore

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join([w for w in text.split() if w not in stopwords.words("english")])  # type: ignore


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = (
        df["Title"]
        + " "
        + df["JobDescription"]
        + " "
        + df["JobRequirment"]
        + " "
        + df["RequiredQual"]
    ).apply(clean_text)  # type: ignore
    return df


# 2. TF-IDF
def create_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(max_features=5000, ngram_range=(1, 2))


def vectorize_jobs(vectorizer: TfidfVectorizer, job_texts: pd.Series[Any]) -> Any:  # type: ignore
    return vectorizer.fit_transform(job_texts.values.astype("U"))  # type: ignore


# 3. Cosine Similarity
def compute_cosine_scores(
    query: str, vectorizer: TfidfVectorizer, job_tfidf: Any
) -> np.ndarray[Any, Any]:  # type: ignore
    vec = vectorizer.transform([clean_text(query)])  # type: ignore
    return cosine_similarity(vec, job_tfidf).flatten()  # type: ignore


# 4. Expert Rules
class JobFact(Fact):
    id = Field(int, mandatory=True)
    location = Field(str, mandatory=False)
    it_flag = Field(int, mandatory=False)


class UserPrefs(Fact):
    location = Field(str)
    it_preference = Field(int)


class JobExpert(KnowledgeEngine):
    @DefFacts()
    def start(self) -> Generator[Fact, None, None]:
        yield UserPrefs(location="", it_preference=0)

    @Rule(UserPrefs(location=MATCH.loc), JobFact(location=MATCH.loc))  # type: ignore
    def loc_match(self, loc: str) -> None:
        if loc and loc.strip():  # Only match non-empty locations
            self.declare(Fact(rule_score=1))  # type: ignore

    @Rule(UserPrefs(it_preference=1), JobFact(it_flag=1))  # type: ignore
    def it_match(self) -> None:
        self.declare(Fact(rule_score=1))  # type: ignore


# 5. Scoring
def compute_rule_score(row: pd.Series[Any], prefs: Dict[str, Any]) -> int:  # type: ignore
    eng = JobExpert()
    eng.reset()  # type: ignore
    eng.declare(  # type: ignore
        UserPrefs(
            location=str(prefs.get("location", "")),
            it_preference=int(prefs.get("it_preference", 0)),
        )
    )
    eng.declare(  # type: ignore
        JobFact(
            id=int(row.name or 0),  # type: ignore
            location=str(row["Location"]),
            it_flag=int(row["IT"]),  # type: ignore
        )
    )  # type: ignore
    eng.run()  # type: ignore
    return sum(
        1
        for f in eng.facts.values()  # type: ignore
        if isinstance(f, Fact) and f.get("rule_score", False)  # type: ignore
    )


# 6. Recommendation
def recommend_jobs(
    query: str,
    prefs: Dict[str, Any],
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    job_tfidf: Any,
    alpha: float = 0.7,
    beta: float = 0.3,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    cos: np.ndarray[Any, Any] = compute_cosine_scores(query, vectorizer, job_tfidf)  # type: ignore
    rules: np.ndarray[Any, Any] = np.array(  # type: ignore
        [compute_rule_score(r, prefs) for _, r in df.iterrows()],  # type: ignore
        dtype=float,  # type: ignore
    )
    if rules.max() > 0:  # type: ignore
        rules = rules / rules.max()  # type: ignore
    final: np.ndarray[Any, Any] = alpha * cos + beta * rules  # type: ignore
    idx: np.ndarray[Any, Any] = np.argsort(final)[::-1]  # type: ignore
    loc: str = prefs.get("location", "").lower()
    matched: List[int] = [
        i
        for i in idx
        if loc and loc in str(df.iloc[i]["Location"]).lower()  # type: ignore
    ]
    others: List[int] = [i for i in idx if i not in matched]
    ordered: List[int] = (matched + others)[:top_n]
    result: pd.DataFrame = df.iloc[ordered][["Title", "Company", "Location"]].copy()  # type: ignore
    result["score"] = final[ordered]
    return result.to_dict(orient="records")  # type: ignore  # type: ignore


# Initialize model and data
def initialize_recommendation_system(
    zip_path: str = "archive.zip",
) -> Tuple[pd.DataFrame, TfidfVectorizer, Any]:
    setup_nlp()
    df: pd.DataFrame = load_data(zip_path=zip_path)
    df = prepare_data(df)
    vectorizer: TfidfVectorizer = create_vectorizer()
    job_tfidf: Any = vectorize_jobs(vectorizer, df["text"])  # type: ignore
    return df, vectorizer, job_tfidf