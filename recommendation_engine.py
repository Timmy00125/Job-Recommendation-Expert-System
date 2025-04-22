import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from experta import KnowledgeEngine, Fact, Field, DefFacts, Rule, MATCH

# ---------- Data & Model Setup ----------
# 1. Load and preprocess data
def load_data():
    df = pd.read_csv('data job posts.csv')
    df.dropna(subset=['Title','JobDescription','JobRequirment','RequiredQual'], inplace=True)
    df['Location'] = df['Location'].fillna('').astype(str)
    df['IT'] = df['IT'].fillna(0).astype(int)
    return df

def setup_nlp():
    nltk.download('stopwords', quiet=True)
    
def clean_text(text):
    from nltk.corpus import stopwords
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join([w for w in text.split() if w not in stopwords.words('english')])

def prepare_data(df):
    df['text'] = (df['Title'] + ' ' + 
                 df['JobDescription'] + ' ' + 
                 df['JobRequirment'] + ' ' + 
                 df['RequiredQual']).apply(clean_text)
    return df

# 2. TF-IDF
def create_vectorizer():
    return TfidfVectorizer(max_features=5000, ngram_range=(1,2))

def vectorize_jobs(vectorizer, job_texts):
    return vectorizer.fit_transform(job_texts.values.astype('U'))

# 3. Cosine Similarity
def compute_cosine_scores(query, vectorizer, job_tfidf):
    vec = vectorizer.transform([clean_text(query)])
    return cosine_similarity(vec, job_tfidf).flatten()

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
    def start(self):
        yield UserPrefs(location='', it_preference=0)

    @Rule(UserPrefs(location=MATCH.loc), JobFact(location=MATCH.loc))
    def loc_match(self, loc):
        if loc and loc.strip():  # Only match non-empty locations
            self.declare(Fact(rule_score=1))

    @Rule(UserPrefs(it_preference=1), JobFact(it_flag=1))
    def it_match(self):
        self.declare(Fact(rule_score=1))

# 5. Scoring
def compute_rule_score(row, prefs):
    eng = JobExpert()
    eng.reset()
    eng.declare(UserPrefs(location=str(prefs.get('location','')), 
                         it_preference=int(prefs.get('it_preference',0))))
    eng.declare(JobFact(id=int(row.name), location=row['Location'], it_flag=row['IT']))
    eng.run()
    return sum(1 for f in eng.facts.values() if isinstance(f, Fact) and f.get('rule_score', False))

# 6. Recommendation
def recommend_jobs(query, prefs, df, vectorizer, job_tfidf, alpha=0.7, beta=0.3, top_n=10):
    cos = compute_cosine_scores(query, vectorizer, job_tfidf)
    rules = np.array([compute_rule_score(r, prefs) for _, r in df.iterrows()], dtype=float)
    
    # Normalize rule scores if any are non-zero
    if rules.max() > 0:
        rules = rules / rules.max()
        
    final = alpha * cos + beta * rules
    
    # Prioritize location
    idx = np.argsort(final)[::-1]
    loc = prefs.get('location','').lower()
    matched = [i for i in idx if loc and loc in df.iloc[i]['Location'].lower()]
    others = [i for i in idx if i not in matched]
    ordered = (matched + others)[:top_n]
    
    result = df.iloc[ordered][['Title','Company','Location']].copy()
    result['score'] = final[ordered]
    return result.to_dict(orient='records')

# Initialize model and data
def initialize_recommendation_system():
    setup_nlp()
    df = load_data()
    df = prepare_data(df)
    vectorizer = create_vectorizer()
    job_tfidf = vectorize_jobs(vectorizer, df['text'])
    return df, vectorizer, job_tfidf