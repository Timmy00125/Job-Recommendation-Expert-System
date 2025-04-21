# app.py
# Bare‑bones Flask interface for the Job Recommender Expert System

from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from experta import KnowledgeEngine, Fact, Field, DefFacts, Rule, MATCH

# ---------- Data & Model Setup ----------
# 1. Load and preprocess data
DF = pd.read_csv('data job posts.csv')
DF.dropna(subset=['Title','JobDescription','JobRequirment','RequiredQual'], inplace=True)
DF['Location'] = DF['Location'].fillna('').astype(str)
DF['IT'] = DF['IT'].fillna(0).astype(int)

nltk.download('stopwords')
from nltk.corpus import stopwords

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join([w for w in text.split() if w not in stopwords.words('english')])

DF['text'] = (DF['Title'] + ' ' + DF['JobDescription'] + ' ' + DF['JobRequirment'] + ' ' + DF['RequiredQual']).apply(clean_text)

# 2. TF-IDF
VECT = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
JOB_TFIDF = VECT.fit_transform(DF['text'].values.astype('U'))

# 3. Cosine

def compute_cosine_scores(query):
    vec = VECT.transform([clean_text(query)])
    return cosine_similarity(vec, JOB_TFIDF).flatten()

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
        self.declare(Fact(rule_score=1))

    @Rule(UserPrefs(it_preference=1), JobFact(it_flag=1))
    def it_match(self):
        self.declare(Fact(rule_score=1))

# 5. Scoring

def compute_rule_score(row, prefs):
    eng = JobExpert()
    eng.reset()
    eng.declare(UserPrefs(location=str(prefs.get('location','')), it_preference=int(prefs.get('it_preference',0))))
    eng.declare(JobFact(id=int(row.name), location=row['Location'], it_flag=row['IT']))
    eng.run()
    return sum(1 for f in eng.facts.values() if isinstance(f, Fact) and f.get('rule_score', False))

# 6. Recommendation

def recommend_jobs(query, prefs, alpha=0.7, beta=0.3, top_n=10):
    cos = compute_cosine_scores(query)
    rules = np.array([compute_rule_score(r, prefs) for _, r in DF.iterrows()], dtype=float)
    final = alpha*cos + beta*rules
    # Prioritize location
    idx = np.argsort(final)[::-1]
    loc = prefs.get('location','').lower()
    matched = [i for i in idx if loc and loc in DF.iloc[i]['Location'].lower()]
    others = [i for i in idx if i not in matched]
    ordered = (matched + others)[:top_n]
    result = DF.iloc[ordered][['Title','Company','Location']].copy()
    result['score'] = final[ordered]
    return result.to_dict(orient='records')

# ---------- Flask App ----------
app = Flask(__name__)

INDEX_HTML = '''
<!doctype html>
<title>Job Recommender</title>
<h1>Job Recommender Expert System</h1>
<form method=post>
  <label>Location:</label>
  <select name=location>
    <option value="">--any--</option>
    {% for loc in locations %}
    <option value="{{loc}}">{{loc}}</option>
    {% endfor %}
  </select><br><br>
  <label>Prefer IT?</label>
  <select name=it_preference>
    <option value=0>No</option>
    <option value=1>Yes</option>
  </select><br><br>
  <label>Keywords:</label>
  <input type=text name=user_text placeholder="e.g. python data analysis"><br><br>
  <input type=submit value=Recommend>
</form>
'''

RESULT_HTML = '''
<!doctype html>
<title>Recommendations</title>
<h1>Top {{results|length}} Matches</h1>
<table border=1>
  <tr><th>Title</th><th>Company</th><th>Location</th><th>Score</th></tr>
  {% for r in results %}
  <tr>
    <td>{{r.Title}}</td><td>{{r.Company}}</td><td>{{r.Location}}</td><td>{{"%.4f"|format(r.score)}}</td>
  </tr>
  {% endfor %}
</table>
<a href="/">Back</a>
'''

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        prefs = {
            'location': request.form.get('location',''),
            'it_preference': int(request.form.get('it_preference',0))
        }
        query = request.form.get('user_text','')
        results = recommend_jobs(query, prefs, top_n=10)
        return render_template_string(RESULT_HTML, results=results)
    else:
        # Populate dropdown
        locs = sorted([l for l in DF['Location'].unique() if l])
        return render_template_string(INDEX_HTML, locations=locs)

if __name__ == '__main__':
    app.run(debug=True)
