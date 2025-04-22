from flask import Flask, request, render_template, jsonify
from recommendation_engine import initialize_recommendation_system, recommend_jobs

app = Flask(__name__)

# Initialize recommendation system
DF, VECT, JOB_TFIDF = initialize_recommendation_system()

@app.route('/', methods=['GET'])
def index():
    # Populate dropdown with locations
    locs = sorted([l for l in DF['Location'].unique() if l])
    return render_template('index.html', locations=locs)

@app.route('/recommend', methods=['POST'])
def recommend():
    prefs = {
        'location': request.form.get('location', ''),
        'it_preference': int(request.form.get('it_preference', 0))
    }
    query = request.form.get('user_text', '')
    results = recommend_jobs(query, prefs, DF, VECT, JOB_TFIDF, top_n=10)
    return render_template('results.html', results=results)

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """API endpoint for AJAX requests"""
    data = request.get_json()
    prefs = {
        'location': data.get('location', ''),
        'it_preference': int(data.get('it_preference', 0))
    }
    query = data.get('user_text', '')
    results = recommend_jobs(query, prefs, DF, VECT, JOB_TFIDF, top_n=10)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)