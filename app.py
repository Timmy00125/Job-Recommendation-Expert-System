from typing import Dict, Any, List
from flask import Flask, request, render_template, jsonify  # type: ignore
from recommendation_engine import initialize_recommendation_system, recommend_jobs

app = Flask(__name__)  # type: ignore

# Initialize recommendation system
DF, VECT, JOB_TFIDF = initialize_recommendation_system()


@app.route("/", methods=["GET"])  # type: ignore
def index() -> str:  # type: ignore
    # Populate dropdown with locations
    locs = sorted([loc for loc in DF["Location"].unique() if loc])  # type: ignore
    return render_template("index.html", locations=locs)  # type: ignore


@app.route("/recommend", methods=["POST"])  # type: ignore
def recommend() -> str:  # type: ignore
    prefs: Dict[str, Any] = {
        "location": request.form.get("location", ""),  # type: ignore
        "it_preference": int(request.form.get("it_preference", 0)),  # type: ignore
    }
    query: str = str(request.form.get("user_text", ""))  # type: ignore
    results: List[Dict[str, Any]] = recommend_jobs(
        query, prefs, DF, VECT, JOB_TFIDF, top_n=10
    )
    return render_template("results.html", results=results)  # type: ignore


@app.route("/api/recommend", methods=["POST"])  # type: ignore
def api_recommend() -> str:  # type: ignore
    """API endpoint for AJAX requests"""
    data = request.get_json()  # type: ignore
    prefs: Dict[str, Any] = {
        "location": data.get("location", ""),  # type: ignore
        "it_preference": int(data.get("it_preference", 0)),  # type: ignore
    }
    query: str = str(data.get("user_text", ""))  # type: ignore
    results: List[Dict[str, Any]] = recommend_jobs(
        query, prefs, DF, VECT, JOB_TFIDF, top_n=10
    )
    return jsonify(results)  # type: ignore


if __name__ == "__main__":
    app.run(debug=True)  # type: ignore
