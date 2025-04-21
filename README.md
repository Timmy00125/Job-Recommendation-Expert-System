# Job Recommender Expert System

A sophisticated job recommendation application that combines TF-IDF text analysis with rule-based expert systems to find suitable job matches based on user preferences.

## Overview

This application helps job seekers find relevant employment opportunities by analyzing job descriptions and matching them against user-specified keywords and preferences. The system uses both text similarity (TF-IDF and cosine similarity) and rule-based matching to provide personalized recommendations.

## Features

- **Keyword-Based Search**: Find jobs matching specific skills, technologies, or requirements
- **Location Filtering**: Prioritize jobs in preferred locations
- **IT Specialization**: Option to focus on IT-related positions
- **Hybrid Recommendation Algorithm**: Combines text analysis with expert system rules
- **Modern Web Interface**: Clean, responsive design with intuitive controls
- **Visual Score Representation**: See match quality through visual indicators

## Installation

### Prerequisites

- Python 3.8 NOTE MUST BE THIS EXACT VERSION
- pip package manager
- Pandas
- NumPy
- NLTK
- scikit-learn
- Flask
- Experta

### Setup

1. Clone this repository:

   ```
   git clone https://github.com/imosudi/Job-Recommendation-Expert-System.git
   cd job-recommender
   ```

2. Install required packages:

   ```
    pip install -r requirements.txt
   ```

3. Make sure you have the job data CSV file:
   ```
   data job posts.csv
   ```

## Project Structure

```
/job-recommender
    /static
        /css
            style.css        # UI styling
        /js
            scripts.js       # Frontend functionality
    /templates
        index.html          # Search form template
        results.html        # Results display template
    app.py                  # Flask application
    recommendation_engine.py # AI/recommendation logic
    data job posts.csv      # Job listings dataset
    README.md               # This file
```

## How It Works

### Recommendation Engine

The recommendation system works in several steps:

1. **Text Processing**: Job descriptions are cleaned and processed using NLTK
2. **Text Vectorization**: TF-IDF vectorization converts text to numerical features
3. **Rule System**: Expert system applies rules based on location and IT preferences
4. **Scoring**: Combines text similarity scores with rule-based scores
5. **Ranking**: Orders results by relevance and applies location priority

### Web Application

The Flask web application provides an intuitive interface for users to:

- Enter keywords related to desired jobs
- Select location preferences
- Indicate IT job preference
- View ranked recommendations with match scores

## Usage

1. Run the application:

   ```
   python app.py
   ```

2. Open your browser and navigate to:

   ```
   http://127.0.0.1:5000/
   ```

3. Enter your search criteria:

   - Type keywords in the text field
   - Select a location preference (if any)
   - Choose whether you prefer IT jobs
   - Click "Recommend Jobs"

4. Review your personalized job recommendations, sorted by match quality

## API

The system also provides a simple API endpoint for programmatic access:

```
POST /api/recommend
{
    "location": "New York",
    "it_preference": 1,
    "user_text": "python data analysis"
}
```

## Dependencies

- Flask: Web framework
- Pandas: Data manipulation
- NumPy: Numerical operations
- NLTK: Natural language processing
- scikit-learn: Machine learning tools including TF-IDF
- Experta: Expert system rules engine
- Font Awesome: UI icons

## Future Improvements

- User accounts to save preferences
- Job filtering by additional criteria (salary, company type, etc.)
- Resume parsing for automated job matching
- Sentiment analysis of job descriptions
- Mobile application version

## License

We will add license latter

## Author

Created by Timmy, Ernest, Abraham - April 2025

---
