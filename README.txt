-------------------------------------------------------------------------------
1. TASK DESCRIPTION
An end-to-end NLP-based machine learning pipeline that automatically classifies
customer support tickets into 7 business categories and assigns a 4-level
response priority (Low / Medium / High / Critical).
 
Trained on 2,200 labelled tickets. Final model performance:
  - Category prediction accuracy : 97.95%  (Random Forest)
  - Priority prediction accuracy : 91.14%  (Logistic Regression)
 
A Streamlit web application allows users to enter a new ticket and instantly
receive a predicted category and priority.


-------------------------------------------------------------------------------
2. REQUIREMENTS
Python 3.9+
Install all dependencies:
    pip install -r requirements.txt
NLTK resources (download once before first run):
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); \
    nltk.download('stopwords'); nltk.download('wordnet'); \
    nltk.download('averaged_perceptron_tagger'); \
    nltk.download('averaged_perceptron_tagger_eng'); nltk.download('omw-1.4')"


-------------------------------------------------------------------------------
3. HOW TO RUN

STEP 1 — View analytics(optional):
    cd Task-1/src
    python analytics.py

    Prints category and priority distribution tables and opens
    3 interactive chart windows.

STEP 2 — Train models and save artefacts (run once):
    cd Task-1/src
    python train_models.py
 
    This will:
      - Load and preprocess the dataset
      - Train all 6 models (3 for category, 3 for priority)
      - Print evaluation metrics for all models
      - Save all model artefacts to Task-1/models/

STEP 3 — Run sample predictions:
    cd Task-1/src
    python sample_predictions.py
 
    Saves results to: Task-1/outputs/sample_predictions.csv
    This step is only for saving sample_predictions.csv file. You can skip because it has nothing to do with app.

STEP 4 — Launch Streamlit web app:
    cd Task-1
    streamlit run src/app.py
 
    Open http://localhost:8501 in your browser.
    Enter subject and ticket text, click Predict.

NOTES
    - Run train_models.py once first before running predict.py, sample_predictions.py, app.py
	- the .pkl files in models/ must exist.
    - All src/ scripts should be run from Task-1/src/ directory.
    - Streamlit app should be launched from Task-1/ as: streamlit run src/app.py
	- or from Task-1/src/ as: streamlit run app.py


-------------------------------------------------------------------------------
3. MODEL SUMMARY

Target     | Best Model          | Accuracy
-----------|---------------------|---------
Category   | Random Forest       | 97.95%
Priority   | Logistic Regression | 91.14%

Three models trained per target: Logistic Regression, Complement Naive Bayes, Random Forest.
Full evaluation results printed by train_models.py and documented in report/Task1_Report.pdf.

