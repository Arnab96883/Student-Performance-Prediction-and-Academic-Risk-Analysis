# Student Performance Prediction and Academic Risk Analysis

**SEC C | Data Science & Machine Learning Project**

A Python dashboard that explores student academic data and demonstrates:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Linear Regression for predicted final marks
- Logistic Regression for pass/fail probability
- KNN for neighbour-based pass/fail prediction
- K-Means clustering for performance segments
- Cross-validation, GridSearchCV and permutation feature importance

## Important dataset note
`student_performance.csv` is a **synthetic illustrative dataset** generated for demonstrating the project workflow. It is not real student information. Replace it with authorized, anonymized data before making claims about actual students.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. (Recommended) create and activate a virtual environment:
   - Windows: `python -m venv .venv`
   - Windows activate: `.venv\\Scripts\\activate`
4. Install dependencies:
   `pip install -r requirements.txt`
5. Start the dashboard:
   `streamlit run app.py`

## CSV format
Required numeric columns:
`attendance, study_hours, previous_marks, assignment_marks, internal_marks, final_marks`

An optional `student_id` column is accepted. Attendance and marks should be on a 0–100 scale; study hours should be 0–24.

You can upload another CSV from the sidebar.

## Models and evaluation
- Regression: MAE, RMSE and R² on a held-out test set.
- Classification: test accuracy and 5-fold cross-validation accuracy.
- KNN: GridSearchCV tests k = 3, 5, 7, 9, 11.
- Clustering: K-Means with three clusters; cluster names are assigned by ordering cluster mean final marks.

The pass threshold is 40 marks. The dashboard's risk thresholds are transparent project heuristics and should be reviewed before real use.

## Suggested GitHub repository description
A Python and Streamlit data science project that predicts student exam performance, estimates pass/fail probability, identifies academic risk and segments student performance using machine learning.
