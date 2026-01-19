# Smart Study Planner using Machine Learning

## Problem Statement
Students often fail to identify weak topics and allocate study time efficiently.
This project predicts student test scores using Machine Learning and generates a personalized study plan by classifying topics as WEAK or STRONG.

## Dataset
The dataset contains the following fields:
- student_id
- subject
- topic
- hours_studied
- previous_score
- topic_difficulty
- test_score

## Methodology
- Load and preprocess student performance data
- Encode categorical features
- Split data into training and testing sets
- Scale features for Linear Regression
- Train and evaluate the model
- Predict scores and classify topics

## Models Used
- Linear Regression (primary model)
- Random Forest Regressor (used for comparison)

Random Forest was evaluated but Linear Regression was selected for simplicity and interpretability.

## Project Structure
```markdown
Smart-Study-Planner-ML/
├── data/
│   └── student_performance.csv
├── src/
│   ├── train_model.py
│   └── predict.py
├── outputs/
│   └── study_plan.csv
├── requirements.txt
└── README.md
```

## How to Run
### Install dependencies
```bash
pip install -r requirements.txt
```
### Train the model
```bash
python src/train_model.py
```
This trains the models, evaluates performance, and saves the trained model and scaler.
### Generate study plan
```bash
python src/predict.py
```
This loads the saved model, predicts scores, and generates study_plan.csv inside the outputs folder.

## Output
The generated `study_plan.csv` contains:
- student_id
- subject
- topic
- predicted_score
- topic_status (WEAK / STRONG)
- recommended_study_hours