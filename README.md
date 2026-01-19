Smart Study Planner using Machine Learning

Problem Statement
Students often fail to identify weak topics and allocate study time efficiently.
This project predicts student test scores using Machine Learning and generates a personalized study plan by classifying topics as WEAK or STRONG.

Dataset
The dataset contains the following fields:
student_id
subject
topic
hours_studied
previous_score
topic_difficulty
test_score

Approach
Load and preprocess student performance data
Encode categorical features (subject, topic)
Split data into training and testing sets
Scale features for Linear Regression
Train and evaluate ML models
Predict scores for all student-topic pairs
Identify weak and strong topics
Generate recommended study hours

Models Used
Linear Regression
Random Forest Regressor

Project Structure
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


How to Run:
1.Install dependencies
pip install -r requirements.txt
2.Train the model
python src/train_model.py
This trains the models, evaluates performance, and saves the trained model and scaler.
3.Generate study plan
python src/predict.py
This loads the saved model, predicts scores, and generates study_plan.csv inside the outputs folder.

Output
The final output includes:
Predicted test score
Topic status (WEAK / STRONG)
Recommended study hours

Key Learnings
End-to-end Machine Learning workflow
Feature encoding and scaling
Model evaluation using MAE and R² score
Separation of training and prediction logic
Structuring a real-world ML project
