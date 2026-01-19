import pandas as pd
import joblib

#Load saved model & scaler 
lr = joblib.load("outputs/linear_regression_model.pkl")
scaler = joblib.load("outputs/scaler.pkl")

#Load dataset 
df = pd.read_csv("data/student_performance.csv")

#Encode categorical columns
df_encoded = pd.get_dummies(df, columns=["subject", "topic"], drop_first=True)

#Match training features
X = df_encoded.drop("test_score", axis=1)

#Scale features
X_scaled = scaler.transform(X)

#Predict scores
df["predicted_score"] = lr.predict(X_scaled)

#Weak / Strong topic detection
THRESHOLD = 60

df["topic_status"] = df["predicted_score"].apply(
    lambda x: "WEAK" if x < THRESHOLD else "STRONG"
)

#Study time allocation
def allocate_study_time(score):
    return round((100 - score) / 10, 1)

df["recommended_hours"] = df["predicted_score"].apply(allocate_study_time)

#Save study plan
study_plan = df[
    ["student_id", "subject", "topic",
     "predicted_score", "topic_status", "recommended_hours"]
]

study_plan.to_csv("outputs/study_plan.csv", index=False)

print("\nStudy plan generated successfully!")
print(study_plan)
