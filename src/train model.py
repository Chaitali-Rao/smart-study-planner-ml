import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

#Load dataset
df = pd.read_csv("data/student_performance.csv")

#Encode categorical columns
df_encoded = pd.get_dummies(df, columns=["subject", "topic"], drop_first=True)

#Split features & target
X = df_encoded.drop("test_score", axis=1)
y = df_encoded["test_score"]

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

#Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#MODEL 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_predict = lr.predict(X_test_scaled)

lr_mae = mean_absolute_error(y_test, lr_predict)
lr_r2 = r2_score(y_test, lr_predict)

#MODEL 2: Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)   
rf_preds = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

# 6. Print results
print("Linear Regression:")
print("MAE:", lr_mae)
print("R2 Score:", lr_r2)

print("\nRandom Forest:")
print("MAE:", rf_mae)
print("R2 Score:", rf_r2)

print("Linear Regression score:", lr.score(X_test_scaled, y_test) * 100)
print("Random Forest score:", rf.score(X_test, y_test) * 100)

#Save model and scaler
joblib.dump(lr, "outputs/linear_regression_model.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")

print("\nModel and scaler saved successfully!")
