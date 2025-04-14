import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import pickle


# This is the Logistic Regression Start
# -------------------------------------------------------------------------------------------

# Step 1: Load Train and Test CSVs
train_data = pd.read_csv("data/train_data.csv", delimiter=",")
test_data = pd.read_csv("data/test_data.csv", delimiter=",")

# Step 2: Separate Features and Target
X_train = train_data.drop(columns=["GPA", "GradeClass", "StudentID"])
y_train = train_data["GradeClass"]

X_test = test_data.drop(columns=["GPA" ,"GradeClass", "StudentID"])
y_test = test_data["GradeClass"]

# Step 3: Feature Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Step 4: Train Logistic Regression Model
model = LogisticRegression(solver='lbfgs', warm_start=True)

accuracies = []

# Iterate through 15 iterations displaying accuracy on graph
for i in range(1, 16):
    model.max_iter = i
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    accuracies.append(accuracy)

plt.plot(range(1, 16), accuracies, marker='o', color='b')
plt.title("Accuracy per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

model.fit(X_train_scaled, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test_scaled)

# Step 6: Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['A', 'B', 'C', 'D', 'F']))

#print("\nConfusion Matrix:")
#print(confusion_matrix(y_test, y_pred))

# Step 7: Save Model
with open("artifacts/regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open('artifacts/regression_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Step 8: Save Test Predictions
grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}

comparison_df = test_data[["StudentID", "GradeClass"]].copy()
comparison_df["Predicted"] = y_pred
comparison_df["ActualGrade"] = comparison_df["GradeClass"].astype(int).map(grade_map)
comparison_df["PredictedGrade"] = pd.Series(y_pred).astype(int).map(grade_map)
comparison_df["Match"] = comparison_df["ActualGrade"] == comparison_df["PredictedGrade"]
comparison_df = comparison_df.sort_values("Match").reset_index(drop=True)

# Save Prediction Table
comparison_df.to_csv("artifacts/regression_prediction.csv", index=False)

# This is the Logistic Regression End
# -------------------------------------------------------------------------------------------