import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns
# Grade Map
grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}

# Load the data from the Excel files
def load_data():
    train = pd.read_csv("data/train_data.csv", delimiter=",")
    test = pd.read_csv("data/test_data.csv", delimiter=",")
    
    return train, test

# Separate the features and targets
def prepare_data(train, test):  
    # Drop the StudentID, GPA, and GradeClass
    x_train = train.drop(columns=["StudentID", "GPA", "GradeClass"])
    y_train = train["GradeClass"] #Dependent variable for "training"

    # Test values
    x_test = test.drop(columns=["StudentID", "GPA", "GradeClass"])
    y_test = test["GradeClass"] #Dependent variable for testing accuracy

    return x_train, y_train, x_test, y_test


# Logistic Regression needs to be scaled and ouliers need to be treated
def run_logistic_regression(x_train, y_train, x_test, y_test):
    # RobustScaler to take into account outliers
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    print("test")
    print(x_train_scaled)

    # Train Logistic Regression model through 100 iterations
    model = LogisticRegression(max_iter=100, solver='lbfgs', warm_start=True)

    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print("Logistic Regression Accuracy:", round(accuracy, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['A', 'B', 'C', 'D', 'F']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the model into a pkl file
    with open("artifacts/regression_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save the scale into a pkl file
    with open('artifacts/regression_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return y_pred

def logistic_regression_graph(x_train, y_train, x_test, y_test):
    # RobustScaler to take into account outliers
    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    # Train Logistic Regression model through 100 iterations
    model = LogisticRegression(solver='lbfgs', warm_start=True)

    # Create accuracies list
    accuracies = []

    # Iterate through 15 iterations
    for i in range(1, 16):
        model.max_iter = i
        model.fit(x_train_scaled, y_train)

        y_pred = model.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Save accuracy to list
        accuracies.append(accuracy)

    # Plot the accuracies list to a graph to show increase in accuracy with more iterations
    plt.plot(range(1, 16), accuracies, marker='o', color='b')
    plt.title("Accuracy per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

# Random Forest does not need to be scaled or outlier treated
def run_random_forest(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("\nRandom Forest Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, y_pred, target_names=list(grade_map.values())))

    with open("artifacts/random_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return y_pred

# XGBoost does not need to be scaled or outlier treated
def run_xgboost(x_train, y_train, x_test, y_test):
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("\nXGBoost Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report (XGBoost):")
    print(classification_report(y_test, y_pred, target_names=list(grade_map.values())))

    with open("artifacts/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return y_pred

# Predict and save models test_data.csv predictions
def save_predictions(test, y_pred, model_name):
    df = test[["StudentID", "GradeClass"]].copy()
    df[f"Predicted_{model_name}"] = y_pred
    df["ActualGrade"] = df["GradeClass"].astype(int).map(grade_map)
    df[f"PredictedGrade_{model_name}"] = pd.Series(y_pred).astype(int).map(grade_map)
    df["Match"] = df["ActualGrade"] == df[f"PredictedGrade_{model_name}"]
    
    # Convert boolean values to 'True' and 'False' strings
    df["Match"] = df["Match"].apply(lambda x: "True" if x else "False")

    # Sort with 'False' rows first
    df = df.sort_values(by="Match").reset_index(drop=True)

    # Save predictions to excel file
    df.to_csv(f"artifacts/{model_name}_predictions.csv", index=False)

# Main
def main():
    train_data, test_data = load_data()
    x_train, y_train, x_test, y_test = prepare_data(train_data, test_data)

    y_pred_logreg = run_logistic_regression(x_train, y_train, x_test, y_test)
    save_predictions(test_data, y_pred_logreg, "regression")

    y_pred_rf = run_random_forest(x_train, y_train, x_test, y_test)
    save_predictions(test_data, y_pred_rf, "random_forest")

    y_pred_xgb = run_xgboost(x_train, y_train, x_test, y_test)
    save_predictions(test_data, y_pred_xgb, "xgboost")


if __name__ == "__main__":
    main()