import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn

# Grade Map
grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}

def load_data():
    train = pd.read_csv("data/train_data.csv", delimiter=",")
    test = pd.read_csv("data/test_data.csv", delimiter=",")
    
    return train, test

def prepare_data(train, test):  
    x_train = train.drop(columns=["StudentID", "GPA", "GradeClass"])
    y_train = train["GradeClass"] #Dependent variable for "training"

    x_test = test.drop(columns=["StudentID", "GPA", "GradeClass"])
    y_test = test["GradeClass"] #Dependent variable for testing accuracy

    return x_train, y_train, x_test, y_test


def run_logistic_regression(x_train, y_train, x_test, y_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    print("test")
    print(x_train_scaled)

    model = LogisticRegression(max_iter=100, solver='lbfgs', warm_start=True)     # Train model


    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print("Logistic Regression Accuracy:", round(accuracy, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['A', 'B', 'C', 'D', 'F']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    with open("artifacts/regression_model.pkl", "wb") as f:     # Save the model into a pkl file
        pickle.dump(model, f)

    with open('artifacts/regression_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return y_pred

def logistic_regression_graph(x_train, y_train, x_test, y_test):
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    # Train model
    model = LogisticRegression(solver='lbfgs', warm_start=True)

    accuracies = []

    for i in range(1, 16):
        model.max_iter = i
        model.fit(x_train_scaled, y_train)

        y_pred = model.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        accuracies.append(accuracy)

    plt.plot(range(1, 16), accuracies, marker='o', color='b')
    plt.title("Accuracy per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

def run_random_forest(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=50)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("\nRandom Forest Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, y_pred, target_names=list(grade_map.values())))

    with open("artifacts/random_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return y_pred

def run_xgboost(x_train, y_train, x_test, y_test):
    model = XGBClassifier(n_estimator=100, use_label_encoder=False, eval_metric='mlogloss', random_state=50)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("\nXGBoost Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report (XGBoost):")
    print(classification_report(y_test, y_pred, target_names=list(grade_map.values())))

    with open("artifacts/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return y_pred

def preprocess_for_deep_learning(x_train, y_train, x_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    return train_dataset, test_dataset, X_train_tensor.shape[1], scaler

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(32, 5)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.softmax(self.output(x))
    
def build_deep_learning_model(input_dim):
    return SimpleNN(input_dim)

def train_deep_learning_model(model, train_dataset, epochs=100, batch_size=32, learning_rate=0.001):
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

def evaluate_deep_learning_model(model, test_dataset):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=32)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    print("\nDeep Learning Accuracy:", round(accuracy_score(all_labels, all_preds), 4))
    print("\nClassification Report (Deep Learning):")
    print(classification_report(all_labels, all_preds, target_names=list(grade_map.values())))
    print("\nConfusion Matrix (Deep Learning):")
    print(confusion_matrix(all_labels, all_preds))

    return all_preds

# Save the Deep Learning model
def save_deep_learning_model(model):
    torch.save(model.state_dict(), "artifacts/deep_learning_model.pth")
    print("Deep Learning model saved as 'artifacts/deep_learning_model.pth'")


def save_predictions(test, y_pred, model_name): # Predict and save models test_data.csv predictions
    df = test[["StudentID", "GradeClass"]].copy()
    df[f"Predicted_{model_name}"] = y_pred
    df["ActualGrade"] = df["GradeClass"].astype(int).map(grade_map)
    df[f"PredictedGrade_{model_name}"] = pd.Series(y_pred).astype(int).map(grade_map)
    df["Match"] = df["ActualGrade"] == df[f"PredictedGrade_{model_name}"]
    
    df["Match"] = df["Match"].apply(lambda x: "True" if x else "False")     # Convert boolean values to 'True' and 'False' strings

    df = df.sort_values(by="Match").reset_index(drop=True)
    df.to_csv(f"artifacts/{model_name}_predictions.csv", index=False)

def main():
    train_data, test_data = load_data()
    x_train, y_train, x_test, y_test = prepare_data(train_data, test_data)

    y_pred_logreg = run_logistic_regression(x_train, y_train, x_test, y_test)
    save_predictions(test_data, y_pred_logreg, "regression")

    y_pred_rf = run_random_forest(x_train, y_train, x_test, y_test)
    save_predictions(test_data, y_pred_rf, "random_forest")

    y_pred_xgb = run_xgboost(x_train, y_train, x_test, y_test)
    save_predictions(test_data, y_pred_xgb, "xgboost")

    # Deep Learning workflow
    train_dataset, test_dataset, input_dim, _ = preprocess_for_deep_learning(x_train, y_train, x_test, y_test)
    model = build_deep_learning_model(input_dim)
    train_deep_learning_model(model, train_dataset)
    y_pred_dl = evaluate_deep_learning_model(model, test_dataset)
    save_deep_learning_model(model)
    save_predictions(test_data, y_pred_dl, "deep_learning")
    
if __name__ == "__main__":
    main()