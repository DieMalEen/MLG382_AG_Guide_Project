import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
# Grade Map
grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}

def load_data():
    train = pd.read_csv("../data/train_data.csv", delimiter=",")
    test = pd.read_csv("../data/test_data.csv", delimiter=",")
    
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

    with open("../artifacts/regression_model.pkl", "wb") as f:     # Save the model into a pkl file
        pickle.dump(model, f)

    with open('../artifacts/regression_scaler.pkl', 'wb') as f:
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

    with open("../artifacts/random_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return y_pred

def run_xgboost(x_train, y_train, x_test, y_test):
    model = XGBClassifier(n_estimator=100, use_label_encoder=False, eval_metric='mlogloss', random_state=50)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("\nXGBoost Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report (XGBoost):")
    print(classification_report(y_test, y_pred, target_names=list(grade_map.values())))

    with open("../artifacts/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return y_pred

def preprocess_for_deep_learning(x_train, y_train, x_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    # One-hot encode the target (5 classes: A, B, C, D, F)
    y_train_encoded = to_categorical(y_train, num_classes=5)
    y_test_encoded = to_categorical(y_test, num_classes=5)

    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded

def build_deep_learning_model(input_shape):
    """Build and compile the Deep Learning model."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(5, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])    # Compile the model
    model.summary()    
    return model

def train_deep_learning_model(model, x_train, y_train):
    """Train the Deep Learning model with early stopping."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    return history

def evaluate_deep_learning_model(model, x_test, y_test):
    """Evaluate the Deep Learning model on the test set."""
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # evaluation metrics
    print("\nDeep Learning Accuracy:", round(accuracy_score(y_test_classes, y_pred_classes), 4))
    print("\nClassification Report (Deep Learning):")
    print(classification_report(y_test_classes, y_pred_classes, target_names=list(grade_map.values())))
    print("\nConfusion Matrix (Deep Learning):")
    print(confusion_matrix(y_test_classes, y_pred_classes))

    return y_pred_classes

# Save the Deep Learning model
def save_deep_learning_model(model):
    """Save the trained Deep Learning model as a .pkl file."""
    with open("../artifacts/deep_learning_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Deep Learning model saved as 'artifacts/deep_learning_model.pkl'")


def save_predictions(test, y_pred, model_name): # Predict and save models test_data.csv predictions
    df = test[["StudentID", "GradeClass"]].copy()
    df[f"Predicted_{model_name}"] = y_pred
    df["ActualGrade"] = df["GradeClass"].astype(int).map(grade_map)
    df[f"PredictedGrade_{model_name}"] = pd.Series(y_pred).astype(int).map(grade_map)
    df["Match"] = df["ActualGrade"] == df[f"PredictedGrade_{model_name}"]
    
    df["Match"] = df["Match"].apply(lambda x: "True" if x else "False")     # Convert boolean values to 'True' and 'False' strings

    df = df.sort_values(by="Match").reset_index(drop=True)
    df.to_csv(f"../artifacts/{model_name}_predictions.csv", index=False)

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
    x_train_dl, x_test_dl, y_train_dl, y_test_dl = preprocess_for_deep_learning(x_train, y_train, x_test, y_test)
    dl_model = build_deep_learning_model(x_train_dl.shape[1])
    train_deep_learning_model(dl_model, x_train_dl, y_train_dl)
    y_pred_dl = evaluate_deep_learning_model(dl_model, x_test_dl, y_test_dl)
    save_deep_learning_model(dl_model)
    save_predictions(test_data, y_pred_dl, "deep_learning")
    
if __name__ == "__main__":
    main()