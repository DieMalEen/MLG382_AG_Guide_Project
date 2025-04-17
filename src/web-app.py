import dash
from dash import html, dcc, Input, Output
import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn as nn
import os

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size=12):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(32, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x

# Setup Dash app
app = dash.Dash(__name__, external_stylesheets=["https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"])
server = app.server

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load all models and scalers once
def load_all_models():
    models = {}
    
    # Logistic Regression
    with open("../artifacts/regression_model.pkl", "rb") as f:
        models["regression"] = {"model": pickle.load(f)}
    with open("../artifacts/regression_scaler.pkl", "rb") as f:
        models["regression"]["scaler"] = pickle.load(f)

    # Random Forest
    with open("../artifacts/random_forest_model.pkl", "rb") as f:
        models["random_forest"] = {"model": pickle.load(f)}

    # XGBoost
    with open("../artifacts/xgboost_model.pkl", "rb") as f:
        models["xgboost"] = {"model": pickle.load(f)}

    # Deep Learning
    dl_model = SimpleNN()
    dl_model.load_state_dict(torch.load("../artifacts/deep_learning_model.pth", map_location=device))
    dl_model.to(device)
    dl_model.eval()
    with open("../artifacts/regression_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    models["deep_learning"] = {"model": dl_model, "scaler": scaler}

    return models

models = load_all_models()

columns = ["Age", "Gender", "Ethnicity", "ParentalEducation", "StudyTimeWeekly", "Absences", 
           "Tutoring", "ParentalSupport", "Extracurricular", "Sports", "Music", "Volunteering"]

categorical_options = {
    "Gender": [{"label": "Male", "value": 0}, {"label": "Female", "value": 1}],
    "Ethnicity": [{"label": "Group A", "value": 0}, {"label": "Group B", "value": 1}, {"label": "Group C", "value": 2}],
    "ParentalEducation": [{"label": "High School", "value": 0}, {"label": "Diploma", "value": 1}, {"label": "Degree", "value": 2}],
    "Tutoring": [{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
    "ParentalSupport": [{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
    "Extracurricular": [{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
    "Sports": [{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
    "Music": [{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
    "Volunteering": [{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
}

# App layout
app.layout = html.Div(className="min-h-screen bg-gray-100 flex justify-center items-center p-6", children=[
    html.Div(className="bg-white rounded-xl shadow-lg p-8 max-w-2xl w-full", children=[
        html.H1("BrightPath Academy Grade Prediction", className="text-3xl font-bold text-center text-blue-800 mb-6"),

        html.Div(className="mb-6", children=[
            html.Label("Select Model", className="block text-gray-700 font-medium mb-2"),
            dcc.Dropdown(
                id='model-selector',
                options=[
                    {'label': 'Logistic Regression', 'value': 'regression'},
                    {'label': 'Random Forest', 'value': 'random_forest'},
                    {'label': 'XGBoost', 'value': 'xgboost'},
                    {'label': 'Deep Learning', 'value': 'deep_learning'}
                ],
                value='regression',
                className="border border-gray-300 rounded-lg p-2 w-full focus:outline-none focus:ring-2 focus:ring-blue-500"
            ),
        ]),

        html.Div(className="mb-6", children=[
            html.Label("Student Details", className="block text-gray-700 font-medium mb-4"),
            html.Div(className="grid grid-cols-1 sm:grid-cols-2 gap-4", children=[
                html.Div([
                    html.Label(f"{col}", className="block text-gray-600 text-sm mb-1"),
                    dcc.Dropdown(
                        id=f'input-{col.lower()}',
                        options=categorical_options[col],
                        value=categorical_options[col][0]['value'],
                        className="border border-gray-300 rounded-lg p-2 w-full focus:outline-none focus:ring-2 focus:ring-blue-500"
                    ) if col in categorical_options else
                    dcc.Input(
                        id=f'input-{col.lower()}',
                        type='number',
                        value=0,
                        className="border border-gray-300 rounded-lg p-2 w-full focus:outline-none focus:ring-2 focus:ring-blue-500"
                    )
                ]) for col in columns
            ])
        ]),

        html.Button('Predict Grade', id='predict-button', n_clicks=0, 
                    className="w-full bg-blue-600 text-white font-semibold py-2 rounded-lg hover:bg-blue-700 transition duration-200"),

        html.Div(id='prediction-output', className="mt-6 text-center text-xl font-medium text-blue-800")
    ])
])

# Prediction callback
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    Input('model-selector', 'value'),
    *[Input(f'input-{col.lower()}', 'value') for col in columns]
)
def predict(n_clicks, model_type, *input_values):
    if n_clicks == 0:
        return "Enter student details and click Predict Grade."

    try:
        input_data = pd.DataFrame([dict(zip(columns, input_values))])
        model_info = models[model_type]
        model = model_info['model']
        scaler = model_info.get('scaler', None)

        if scaler:
            input_data = scaler.transform(input_data)

        if model_type == "deep_learning":
            input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
        else:
            predicted_class = model.predict(input_data)[0]

        grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
        return f"Predicted Grade: {grade_map.get(predicted_class, 'Unknown')}"

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
