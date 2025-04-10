import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler (pre-fitted scaler)
with open("artifacts/regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the pre-fitted scaler
with open("artifacts/regression_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def regression_predict_grade(input_string):
    # Define the column names as per your data
    columns = ["Age", "Gender", "Ethnicity", "ParentalEducation", "StudyTimeWeekly", "Absences", 
               "Tutoring", "ParentalSupport", "Extracurricular", "Sports", "Music", "Volunteering"]

    # Split the input string by commas
    input_values = input_string.split(',')
    
    # Remove the StudentID and the GPA and GradeClass (assuming StudentID is first and GPA, GradeClass are last)
    input_values = input_values[1:-2]

    # Convert the input values into a dictionary with proper column names
    input_data = {columns[i]: float(input_values[i]) for i in range(len(input_values))}

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale the input data using the pre-fitted scaler
    input_scaled = scaler.transform(input_df)

    # Make the prediction using the trained model
    predicted_class = model.predict(input_scaled)[0]

    # Mapping numeric class back to the letter grade
    grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}

    predicted_grade = grade_map[predicted_class]
    return predicted_grade

# Example input string from user (ensure it has the correct format)
student_input = "1962,17,1,0,1,19.27635322376361,3,1,4,0,0,0,0,3.576909015459067,0.0"

# Print the predicted grade
print(f"Predicted grade: {regression_predict_grade(student_input)}")