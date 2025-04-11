import pickle
from pydantic import BaseModel
import pandas as pd
from preprocess_input import preprocess_input

with open('SocioDemoRFModel.pkl', 'rb') as f:
    model = pickle.load(f)


# class RequestFormat(BaseModel):

# Load column structure from training data
reference_df = pd.read_csv("dataset/Student-Spending-Habits_PreProcessed.csv")
reference_columns = reference_df.drop(columns=[
    "Living_Expenses", "Food_and_Dining_Expenses", 
    "Transportation_Expenses", "Leisure_and_Entertainment_Expenses", 
    "Academic_Expenses"
]).columns.tolist()

# Raw input
new_user_input = {
  "Age_Group": "21-23",
  "Sex": "Female",
  "Year_Level": "Senior",
  "In_relationship": "No",
  "Personality": "Introvert",
  "Home_Region": "Southern Luzon (Southern Tagalog & Bicol)",
  "Living_Situation": "Inside campus",
  "Dorm_Area": "UP Dorm",
  "Roommates": "I live with 2-3 roommates",
  "Degree_Program": "BS Computer Science",
  "In_Organization": "No",
  "Hours_of_Study_per_Week": "Less then 10 hours",
  "Monthly_Allowance": "7050",
  "Family_Monthly_Income": "P12,031 - P24,060",
  "Have_Scholarship": "No",
  "Have_Job": "No",
  "Meal_Preferences": "A combination of the above",
  "Frequency_of_Going_Home": "Always",
  "Have_Health_Concern": "No",
  "Preferred_Payment_Method": "Cash",
  "Living_Expenses": 1250,
  "Food_and_Dining_Expenses": 3000,
  "Transportation_Expenses": 2000,
  "Leisure_and_Entertainment_Expenses": 300,
  "Academic_Expenses": 500
}


# Preprocess
input_df = preprocess_input(new_user_input, reference_columns)

# Predict
predicted_expenses = model.predict(input_df)
print("Predicted Expenses:", predicted_expenses)
