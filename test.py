
import joblib
import pandas as pd

model = joblib.load("employee_salary_model.pkl")

age = int(input("Age: "))
gender = input("Gender (Male/Female/Other): ").strip()
education = input(
    "Education Level (High School/Bachelor's/Master's/PhD): "
).strip()
job_title = input("Job Title: ").strip()
experience = float(input("Years of Experience: "))

if age < 18 or age > 100:
    raise ValueError("Age must be between 18 and 100.")

if experience < 0 or experience > 80:
    raise ValueError("Experience must be between 0 and 80 years.")

employee = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Education Level": education,
    "Job Title": job_title,
    "Years of Experience": experience
}])

prediction = float(model.predict(employee)[0])

print("\nEstimated Annual Salary: {:.2f}".format(max(0, prediction)))
print("Currency: USD, if your Salary column contains USD values.")