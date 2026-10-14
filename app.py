
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(app.root_path, "employee_salary_model.pkl")
model = joblib.load(MODEL_PATH)

FEATURES = [
    "Age",
    "Gender",
    "Education Level",
    "Job Title",
    "Years of Experience"
]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()

        try:
            age = int(request.form["age"])
            gender = request.form["gender"].strip()
            education = request.form["education"].strip()
            job_title = request.form["job_title"].strip()
            experience = float(request.form["experience"])

            if not 18 <= age <= 100:
                raise ValueError("Age must be between 18 and 100.")

            if not gender or not education or not job_title:
                raise ValueError("Please complete all fields.")

            if not 0 <= experience <= 80:
                raise ValueError(
                    "Experience must be between 0 and 80 years."
                )

            employee = pd.DataFrame([{
                "Age": age,
                "Gender": gender,
                "Education Level": education,
                "Job Title": job_title,
                "Years of Experience": experience
            }], columns=FEATURES)

            result = float(model.predict(employee)[0])

            if not pd.notna(result):
                raise ValueError("The model returned an invalid prediction.")

            prediction = round(max(0, result), 2)

        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc) or "Please enter valid information."

        except Exception:
            app.logger.exception("Salary prediction failed")
            error = (
                "Prediction failed. Please check the trained model "
                "and input data."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True)