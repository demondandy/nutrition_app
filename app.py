from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, login_required, current_user
from auth import auth
from models import db, User, FoodLog, Recommendation, RecommendationHistory

import os
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

from tensorflow.keras.models import load_model

# ---------------------------
# App configuration
# ---------------------------

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nutrition.db'
db.init_app(app)

# Configure upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.init_app(app)

app.register_blueprint(auth, url_prefix='/auth')


# ---------------------------
# Load user for Flask-Login
# ---------------------------

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ---------------------------
# Helper functions
# ---------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def calculate_daily_calories(user):
    # Provide safe fallback if data missing
    if not user.gender or not user.age or not user.height_cm or not user.weight_kg:
        return 2000  # default

    if user.gender == "male":
        bmr = 10 * user.weight_kg + 6.25 * user.height_cm - 5 * user.age + 5
    else:
        bmr = 10 * user.weight_kg + 6.25 * user.height_cm - 5 * user.age - 161

    if user.activity_level == "active":
        calories = bmr * 1.55
    elif user.activity_level == "moderate":
        calories = bmr * 1.3
    else:
        calories = bmr * 1.2

    return round(calories, 2)


# ---------------------------
# Load Model (Dummy fallback)
# ---------------------------

try:
    cnn_model = load_model('models/food_classifier_model.h5')
except:
    from tensorflow.keras import models, layers
    cnn_model = models.Sequential([
        layers.Flatten(input_shape=(150, 150, 3)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    print("Using dummy model for testing.")

# Dummy class labels if no real model
class_labels = [
    'Egusi Soup', 'Jollof Rice', 'Akamu', 'Okra Soup', 'Eba',
    'Moi Moi', 'Yam Porridge', 'Fried Plantain', 'Beans Porridge', 'Oha Soup'
]

# ---------------------------
# Load Nigeria food data
# ---------------------------

csv_path = 'data/nigeria_food.csv'
if os.path.exists(csv_path):
    nigeria_food_df = pd.read_csv(csv_path)
else:
    nigeria_food_df = pd.DataFrame([
        {'Food_Name': 'Egusi Soup', 'Calories_kcal': 500, 'Protein_g': 18, 'Carbs_g': 20, 'Fat_g': 40, 'Portion_Size_g': 250},
        {'Food_Name': 'Jollof Rice', 'Calories_kcal': 300, 'Protein_g': 6, 'Carbs_g': 50, 'Fat_g': 10, 'Portion_Size_g': 200},
        {'Food_Name': 'Akamu', 'Calories_kcal': 150, 'Protein_g': 3, 'Carbs_g': 32, 'Fat_g': 1, 'Portion_Size_g': 250},
        {'Food_Name': 'Okra Soup', 'Calories_kcal': 200, 'Protein_g': 10, 'Carbs_g': 15, 'Fat_g': 8, 'Portion_Size_g': 200},
        {'Food_Name': 'Eba', 'Calories_kcal': 350, 'Protein_g': 3, 'Carbs_g': 75, 'Fat_g': 1, 'Portion_Size_g': 250},
        {'Food_Name': 'Moi Moi', 'Calories_kcal': 250, 'Protein_g': 8, 'Carbs_g': 30, 'Fat_g': 12, 'Portion_Size_g': 200},
        {'Food_Name': 'Yam Porridge', 'Calories_kcal': 400, 'Protein_g': 6, 'Carbs_g': 60, 'Fat_g': 15, 'Portion_Size_g': 300},
        {'Food_Name': 'Fried Plantain', 'Calories_kcal': 300, 'Protein_g': 2, 'Carbs_g': 45, 'Fat_g': 15, 'Portion_Size_g': 150},
        {'Food_Name': 'Beans Porridge', 'Calories_kcal': 320, 'Protein_g': 12, 'Carbs_g': 50, 'Fat_g': 9, 'Portion_Size_g': 250},
        {'Food_Name': 'Oha Soup', 'Calories_kcal': 350, 'Protein_g': 15, 'Carbs_g': 18, 'Fat_g': 20, 'Portion_Size_g': 250},
    ])
    print("Using built-in dummy food data.")


# ---------------------------
# Routes
# ---------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.height_cm = float(request.form['height'])
        current_user.weight_kg = float(request.form['weight'])
        current_user.age = int(request.form['age'])
        current_user.gender = request.form['gender']
        current_user.activity_level = request.form['activity']
        db.session.commit()
        flash("Profile updated.", "success")
        return redirect(url_for('profile'))

    bmi = None
    if current_user.height_cm and current_user.weight_kg:
        height_m = current_user.height_cm / 100
        bmi = current_user.weight_kg / (height_m ** 2)

    return render_template('profile.html', user=current_user, bmi=bmi)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            img = Image.open(save_path).resize((150, 150))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = cnn_model.predict(img_array)
            label_index = np.argmax(prediction[0])
            predicted_label = class_labels[label_index]

            food_info = nigeria_food_df[nigeria_food_df['Food_Name'].str.lower() == predicted_label.lower()]
            if not food_info.empty:
                row = food_info.iloc[0]
                calories = row['Calories_kcal']
                protein = row['Protein_g']
                carbs = row['Carbs_g']
                fat = row['Fat_g']
                portion = row['Portion_Size_g']
            else:
                calories = protein = carbs = fat = portion = 0

            user_daily_calories = calculate_daily_calories(current_user)
            if calories > user_daily_calories * 0.5:
                advice = "This meal is high in calories compared to your daily need. Consider a lighter option."
            else:
                advice = "This meal fits within a healthy daily intake."

            record = RecommendationHistory(
                user_id=current_user.id,
                source='image',
                result=f"Image: {predicted_label} | Calories: {calories} kcal | Advice: {advice}"
            )
            db.session.add(record)
            db.session.commit()

            return render_template(
                "upload.html",
                prediction=predicted_label,
                calories=calories,
                protein=protein,
                carbs=carbs,
                fat=fat,
                portion=portion,
                advice=advice,
                image_path=os.path.join('uploads', filename)
            )
        else:
            flash("Invalid file type.", "danger")

    return render_template("upload.html")


@app.route('/history')
@login_required
def recommendation_history():
    history = RecommendationHistory.query.filter_by(user_id=current_user.id).order_by(RecommendationHistory.date.desc()).all()
    return render_template('history.html', history=history)


@app.route('/food_log', methods=['GET', 'POST'])
@login_required
def food_log():
    if request.method == 'POST':
        meal = request.form['meal']
        food_name = request.form['food_name']
        calories = float(request.form['calories'])
        new_log = FoodLog(
            user_id=current_user.id,
            meal=meal,
            food_name=food_name,
            calories=calories
        )
        db.session.add(new_log)
        db.session.commit()
        flash("Food log added successfully.")
        return redirect(url_for('food_log'))

    today = datetime.utcnow().date()
    logs = FoodLog.query.filter_by(user_id=current_user.id, date=today).all()
    total_calories = sum(log.calories for log in logs)

    return render_template('food_log.html', logs=logs, total_calories=total_calories)


@app.route('/meal_plan')
@login_required
def meal_plan():
    today = datetime.now().strftime("%a")[:3]
    plan = get_meal_plan("F1", "RF", today)
    if plan:
        rec = Recommendation(
            user_id=current_user.id,
            recommended_plan=plan["diet_plan"],
            explanation_text=plan["explanation"]
        )
        db.session.add(rec)
        db.session.commit()
    return render_template("meal_plan.html", plan=plan)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    age = int(request.form['age'])
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    activity = request.form['activity']
    recommendation = f"Suggested balanced diet for {age} years old with weight {weight}kg."

    new_record = RecommendationHistory(
        user_id=current_user.id,
        source='manual',
        result=recommendation
    )
    db.session.add(new_record)
    db.session.commit()

    return render_template('predict_result.html', result=recommendation)


# ---------------------------
# Meal Plan Helper
# ---------------------------

meal_df = pd.read_csv('data/weekly_meal_plans.csv') if os.path.exists('data/weekly_meal_plans.csv') else pd.DataFrame()

def get_meal_plan(family, model, day):
    plan_row = meal_df[
        (meal_df["Family"] == family) &
        (meal_df["Model"] == model) &
        (meal_df["Day"] == day)
    ]
    if plan_row.empty:
        return None

    row = plan_row.iloc[0]
    explanation = (
        f"Your plan for {day} provides {row['Calories']} kcal, "
        f"{row['Protein_g']}g protein, {row['Carbs_g']}g carbs, and "
        f"{row['Fat_g']}g fat. Meals: {row['DietPlan']}."
    )

    return {
        "day": day,
        "calories": row["Calories"],
        "protein_g": row["Protein_g"],
        "carbs_g": row["Carbs_g"],
        "fat_g": row["Fat_g"],
        "diet_plan": row["DietPlan"],
        "explanation": explanation
    }
