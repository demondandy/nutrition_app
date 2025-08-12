from flask import Flask, render_template, redirect, url_for, request, flash, send_file
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

import random
import csv
from fpdf import FPDF
from io import BytesIO

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

# Nutrition Tips with Images
nutrition_tips = [
    {
        "text": "Stay hydrated! Drinking water can boost metabolism and improve focus.",
        "image": "https://images.unsplash.com/photo-1562512045-32f8d3dba6d5?auto=format&fit=crop&w=800&q=80"
    },
    {
        "text": "Add more fruits and vegetables to your meals for extra vitamins and minerals.",
        "image": "https://images.unsplash.com/photo-1506806732259-39c2d0268443?auto=format&fit=crop&w=800&q=80"
    },
    {
        "text": "Avoid skipping breakfast—it fuels your energy for the day.",
        "image": "https://images.unsplash.com/photo-1504674900247-0877df9cc836?auto=format&fit=crop&w=800&q=80"
    },
    {
        "text": "Limit sugary drinks and choose water or herbal tea instead.",
        "image": "https://images.unsplash.com/photo-1580910051074-7db7d9a06c7b?auto=format&fit=crop&w=800&q=80"
    },
    {
        "text": "Eat slowly and mindfully to help control portion sizes.",
        "image": "https://images.unsplash.com/photo-1600271881276-05051a6314ed?auto=format&fit=crop&w=800&q=80"
    },
    {
        "text": "Include lean protein sources like fish, chicken, beans, and nuts.",
        "image": "https://images.unsplash.com/photo-1613145993482-3f2a3f2f8a7d?auto=format&fit=crop&w=800&q=80"
    },
    {
        "text": "Get enough fiber daily from whole grains, fruits, and veggies.",
        "image": "https://images.unsplash.com/photo-1565958011703-44e2aca4c73e?auto=format&fit=crop&w=800&q=80"
    }
]

# Nutrition Articles (Title, Description, Image, Link)
nutrition_articles = [
    {
        "title": "Top 10 Superfoods for Energy",
        "desc": "Boost your daily energy with these nutrient-rich foods that support brain and body health.",
        "image": "https://images.unsplash.com/photo-1572449043416-55f4685c9bbf?auto=format&fit=crop&w=800&q=80",
        "link": "#"
    },
    {
        "title": "The Science of Hydration",
        "desc": "Learn why staying hydrated is key for metabolism, mood, and cognitive function.",
        "image": "https://images.unsplash.com/photo-1502741338009-cac2772e18bc?auto=format&fit=crop&w=800&q=80",
        "link": "#"
    },
    {
        "title": "Healthy Nigerian Dishes You Should Try",
        "desc": "Discover nutrient-packed traditional Nigerian meals and their health benefits.",
        "image": "https://images.unsplash.com/photo-1600891964599-f61ba0e24092?auto=format&fit=crop&w=800&q=80",
        "link": "#"
    },
    {
        "title": "Understanding Macronutrients",
        "desc": "A beginner-friendly guide to proteins, carbs, and fats, and how they fuel your body.",
        "image": "https://images.unsplash.com/photo-1604908812316-5f8b1d7e2e8d?auto=format&fit=crop&w=800&q=80",
        "link": "#"
    }
]


# ---------------------------
# Routes
# ---------------------------

@app.route('/')
def index():
    import datetime
    today_index = datetime.datetime.now().day % len(nutrition_tips)
    tip_of_the_day = nutrition_tips[today_index]

    return render_template(
        'index.html',
        tip_of_the_day=tip_of_the_day,
        nutrition_articles=nutrition_articles
    )



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
        file = request.files.get('image')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # Preprocess image for prediction
            img = Image.open(save_path).resize((150, 150))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = cnn_model.predict(img_array)
            label_index = np.argmax(prediction[0])
            predicted_label = class_labels[label_index]

            # Lookup nutritional info from Nigeria food CSV
            food_info = nigeria_food_df[
                nigeria_food_df['Food_Name'].str.lower() == predicted_label.lower()
            ]
            if not food_info.empty:
                row = food_info.iloc[0]
                calories = row['Calories_kcal']
                protein = row['Protein_g']
                carbs = row['Carbs_g']
                fat = row['Fat_g']
                portion = row['Portion_Size_g']
            else:
                calories = protein = carbs = fat = portion = 0

            # Calculate user daily calorie needs
            user_daily_calories = calculate_daily_calories(current_user)

            # Generate advice
            if calories == 0:
                advice = "No nutrition data available for this food."
            elif calories > user_daily_calories * 0.5:
                advice = "⚠️ This meal is high in calories compared to your daily needs. Consider a lighter option."
            else:
                advice = "✅ This meal fits well within your daily calorie intake."

            # Save recommendation history
            record = RecommendationHistory(
                user_id=current_user.id,
                source='image',
                result=f"Food: {predicted_label} | Calories: {calories} kcal | Advice: {advice}"
            )
            db.session.add(record)
            db.session.commit()

            # Pass everything to template
            return render_template(
                "upload.html",
                prediction=predicted_label,
                calories=calories,
                protein=protein,
                carbs=carbs,
                fat=fat,
                portion=portion,
                advice=advice,
                image_path=url_for('static', filename=f'uploads/{filename}')
            )
        else:
            flash("Invalid file type. Please upload a JPG, JPEG, or PNG image.", "danger")

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
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly_plan = [get_meal_plan("F1", "RF", day) for day in days]

    # Store in DB for history tracking
    for plan in weekly_plan:
        if plan:
            rec = Recommendation(
                user_id=current_user.id,
                recommended_plan=plan["diet_plan"],
                explanation_text=plan["explanation"]
            )
            db.session.add(rec)
    db.session.commit()

    return render_template("meal_plan.html", weekly_plan=weekly_plan)


@app.route('/meal_plan/download/<file_type>')
@login_required
def download_meal_plan(file_type):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly_plan = [get_meal_plan("F1", "RF", day) for day in days]

    if file_type == "csv":
        filepath = "weekly_meal_plan.csv"
        with open(filepath, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Day", "Calories", "Protein (g)", "Carbs (g)", "Fat (g)", "Diet Plan"])
            for plan in weekly_plan:
                if plan:
                    writer.writerow([plan["day"], plan["calories"], plan["protein_g"], plan["carbs_g"], plan["fat_g"], plan["diet_plan"]])
        return send_file(filepath, as_attachment=True)

    elif file_type == "pdf":
        filepath = "weekly_meal_plan.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, "Weekly Meal Plan", ln=True, align='C')
        pdf.ln(5)

        pdf.set_font("Arial", size=10)
        for plan in weekly_plan:
            if plan:
                pdf.multi_cell(0, 8, f"{plan['day']}: {plan['diet_plan']} "
                                      f"({plan['calories']} kcal, P:{plan['protein_g']}g, C:{plan['carbs_g']}g, F:{plan['fat_g']}g)")
                pdf.ln(1)
        pdf.output(filepath)
        return send_file(filepath, as_attachment=True)

    else:
        flash("Invalid file type requested.", "danger")
        return redirect(url_for("meal_plan"))



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
