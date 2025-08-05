from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    email = db.Column(db.String(100), unique=True)
    password_hash = db.Column(db.String(200))
    height_cm = db.Column(db.Float)
    weight_kg = db.Column(db.Float)
    age = db.Column(db.Integer)
    activity_level = db.Column(db.String(50))
    gender = db.Column(db.String(10), nullable=True)
    role = db.Column(db.String(50), default="user")
    is_admin = db.Column(db.Boolean, default=False)


class FoodLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    date = db.Column(db.Date, default=datetime.utcnow)
    meal = db.Column(db.String(50))
    food_name = db.Column(db.String(100))
    calories = db.Column(db.Float)

class Recommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    date = db.Column(db.DateTime, default=datetime.utcnow)
    recommended_plan = db.Column(db.Text)
    explanation_text = db.Column(db.Text)

class RecommendationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    source = db.Column(db.String(50))  # 'manual', 'image'
    result = db.Column(db.Text)  # prediction text or diet summary

    user = db.relationship('User', backref=db.backref('recommendations', lazy=True))

