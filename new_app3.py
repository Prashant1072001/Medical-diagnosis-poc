## From chatgpt: Switch to PostgreSQL SQLALchemy (Chrome)

### ORIGINAL:
# from flask import Flask, render_template, request, redirect, url_for, session, flash
# from flask_sqlalchemy import SQLAlchemy
# from werkzeug.security import generate_password_hash, check_password_hash
# import numpy as np
# import joblib
# import tensorflow as tf
# from PIL import Image
# import cv2
# import os

# app = Flask(__name__)
# app.secret_key = "pk_db_107"

# # Configure PostgreSQL
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:1234@localhost/medical_poc_db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)

# # User model
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(80), unique=True, nullable=False)
#     password = db.Column(db.String(200), nullable=False)

# # Routes
# @app.route('/')
# @app.route('/register', methods=["GET", "POST"])
# def register():
#     if request.method == "POST":
#         username = request.form["username"]
#         password = request.form["password"]
#         hashed_password = generate_password_hash(password)

#         existing_user = User.query.filter_by(username=username).first()
#         if existing_user:
#             flash("Username already exists. Try another.", "danger")
#             return redirect(url_for('register'))

#         new_user = User(username=username, password=hashed_password)
#         db.session.add(new_user)
#         db.session.commit()
#         flash("Registration successful. Please log in.", "success")
#         return redirect(url_for('login'))

#     return render_template("register_login.html", action="Register", endpoint="register")

# @app.route('/login', methods=["GET", "POST"])
# def login():
#     if request.method == "POST":
#         username = request.form["username"]
#         password = request.form["password"]

#         user = User.query.filter_by(username=username).first()
#         if user and check_password_hash(user.password, password):
#             session["user_id"] = user.id
#             session["username"] = user.username
#             flash("Login successful!", "success")
#             return redirect(url_for("home"))
#         else:
#             flash("Invalid credentials", "danger")
#             return redirect(url_for("login"))

#     return render_template("register_login.html", action="Login", endpoint="login")

# @app.route('/logout')
# def logout():
#     session.clear()
#     flash("You have been logged out.", "info")
#     return redirect(url_for("login"))




### NEW APPROACH:
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import cv2
import os

app = Flask(__name__)
app.secret_key = "pk_db_107"  # Replace with a secure key in production

# Configure PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:1234@localhost/medical_poc_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# User model
class User(db.Model):
    __tablename__ = 'new_users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Routes
@app.route('/')
@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password)

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already exists. Try another.", "danger")
            return redirect(url_for('register'))

        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for('login'))

    return render_template("register_login.html", action="Register", endpoint="register")

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["username"] = user.username
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid credentials", "danger")
            return redirect(url_for("login"))

    return render_template("register_login.html", action="Login", endpoint="login")

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# Load models
heart_model = joblib.load(open("models/heart_disease_model.pkl", "rb"))
diabetes_model = joblib.load(open("models/diabetes_model_log.pkl", "rb"))
kidney_model = joblib.load(open("models/kidney_disease_model.pkl", "rb"))
liver_model = joblib.load(open("models/liver_disease_model2.pkl", "rb"))
cancer_model = joblib.load(open("models/breast_cancer_model.pkl", "rb"))
malaria_model = tf.keras.models.load_model("models/malaria_cnn_model.h5")

@app.route('/home')
def home():
    return render_template("new_home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/models')
def models():
    return redirect(url_for('home') + '#moveToModels')

# Prediction routes below remain unchanged (same as your original code)

# Add your existing heart, diabetes, cancer, liver, kidney, malaria routes here...
# Heart Disease
@app.route('/heart')
def heart():
    return render_template("new_heart.html")

@app.route('/predict_heart', methods=["POST"])
def predict_heart():
    features = [float(request.form[key]) for key in [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]]
    prediction = heart_model.predict([features])[0]
    return render_template("new_heart_result.html", prediction=prediction)

# Diabetes
@app.route('/diabetes')
def diabetes():
    return render_template("new_diabetes.html")

@app.route('/predict_diabetes', methods=["POST"])
def predict_diabetes():
    features = [float(request.form[key]) for key in [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]]
    prediction = diabetes_model.predict([features])[0]
    return render_template("new_diabetes_result.html", prediction=prediction)

# Breast Cancer
@app.route('/breast_cancer')
def breast_cancer():
    return render_template("new_cancer.html")

@app.route('/predict_breast_cancer', methods=["POST"])
def predict_cancer():
    features = [float(request.form[key]) for key in [
        'perimeter_worst', 'radius_worst', 'area_worst',
        'concave points_mean', 'concave points_worst',
        'concavity_worst', 'radius_mean', 'area_se',
        'area_mean', 'concavity_mean'
    ]]
    prediction = cancer_model.predict([features])[0]
    return render_template("new_cancer_result.html", prediction=prediction)

# Liver Disease
@app.route('/liver')
def liver():
    return render_template("new_liver.html")

@app.route('/predict_liver', methods=["POST"])
def predict_liver():
    features = [float(request.form[key]) for key in [
        'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
        'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
        'Aspartate_Aminotransferase', 'Total_Protiens',
        'Albumin', 'Albumin_and_Globulin_Ratio'
    ]]
    prediction = liver_model.predict([features])[0]
    return render_template("new_liver_result.html", prediction=prediction)

# Kidney Disease
@app.route('/kidney')
def kidney():
    return render_template("new_kidney.html")

@app.route('/predict_kidney', methods=["POST"])
def predict_kidney():
    features = [float(request.form[key]) if key != 'htn' else int(request.form[key]) for key in [
        'hemo', 'pcv', 'sc', 'sg', 'rc', 'al', 'htn', 'bgr', 'sod', 'bu'
    ]]
    prediction = kidney_model.predict([features])[0]
    return render_template("new_kidney_result.html", prediction=prediction)

# Malaria (Image Upload)
@app.route('/malaria')
def malaria():
    return render_template("new_malaria.html")

@app.route('/predict_malaria', methods=["POST"])
def predict_malaria():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    img = Image.open(file.stream).convert("RGB")
    img = img.resize((50, 50))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = np.argmax(malaria_model.predict(img_array), axis=1)[0]
    return render_template("new_malaria_result.html", prediction=prediction)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
    app.run(debug=True)
