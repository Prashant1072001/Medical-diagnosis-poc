""" CREATING SEPARATE TABLES FOR EACH MODEL TO STORE THE INPUTS (HEART AND MALARIA ONLY FOR NOW) """

from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import os
from datetime import datetime, timezone

app = Flask(__name__)
app.secret_key = "pk_db_107"

# Configure PostgreSQL
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:1234@localhost/medical_poc_db'
app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db/{os.getenv('POSTGRES_DB')}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'

db = SQLAlchemy(app)

# ------------------ Models ------------------ #
# Table: more_users
class User(db.Model):
    __tablename__ = 'more_users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Table: heart_input
class HeartInput(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('more_users.id'), nullable=False)
    input_data = db.Column(db.String, nullable=False)
    prediction = db.Column(db.String, nullable=False)
    # timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

# Table: breast_cancer_input
class BreastCancerInput(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('more_users.id'), nullable=False)
    input_data = db.Column(db.String, nullable=False)
    prediction = db.Column(db.String, nullable=False)
    # timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

# Table: diabetes_input
class DiabetesInput(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('more_users.id'), nullable=False)
    input_data = db.Column(db.String, nullable=False)
    prediction = db.Column(db.String, nullable=False)
    # timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

# Table: kidney_input
class KidneyInput(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('more_users.id'), nullable=False)
    input_data = db.Column(db.String, nullable=False)
    prediction = db.Column(db.String, nullable=False)
    # timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

# Table: liver_input
class LiverInput(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('more_users.id'), nullable=False)
    input_data = db.Column(db.String, nullable=False)
    prediction = db.Column(db.String, nullable=False)
    # timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

# Table: malaria_input
class MalariaInput(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('more_users.id'), nullable=False)
    image_path = db.Column(db.String, nullable=False)
    prediction = db.Column(db.String, nullable=False)
    # timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


# ------------------ Routes ------------------ #
@app.route('/')
@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password)

        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()
        if existing_user:
            flash("Username or Email already exists.", "danger")
            return redirect(url_for('register'))

        new_user = User(username=username, email=email, password=hashed_password)
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

@app.route('/home')
def home():
    return render_template("new_home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/models')
def models():
    return redirect(url_for('home') + '#moveToModels')

# ------------------ Loading Models ------------------ #
heart_model = joblib.load(open("models/heart_disease_model.pkl", "rb"))
diabetes_model = joblib.load(open("models/diabetes_model_log.pkl", "rb"))
kidney_model = joblib.load(open("models/kidney_disease_model.pkl", "rb"))
liver_model = joblib.load(open("models/liver_disease_model2.pkl", "rb"))
cancer_model = joblib.load(open("models/breast_cancer_model.pkl", "rb"))
malaria_model = tf.keras.models.load_model("models/malaria_cnn_model.h5")

# ------------------ Heart Disease Prediction ------------------ #
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

    # Saving to DB
    if "user_id" in session:
        heart_input = HeartInput(
            user_id=session["user_id"],
            input_data=str(features),
            prediction=str(prediction)
        )
        db.session.add(heart_input)
        db.session.commit()

    return render_template("new_heart_result.html", prediction=prediction)

# ------------------ Diabetes Prediction ------------------ #
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

    # Saving to DB
    if "user_id" in session:
        diabetes_input = DiabetesInput(
            user_id=session["user_id"],
            input_data=str(features),
            prediction=str(prediction)
        )
        db.session.add(diabetes_input)
        db.session.commit()

    return render_template("new_diabetes_result.html", prediction=prediction)

# ------------------ Breast Cancer Prediction ------------------ #
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

    # Saving to DB
    if "user_id" in session:
        breast_cancer_input = BreastCancerInput(
            user_id=session["user_id"],
            input_data=str(features),
            prediction=str(prediction)
        )
        db.session.add(breast_cancer_input)
        db.session.commit()

    return render_template("new_cancer_result.html", prediction=prediction)

# ------------------ Liver Disease Prediction ------------------ #
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

    # Saving to DB
    if "user_id" in session:
        liver_input = LiverInput(
            user_id=session["user_id"],
            input_data=str(features),
            prediction=str(prediction)
        )
        db.session.add(liver_input)
        db.session.commit()

    return render_template("new_liver_result.html", prediction=prediction)

# ------------------ Kidney Disease Prediction ------------------ #
@app.route('/kidney')
def kidney():
    return render_template("new_kidney.html")

@app.route('/predict_kidney', methods=["POST"])
def predict_kidney():
    features = [float(request.form[key]) if key != 'htn' else int(request.form[key]) for key in [
        'hemo', 'pcv', 'sc', 'sg', 'rc', 'al', 'htn', 'bgr', 'sod', 'bu'
    ]]
    prediction = kidney_model.predict([features])[0]

    # Saving to DB
    if "user_id" in session:
        kidney_input = KidneyInput(
            user_id=session["user_id"],
            input_data=str(features),
            prediction=str(prediction)
        )
        db.session.add(kidney_input)
        db.session.commit()

    return render_template("new_kidney_result.html", prediction=prediction)

# ------------------ Malaria Prediction ------------------ #
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

    # Saving image
    # filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{file.filename}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(image_path)

    # Processing and prediction
    img = Image.open(image_path).convert("RGB")
    img = img.resize((50, 50))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = int(np.argmax(malaria_model.predict(img_array), axis=1)[0])

    # Saving to DB
    if "user_id" in session:
        malaria_input = MalariaInput(
            user_id=session["user_id"],
            image_path=image_path,
            prediction=str(prediction)
        )
        db.session.add(malaria_input)
        db.session.commit()

    return render_template("new_malaria_result.html", prediction=prediction)

# ------------------ Main ------------------ #
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
