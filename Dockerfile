# Using the official Python image
FROM python:3.7-slim

# Setting environment variables
# Prevents Python from writing .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1    
# Forces Python to output logs in real-time (not buffered). Useful for debugging and Docker logs.
ENV PYTHONUNBUFFERED=1

# Setting working directory inside the container
WORKDIR /app

# Copying dependency file and install requirements
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Copying the rest of the application
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Creating the upload folder (to avoid runtime errors)
RUN mkdir -p static/uploads

# Exposing the port Flask runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "final_app.py"]
