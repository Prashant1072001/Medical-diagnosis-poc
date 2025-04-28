# Use the official Python image
FROM python:3.7-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy dependency file and install requirements
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Create the upload folder (to avoid runtime errors)
RUN mkdir -p static/uploads

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "final_app.py"]
