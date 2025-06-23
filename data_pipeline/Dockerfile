<<<<<<< HEAD
FROM python:3.12-alpine AS loader
=======
FROM python:3.12-alpine
>>>>>>> agentic_classification
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
<<<<<<< HEAD
CMD ["python", "-m", "src.main", "--update_db"]
=======
CMD ["python", "-m", "src.write_db"]
>>>>>>> agentic_classification
