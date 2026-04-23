FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8000

# Make startup script executable
RUN chmod +x start.sh

# Run both Streamlit and FastAPI
CMD ["bash", "start.sh"]