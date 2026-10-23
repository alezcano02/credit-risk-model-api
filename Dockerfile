FROM python:3.12-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY creditapi/ creditapi/
COPY model/ model/

EXPOSE 8000
CMD ["uvicorn", "creditapi.app:app", "--host", "0.0.0.0", "--port", "8000"]
