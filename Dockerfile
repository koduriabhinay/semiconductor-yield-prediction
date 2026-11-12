FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/

# Bake the dataset + trained artifacts into the image.
RUN python scripts/download_data.py && python scripts/train.py

EXPOSE 8000
ENV PYTHONPATH=/app/src
CMD ["uvicorn", "secom_yield.api:app", "--host", "0.0.0.0", "--port", "8000"]
