FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
ENV PYTHONPATH=/app/src ARTIFACT_DIR=/app/artifacts
# artifacts mounted at runtime: docker run -v ./artifacts:/app/artifacts
EXPOSE 8000
CMD ["uvicorn", "dfs.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
