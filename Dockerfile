FROM python:3.11

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py ./

COPY artifacts ./artifacts

CMD ["python", "api.py"] 