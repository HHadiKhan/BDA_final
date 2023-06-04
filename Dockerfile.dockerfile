FROM python:latest

COPY ./my_app /app
WORKDIR /app

COPY ./requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["python", "my_script.py"]
