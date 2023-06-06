FROM python:3.9.12

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

CMD [ "streamlit", "run", "my_webapp.py" ]
