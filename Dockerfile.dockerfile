FROM python:3.9.12

WORKDIR /app

COPY ./requirements.txt /app/

RUN pip install -r requirements.txt

COPY ./my_webapp.py /app/

CMD [ "streamlit", "run", "my_webapp.py" ]
