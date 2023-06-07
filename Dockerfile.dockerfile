FROM python:3.9.12

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY diabetes_prediction_dataset.csv /app/diabetes_prediction_dataset.csv
COPY LGBMBoost.pkl /app/
COPY my_script.py /app/
COPY my_webapp.py /app/

EXPOSE 8503

CMD [ "streamlit", "run", "--server-port","8051","my_webapp.py" ]
