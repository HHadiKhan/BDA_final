FROM python:3.9.12

WORKDIR /app2

COPY requirements.txt /app2/
RUN pip install --no-cache-dir -r requirements.txt
COPY diabetes_prediction_dataset.csv /app2/diabetes_prediction_dataset.csv
COPY LGBMBoost.pkl /app2/
COPY my_script.py /app/
COPY my_webapp.py /app2/

EXPOSE 8501

CMD [ "streamlit", "run", "--server-port","8051","my_webapp.py" ]