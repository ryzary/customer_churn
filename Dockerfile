FROM python:3.8.12-slim 

WORKDIR = /app
COPY ["requirements.txt", "./"]

RUN pip3 install -r requirements.txt
RUN mkdir models

COPY ["models/LR", "./models"]
COPY ["churn_app.py", "utils.py", "./"]


EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "churn_app:app"]