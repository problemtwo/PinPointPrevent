FROM python:3.6.5
ADD .
WORKDIR .
RUN pip install -r requirements.txt
CMD python3 main.py Abhi 10

