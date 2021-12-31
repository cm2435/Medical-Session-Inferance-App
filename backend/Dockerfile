# backend/Dockerfile

FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
COPY ./models/. /models/

RUN apt-get -y update  && apt-get install -y \
   python3-dev \
   apt-utils \
   python-dev \
   build-essential \
   && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade setuptools  \
    && pip install \ 
    cython==0.29.24 \
    -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
