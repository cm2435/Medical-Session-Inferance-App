# frontend/Dockerfile

FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .


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

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]

