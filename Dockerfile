FROM python:3.11

RUN mkdir -p /generator
WORKDIR /generator

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY main.py .
COPY reader.py .
COPY gui.py .

EXPOSE 7860

CMD ["python", "gui.py"]