FROM python:3.10

WORKDIR /code

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN python -c "import whisper; whisper.load_model('tiny')"

COPY ./app /code/app

CMD ["fastapi", "run", "app/main.py", "--port", "80", "--host", "0.0.0.0"]
