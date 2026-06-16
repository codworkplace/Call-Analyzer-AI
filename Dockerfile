FROM python:3.11-slim

# Установка ffmpeg (необходим для конвертации аудио)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render передаёт порт через переменную окружения PORT, но мы используем фиксированный 8000 внутри контейнера
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
