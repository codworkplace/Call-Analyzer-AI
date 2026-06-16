# Call Analyzer Backend

Бэкенд для анализа звонков: транскрипция аудио, диаризация, анализ по критериям через API vsegpt.ru.

## Локальный запуск

1. Создать виртуальное окружение: `python -m venv venv`
2. Активировать: `source venv/bin/activate` (Linux/Mac) или `venv\Scripts\activate` (Windows)
3. Установить зависимости: `pip install -r requirements.txt`
4. Установить ffmpeg (системный) или пропустить (будет использоваться из Docker)
5. Создать `.env` с `OPENAI_API_KEY=ваш_ключ_vsegpt`
6. Запустить: `uvicorn app:app --reload`

## Деплой на Render

- Подключить репозиторий к Render как Web Service.
- Выбрать Docker окружение.
- Добавить переменную окружения `OPENAI_API_KEY`.
- Деплой автоматический.
