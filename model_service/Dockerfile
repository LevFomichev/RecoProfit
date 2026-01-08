FROM python:3.10-slim

WORKDIR /app

# Устанавливаем системные зависимости для XGBoost
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 1. Сначала копируем и устанавливаем зависимости (для кэширования)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Затем копируем все исходные файлы
COPY . .

# 3. Проверяем наличие критически важных файлов
RUN echo "=== ПРОВЕРКА ФАЙЛОВ ПЕРЕД ЗАПУСКОМ ===" && \
    echo "Текущая директория: $(pwd)" && \
    echo "" && \
    echo "Все файлы:" && \
    ls -la && \
    echo "" && \
    echo "Критические .pkl файлы:" && \
    (ls -la *.pkl 2>/dev/null || echo "⚠️  Нет .pkl файлов!") && \
    echo "" && \
    echo "Python файлы:" && \
    ls -la *.py && \
    echo "" && \
    echo "Статистика:" && \
    echo "Всего файлов: $(find . -type f | wc -l)" && \
    echo ".pkl файлов: $(find . -name '*.pkl' | wc -l)" && \
    echo "Python файлов: $(find . -name '*.py' | wc -l)" && \
    echo "=== ПРОВЕРКА ЗАВЕРШЕНА ==="

# 4. Тестируем импорт основных библиотек
RUN echo "=== ТЕСТ ИМПОРТОВ ===" && \
    python -c "import pandas; print('✅ pandas:', pandas.__version__)" && \
    python -c "import numpy; print('✅ numpy:', numpy.__version__)" && \
    python -c "import xgboost; print('✅ xgboost:', xgboost.__version__)" && \
    python -c "import flask; print('✅ flask:', flask.__version__)" && \
    echo "=== ТЕСТ УСПЕШЕН ==="

EXPOSE 5000

CMD ["python", "app.py"]