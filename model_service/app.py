from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
from recommendation_utils import recommend_items_for_user

app = Flask(__name__)

# Загружаем все необходимые данные при старте
print("=" * 60)
print("ЗАГРУЗКА МОДЕЛИ И ДАННЫХ")
print("=" * 60)

try:
    # Проверяем наличие необходимых файлов
    required_files = [
        'xgb_model.pkl',
        'feature_columns.pkl', 
        'best_threshold.pkl',
        'users_data.pkl',
        'all_items_data.pkl'
    ]
    
    print('Проверка файлов в директории:', os.listdir('.'))
    print("-" * 40)
    
    # Загружаем модель и метаданные
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print('Модель загружена!')
    
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    print(f'Фичи загружены!')
    
    with open('best_threshold.pkl', 'rb') as f:
        best_threshold = pickle.load(f)
    print(f'Порог загружен!')
    
    # Загружаем данные пользователей
    users_data = pd.read_pickle('users_data.pkl')
    print(f'Данные пользователей загружены!')
    
    # Загружаем данные товаров
    all_items_data = pd.read_pickle('all_items_data.pkl')
    print(f'Данные товаров загружены!')
    
    print("=" * 60)
    print('ВСЕ ДАННЫЕ ЗАГРУЖЕНЫ УСПЕШНО!')
    print("=" * 60)
    
except FileNotFoundError as e:
    print(f'ФАЙЛ НЕ НАЙДЕН: {e}')
    print('Доступные файлы:')
    print(os.listdir('.'))
    raise
except Exception as e:
    print(f'ОШИБКА ПРИ ЗАГРУЗКЕ: {e}')
    import traceback
    traceback.print_exc()
    raise

@app.route('/')
def home():
    return jsonify({
        "message": "Recommendation API",
        "description": "API для рекомендации товаров пользователям",
        "endpoints": {
            "/recommend/<int:user_id>": "GET - рекомендации для пользователя",
            "/recommend": "POST - рекомендации с кастомными параметрами",
            "/user_info/<int:user_id>": "GET - информация о пользователе",
            "/health": "GET - проверка работоспособности"
        },
        "model_info": {
            "feature_count": len(feature_columns),
            "threshold": float(best_threshold),
            "users_in_data": int(users_data['visitorid'].nunique()),
            "items_in_data": int(all_items_data.shape[0])
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка работоспособности API"""
    return jsonify({
        "status": "healthy",
        "service": "recommendation-api",
        "timestamp": pd.Timestamp.now().isoformat()
    })

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend_for_user(user_id):
    """Рекомендации для конкретного пользователя"""
    try:
        # Получаем рекомендации
        recommendations = recommend_items_for_user(
            user_id=user_id,
            model=model,
            all_items_data=all_items_data,
            users_data=users_data,  # ← ИСПРАВЛЕНО!
            feature_columns=feature_columns,
            top_k=3,
            threshold=best_threshold
        )
        
        # Формируем ответ
        result = {
            "user_id": user_id,
            "recommendations": [
                {
                    "item_id": int(item_id),
                    "probability": float(probability),
                    "rank": i + 1
                }
                for i, (item_id, probability) in enumerate(recommendations)
            ],
            "count": len(recommendations),
            "threshold": float(best_threshold),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Дополнительная информация о пользователе
        if user_id in users_data['visitorid'].values:
            user_events = users_data[users_data['visitorid'] == user_id]
            last_event = user_events.iloc[-1]
            
            # Получаем купленные товары пользователя
            purchased_items = users_data[
                (users_data['visitorid'] == user_id) & 
                (users_data['target'] == 1)
            ]['itemid'].unique()
            
            result["user_info"] = {
                "status": "existing_user",
                "total_events": int(len(user_events)),
                "total_purchases": int(len(purchased_items)),
                "purchase_rate": float(last_event['purchase_rate']) if 'purchase_rate' in last_event else float(last_event.get('proper_purchase_rate', 0)),
                "unique_items": int(last_event['unique_items']) if 'unique_items' in last_event else int(last_event.get('proper_unique_items', 0)),
                "purchase_ratio": len(purchased_items) / len(user_events) if len(user_events) > 0 else 0
            }
            
            # Проверяем, не рекомендовали ли уже купленные товары
            recommended_items = [item for item, _ in recommendations]
            already_purchased = [item for item in recommended_items if item in purchased_items]
            if already_purchased:
                result["warnings"] = {
                    "already_purchased_items": already_purchased,
                    "message": "Некоторые рекомендованные товары уже были куплены пользователем"
                }
        else:
            result["user_info"] = {
                "status": "new_user",
                "message": "Используются популярные товары для нового пользователя"
            }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "user_id": user_id,
            "timestamp": pd.Timestamp.now().isoformat()
        }), 400

@app.route('/recommend', methods=['POST'])
def recommend_custom():
    """Рекомендации с настраиваемыми параметрами"""
    try:
        request_data = request.json
        
        if not request_data:
            return jsonify({"error": "Request body is empty"}), 400
        
        # Обязательные параметры
        user_id = request_data.get('user_id')
        if user_id is None:
            return jsonify({"error": "user_id is required"}), 400
        
        # Опциональные параметры
        top_k = request_data.get('top_k', 3)
        threshold = request_data.get('threshold', best_threshold)
        include_user_info = request_data.get('include_user_info', True)
        
        # Валидация параметров
        if not isinstance(top_k, int) or top_k <= 0 or top_k > 50:
            return jsonify({"error": "top_k must be integer between 1 and 50"}), 400
        
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            return jsonify({"error": "threshold must be float between 0 and 1"}), 400
        
        # Получаем рекомендации
        recommendations = recommend_items_for_user(
            user_id=user_id,
            model=model,
            all_items_data=all_items_data,
            users_data=users_data,  # ← ИСПРАВЛЕНО!
            feature_columns=feature_columns,
            top_k=top_k,
            threshold=threshold
        )
        
        # Формируем базовый ответ
        result = {
            "user_id": user_id,
            "recommendations": [
                {
                    "item_id": int(item_id),
                    "probability": float(probability),
                    "rank": i + 1
                }
                for i, (item_id, probability) in enumerate(recommendations)
            ],
            "parameters": {
                "top_k": top_k,
                "threshold": float(threshold),
                "actual_count": len(recommendations)
            },
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Добавляем информацию о пользователе, если запрошено
        if include_user_info and user_id in users_data['visitorid'].values:
            user_events = users_data[users_data['visitorid'] == user_id]
            purchases = user_events[user_events['target'] == 1]
            
            result["user_info"] = {
                "total_events": int(len(user_events)),
                "total_purchases": int(len(purchases)),
                "purchase_ratio": len(purchases) / len(user_events) if len(user_events) > 0 else 0,
                "last_event_date": str(user_events['timestamp'].max()) if 'timestamp' in user_events.columns else None
            }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/user_info/<int:user_id>', methods=['GET'])
def get_user_info(user_id):
    """Информация о пользователе"""
    try:
        if user_id not in users_data['visitorid'].values:
            return jsonify({
                "user_id": user_id,
                "status": "not_found",
                "message": "User not found in training data",
                "timestamp": pd.Timestamp.now().isoformat()
            })
        
        user_events = users_data[users_data['visitorid'] == user_id]
        last_event = user_events.iloc[-1]
        purchases = user_events[user_events['target'] == 1]
        
        return jsonify({
            "user_id": user_id,
            "status": "found",
            "total_events": int(len(user_events)),
            "total_purchases": int(len(purchases)),
            "purchase_rate": float(last_event['purchase_rate']) if 'purchase_rate' in last_event else float(last_event.get('proper_purchase_rate', 0)),
            "unique_items": int(last_event['unique_items']) if 'unique_items' in last_event else int(last_event.get('proper_unique_items', 0)),
            "first_event_date": str(user_events['timestamp'].min()) if 'timestamp' in user_events.columns else None,
            "last_event_date": str(user_events['timestamp'].max()) if 'timestamp' in user_events.columns else None,
            "purchased_items": purchases['itemid'].unique().tolist()[:20],  # первые 20 товаров
            "timestamp": pd.Timestamp.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)