import pandas as pd
import numpy as np
from typing import List, Tuple

def get_popular_items(all_items_data: pd.DataFrame, users_data: pd.DataFrame, top_k: int = 3) -> List[Tuple[int, float]]:
    """
    Возвращает популярные товары для новых пользователей
    
    Args:
        all_items_data: DataFrame с фичами товаров
        users_data: DataFrame с событиями пользователей
        top_k: количество рекомендаций
    
    Returns:
        Список кортежей (item_id, probability)
    """
    # Находим самые популярные товары среди купленных
    if 'target' in users_data.columns:
        popular_items = users_data[users_data['target'] == 1]['itemid'].value_counts()
    else:
        # Если нет колонки target, используем event == 1
        popular_items = users_data[users_data['event'] == 1]['itemid'].value_counts()
    
    # Берем top_k самых популярных
    top_items = popular_items.head(top_k).index.tolist()
    
    # Возвращаем с фиксированной вероятностью 0.5
    return [(int(item_id), 0.5) for item_id in top_items]

def recommend_items_for_user(
    user_id: int,
    model,
    all_items_data: pd.DataFrame,
    users_data: pd.DataFrame,
    feature_columns: List[str],
    top_k: int = 3,
    threshold: float = None
) -> List[Tuple[int, float]]:
    """
    Основная функция рекомендаций для пользователя
    
    Args:
        user_id: ID пользователя
        model: обученная модель
        all_items_data: DataFrame с фичами товаров
        users_data: DataFrame с событиями пользователей
        feature_columns: список фич
        top_k: количество рекомендаций
        threshold: порог вероятности
    
    Returns:
        Список кортежей (item_id, probability)
    """
    # Проверяем, есть ли пользователь в данных
    if user_id not in users_data['visitorid'].values:
        print(f'Пользователь {user_id} новый - используем популярные товары')
        return get_popular_items(all_items_data, users_data, top_k)
    
    # Берем последнее событие пользователя
    user_events = users_data[users_data['visitorid'] == user_id]
    if len(user_events) == 0:
        print(f'Пользователь {user_id} не имеет событий')
        return get_popular_items(all_items_data, users_data, top_k)
    
    last_event = user_events.iloc[-1]
    user_features = last_event[feature_columns].to_dict()
    
    # Получаем уже купленные товары (чтобы не рекомендовать их повторно)
    if 'target' in users_data.columns:
        purchased_items = set(users_data[
            (users_data['visitorid'] == user_id) & 
            (users_data['target'] == 1)
        ]['itemid'].unique())
    else:
        # Если нет колонки target, используем event == 1
        purchased_items = set(users_data[
            (users_data['visitorid'] == user_id) & 
            (users_data['event'] == 1)
        ]['itemid'].unique())
    
    recommendations = []
    
    # Ограничиваем количество проверяемых товаров для скорости
    n_items_to_check = min(1000, len(all_items_data))
    
    # Берем случайные товары, но исключаем из них уже купленные
    available_items = [item_id for item_id in all_items_data.index if item_id not in purchased_items]
    
    if len(available_items) == 0:
        print(f'У пользователя {user_id} уже куплены все доступные товары')
        return []
    
    n_items_to_check = min(n_items_to_check, len(available_items))
    
    # Для воспроизводимости используем user_id как seed
    np.random.seed(user_id % 10000)
    sampled_items_idx = np.random.choice(available_items, size=n_items_to_check, replace=False)
    sampled_items = all_items_data.loc[sampled_items_idx]
    
    for item_id, item_features in sampled_items.iterrows():
        # Создаем фичи для пары (user, item)
        combined_features = user_features.copy()
        
        for col in feature_columns:
            if col in item_features:
                combined_features[col] = item_features[col]
        
        try:
            X_pair = pd.DataFrame([combined_features])[feature_columns]
            proba = model.predict_proba(X_pair)[0][1]
            
            recommendations.append((int(item_id), float(proba)))
        except Exception as e:
            # В случае ошибки пропускаем этот товар
            continue
    
    # Сортируем по вероятности
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Применяем порог, если указан
    if threshold is not None:
        recommendations = [(item, prob) for item, prob in recommendations if prob >= threshold]
    
    # Возвращаем топ-K рекомендаций
    return recommendations[:top_k]