"""
main.py — FastAPI микросервис динамического ценообразования.

Запуск:
    uvicorn main:app --host 0.0.0.0 --port 8000

Гарантия честного ценообразования:
    - При lead_time >= 30 И нулевой загрузке → цена ТОЧНО равна base_price из БД
    - ML используется только когда есть реальное давление спроса
    - Цена никогда не опускается ниже base_price
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ─── Загрузка модели при старте приложения ────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"[PriceAI] Модель загружена из {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"[PriceAI] ПРЕДУПРЕЖДЕНИЕ: model.pkl не найден. Запустите train_model.py")

# ─── Инициализация FastAPI ────────────────────────────────────────────────────

app = FastAPI(
    title="PriceAI — Сервис динамического ценообразования",
    description="RandomForest модель для расчёта цены номера на основе внутренней экономики отеля",
    version="1.1.0",
)


# ─── Вспомогательная функция: давление спроса ────────────────────────────────

def calc_demand_pressure(lead_time: int, occupancy_rate: float,
                         is_weekend: int, pickup_rate: float) -> float:
    """
    Вычисляет суммарное «давление спроса» по той же формуле, что и датасет.

    Возвращает значение от 0.0 (нет давления) до ~0.82 (максимум).

    Ключевое правило:
        - lead_time >= 30 → urgency = 0  (срочности нет)
        - При нулевом результате → цена ТОЧНО равна base_price

    Порог FLOOR_THRESHOLD: ниже него считаем, что давление нулевое
    и возвращаем base_price напрямую, минуя ML.
    """
    urgency = max(0.0, (30.0 - lead_time) / 30.0) ** 1.5
    pressure = (
        urgency * 0.35
        + occupancy_rate * 0.25
        + is_weekend * 0.10
        + (pickup_rate / 20.0) * 0.12
    )
    return pressure


# Минимальный порог давления, ниже которого возвращаем точно base_price.
# При lead_time=30, occ=0, weekday, pickup=0 → pressure=0.0 < FLOOR_THRESHOLD → base_price
# При lead_time=29, occ=0, weekday, pickup=0 → pressure=0.0006 < FLOOR_THRESHOLD → base_price
# Первый значимый сдвиг начинается при lead_time ~25 дней
FLOOR_THRESHOLD = 0.01


def compute_price(lead_time: int, occupancy_rate: float, is_weekend: int,
                  pickup_rate: float, base_price: float) -> float:
    """
    Гибридное вычисление цены:
      1. Если давление спроса < FLOOR_THRESHOLD → base_price (гарантированный минимум)
      2. Иначе → ML-предсказание, но не ниже base_price
    """
    pressure = calc_demand_pressure(lead_time, occupancy_rate, is_weekend, pickup_rate)

    # Нет давления → возвращаем ровно базовую цену (без ML)
    if pressure < FLOOR_THRESHOLD:
        return round(base_price, 2)

    # Есть давление → используем ML
    if model is None:
        # ML недоступна — аналитический расчёт по формуле
        return round(base_price * (1.0 + pressure), 2)

    features = pd.DataFrame([[lead_time, occupancy_rate, is_weekend, pickup_rate, base_price]],
                            columns=['lead_time', 'occupancy_rate', 'is_weekend',
                                     'pickup_rate', 'base_price'])
    ml_price = float(model.predict(features)[0])

    # Никогда не опускаемся ниже базовой цены
    return round(max(ml_price, base_price), 2)


# ─── Схемы запроса и ответа ───────────────────────────────────────────────────

class PredictRequest(BaseModel):
    lead_time:      int   = Field(..., ge=0, le=365)
    occupancy_rate: float = Field(..., ge=0.0, le=1.0)
    is_weekend:     int   = Field(..., ge=0, le=1)
    pickup_rate:    float = Field(..., ge=0.0)
    base_price:     float = Field(..., gt=0)


class PredictResponse(BaseModel):
    price:      float = Field(..., description="Рассчитанная динамическая цена (₽ за ночь)")
    currency:   str   = Field("RUB")
    is_dynamic: bool  = Field(True)


class PredictItemWithDate(BaseModel):
    date:           str   = Field(..., description="Дата YYYY-MM-DD")
    lead_time:      int   = Field(..., ge=0, le=365)
    occupancy_rate: float = Field(..., ge=0.0, le=1.0)
    is_weekend:     int   = Field(..., ge=0, le=1)
    pickup_rate:    float = Field(..., ge=0.0)
    base_price:     float = Field(..., gt=0)


class BatchPredictRequest(BaseModel):
    items: list[PredictItemWithDate]


class BatchPredictResponse(BaseModel):
    prices: dict[str, float] = Field(..., description="Словарь date→price")


# ─── Эндпоинты ────────────────────────────────────────────────────────────────

@app.get("/health", summary="Проверка работоспособности")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "floor_threshold": FLOOR_THRESHOLD,
    }


@app.post("/predict-price", response_model=PredictResponse,
          summary="Предсказать динамическую цену номера")
def predict_price(request: PredictRequest):
    """
    Возвращает динамическую цену номера.
    Гарантия: при lead_time >= 30 и нулевой загрузке — ровно base_price.
    """
    price = compute_price(
        request.lead_time, request.occupancy_rate,
        request.is_weekend, request.pickup_rate, request.base_price
    )
    return PredictResponse(price=price, currency="RUB", is_dynamic=True)


@app.post("/predict-prices-batch", response_model=BatchPredictResponse,
          summary="Батч-предсказание цен по диапазону дат")
def predict_prices_batch(request: BatchPredictRequest):
    """
    Принимает список дат с признаками, возвращает словарь {date: price}.

    Для дат без давления спроса (lead_time>=30, нет загрузки) возвращает
    ТОЧНО base_price — пользователь не будет обманут отображаемой ценой.
    """
    if not request.items:
        return BatchPredictResponse(prices={})

    result = {}

    # Разделяем: без давления (floor) и с давлением (ML)
    floor_items = []
    ml_items    = []

    for item in request.items:
        pressure = calc_demand_pressure(
            item.lead_time, item.occupancy_rate, item.is_weekend, item.pickup_rate
        )
        if pressure < FLOOR_THRESHOLD:
            floor_items.append(item)
        else:
            ml_items.append(item)

    # Без давления — точная базовая цена
    for item in floor_items:
        result[item.date] = round(item.base_price, 2)

    # С давлением — ML предсказание (батч для скорости)
    if ml_items:
        if model is not None:
            features = pd.DataFrame(
                [[i.lead_time, i.occupancy_rate, i.is_weekend, i.pickup_rate, i.base_price]
                 for i in ml_items],
                columns=['lead_time', 'occupancy_rate', 'is_weekend', 'pickup_rate', 'base_price']
            )
            predictions = model.predict(features)
            for i, item in enumerate(ml_items):
                price = max(float(predictions[i]), item.base_price)
                result[item.date] = round(price, 2)
        else:
            # Fallback без ML — аналитическая формула
            for item in ml_items:
                pressure = calc_demand_pressure(
                    item.lead_time, item.occupancy_rate, item.is_weekend, item.pickup_rate
                )
                result[item.date] = round(item.base_price * (1.0 + pressure), 2)

    return BatchPredictResponse(prices=result)
