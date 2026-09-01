"""
train_model.py — обучение RandomForestRegressor для динамического ценообразования.

Ключевые принципы новой модели:
  - При lead_time >= 30 дней И нулевой загрузке → цена ≈ базовая (минимум)
  - Надбавка за срочность работает только в диапазоне 0–30 дней до заезда
  - Каждый фактор (загрузка, выходной, pickup) добавляет процент сверху базы
  - Итоговый диапазон: от base_price (низкий спрос, далеко) до ~+80% (срочно + полная загрузка)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# ─── 1. Генерация синтетического датасета ────────────────────────────────────

np.random.seed(42)
N = 3000  # увеличиваем датасет для лучшего обобщения

# Признаки
lead_time      = np.random.randint(0, 181, size=N)       # 0–180 дней (0–6 месяцев)
occupancy_rate = np.random.uniform(0.0, 1.0, size=N)     # загрузка отеля 0–100%
is_weekend     = np.random.randint(0, 2, size=N)          # 0 = будни, 1 = выходные
pickup_rate    = np.random.uniform(0.0, 20.0, size=N)    # номеров/сутки за 24ч

# Базовая цена (варьируется по типу номера)
base_price = np.random.uniform(2500.0, 8000.0, size=N)

# ── Коэффициент срочности ────────────────────────────────────────────────────
# Активируется только при lead_time < 30 дней, растёт по квадратичной кривой.
# При lead_time >= 30: urgency = 0 → цена остаётся на базовом уровне.
# При lead_time = 0:  urgency = 1 → максимальная надбавка за срочность.
urgency = np.maximum(0.0, (30.0 - lead_time) / 30.0) ** 1.5

# ── Итоговая цена (аддитивная модель, всё в % от base_price) ─────────────────
# Каждый фактор независимо добавляет свою долю сверх базовой цены:
price = base_price * (
    1.0
    + urgency * 0.35                        # срочность: до +35% (только последние 30 дней)
    + occupancy_rate * 0.25                 # загрузка:  до +25% при 100% occupancy
    + is_weekend * 0.10                     # выходные:  +10%
    + (pickup_rate / 20.0) * 0.12           # пиковый спрос: до +12%
) + np.random.normal(0, 120, size=N)        # реалистичный шум ±120 руб

# Жёсткое ограничение: цена не опускается ниже 98% базовой (без скидок)
price = np.maximum(price, base_price * 0.98)

# Контрольные значения для проверки логики
print("=" * 50)
print("[train_model] Проверка логики формулы:")
base_test = 5000.0

lead_occ_tests = [
    (90, 0.0, 0, 0,    "90 дней, нет брон, будни  → должно быть ≈ base"),
    (30, 0.0, 0, 0,    "30 дней, нет брон, будни  → должно быть ≈ base"),
    (14, 0.5, 0, 5,    "14 дней, 50% загр, будни  → умеренная надбавка"),
    (7,  0.8, 1, 10,   "7 дней,  80% загр, выходной → значительная надбавка"),
    (0,  1.0, 1, 20,   "0 дней,  100% загр, выходной, пик → максимум"),
]
for lt, occ, iw, pk, label in lead_occ_tests:
    urg = max(0.0, (30.0 - lt) / 30.0) ** 1.5
    p = base_test * (1.0 + urg * 0.35 + occ * 0.25 + iw * 0.10 + (pk / 20.0) * 0.12)
    pct = (p / base_test - 1) * 100
    print(f"  {label}")
    print(f"    → {p:,.0f} руб (+{pct:.1f}%)")
print("=" * 50)
print()

# ─── 2. Датасет ───────────────────────────────────────────────────────────────

df = pd.DataFrame({
    'lead_time':      lead_time,
    'occupancy_rate': occupancy_rate,
    'is_weekend':     is_weekend,
    'pickup_rate':    pickup_rate,
    'base_price':     base_price,
    'price':          price
})

print(f"[train_model] Датасет: {len(df)} строк")
print(f"[train_model] Диапазон цен: {price.min():.0f} - {price.max():.0f} RUB\n")

# ─── 3. Разделение train/test ─────────────────────────────────────────────────

FEATURES = ['lead_time', 'occupancy_rate', 'is_weekend', 'pickup_rate', 'base_price']
TARGET   = 'price'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"[train_model] Train: {len(X_train)} | Test: {len(X_test)}\n")

# ─── 4. Обучение RandomForestRegressor ───────────────────────────────────────

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,        # немного ограничиваем — меньше переобучение
    min_samples_leaf=8,  # листья с большим кол-вом примеров → сглаживает предсказание
    random_state=42,
    n_jobs=-1
)

print("[train_model] Обучение...")
model.fit(X_train, y_train)

# ─── 5. Метрики ───────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"[train_model] === Метрики ===")
print(f"[train_model] MAE: {mae:.1f} RUB")
print(f"[train_model] R2:  {r2:.4f}")
print()

# Важность признаков
importances = dict(zip(FEATURES, model.feature_importances_))
print("[train_model] === Важность признаков ===")
for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"[train_model]   {feat:<20} {imp:.4f}")
print()

# ─── 6. Сохранение ───────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
joblib.dump(model, MODEL_PATH)
print(f"[train_model] Модель сохранена: {MODEL_PATH}")
