import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib
import time
import os

os.makedirs('models', exist_ok=True)
os.makedirs('images', exist_ok=True)

print("загрузка данных...")
df = pd.read_csv('data/emotions.csv')
df['target'] = df['label']

# кодирование меток
label_encoder = LabelEncoder()
df['target_encoded'] = label_encoder.fit_transform(df['target'])

# разделение на признаки и целевую переменную
X = df.drop(['label', 'target', 'target_encoded'], axis=1)
y = df['target_encoded']

# разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("начало обучения модели...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print(f"модель обучена! время обучения: {end_time - start_time:.2f} секунд")

# оценка качества модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nточность модели: {accuracy:.4f}")

target_names = label_encoder.classes_
print("\nдетальный отчет по классам:")
print(classification_report(y_test, y_pred, target_names=target_names))

# сохранение модели и кодировщика
model_filename = 'models/random_forest_model_3_classes.pkl'
joblib.dump(model, model_filename)
print(f"модель сохранена в файл: {model_filename}")

encoder_filename = 'models/label_encoder_3_classes.pkl'
joblib.dump(label_encoder, encoder_filename)
print(f"кодировщик меток сохранен в файл: {encoder_filename}")

# сохранение тестовой выборки
test_data_to_save = {
    'X_test': X_test,
    'y_test': y_test,
    'feature_names': X_train.columns.tolist(),
    'target_names': label_encoder.classes_.tolist()
}

test_data_filename = 'models/test_data.pkl'
joblib.dump(test_data_to_save, test_data_filename)
print(f"тестовая выборка сохранена в файл: {test_data_filename}")

print("\nобучение завершено успешно!")