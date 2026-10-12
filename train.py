import numpy as np
import random
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

def main():
    # 1. Генерация и фиксация случайного SEED для воспроизводимости
    SEED = random.randint(1000, 9999)
    np.random.seed(SEED)
    random.seed(SEED)
    print(f"Используемый SEED: {SEED}")

    # Сохраняем SEED в файл, чтобы predict.py подставил его в название архива
    with open('seed.txt', 'w') as f:
        f.write(str(SEED))

    # 2. Загрузка данных
    print("Загрузка обучающей выборки (train.npz)...")
    train_data = np.load('train.npz', allow_pickle=True)
    X = train_data['embeddings']
    targets = train_data['targets']

    # 3. Кодирование текстовых меток в числа
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(targets)
    classes = label_encoder.classes_

    # Сохраняем энкодер
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # 4. Разбиение на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.15, random_state=SEED, stratify=y_encoded
    )

    print(f"Размер обучающей выборки: {X_train.shape[0]}")
    print(f"Размер валидационной выборки: {X_val.shape[0]}")

    # 5. Обучение модели
    # Используем solver='saga', который отлично оптимизирован для больших объемов данных
    print("Обучение модели (Logistic Regression с solver='saga')...")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            solver='saga',
            class_weight='balanced', 
            max_iter=100, 
            random_state=SEED
        ))
    ])

    model.fit(X_train, y_train)

    # Сохранение весов модели
    with open('model_weights.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Модель успешно сохранена в model_weights.pkl")

    # 6. Валидация
    y_pred_val = model.predict(X_val)
    f1_macro = f1_score(y_val, y_pred_val, average='macro')

    print("\n" + "="*50)
    print(f"Validation macro-F1: {f1_macro:.4f}")
    print("="*50)
    print(classification_report(y_val, y_pred_val, target_names=classes))

if __name__ == "__main__":
    main()