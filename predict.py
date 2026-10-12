import numpy as np
import pickle
import os

def main():
    # 1. Чтение SEED
    if os.path.exists('seed.txt'):
        with open('seed.txt', 'r') as f:
            SEED = int(f.read().strip())
    else:
        SEED = 42  # Резервное значение
    
    print(f"Используемый SEED для имени файла: {SEED}")

    # 2. Загрузка модели и энкодера
    if not os.path.exists('model_weights.pkl') or not os.path.exists('label_encoder.pkl'):
        print("Ошибка: Сначала запустите train.py для обучения модели!")
        return

    print("Загрузка модели и энкодера...")
    with open('model_weights.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    # 3. Загрузка тестовых данных
    print("Загрузка тестовой выборки (public_test.npz)...")
    test_data = np.load('public_test.npz', allow_pickle=True)
    X_test = test_data['embeddings']

    # 4. Предсказание
    print("Генерация предсказаний...")
    y_pred_encoded = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    # 5. Сохранение результата
    submission_filename = f"submission_seed_{SEED}.npz"
    np.savez_compressed(
        submission_filename,
        embeddings=X_test,
        targets=y_pred_labels
    )

    print(f"Готово! Файл с предсказаниями сохранен как: {submission_filename}")

if __name__ == "__main__":
    main()