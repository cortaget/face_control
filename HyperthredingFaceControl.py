from deepface import DeepFace
import cv2
import os
from concurrent.futures import ProcessPoolExecutor


def preprocess_image(image_path, output_folder):
    """
    Функция предобработки изображения:
    - загрузка изображения
    - изменение размера
    - размытие
    - преобразование в оттенки серого
    - выравнивание гистограммы
    - сохранение обработанного изображения в новую папку
    """
    image = cv2.imread(image_path)

    if image is None:
        return None

    # Изменяем размер изображений для консистентности
    image = cv2.resize(image, (224, 224))

    # Применяем размытие
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Преобразуем изображения в оттенки серого
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применяем выравнивание гистограммы
    image_eq = cv2.equalizeHist(image_gray)

    # Сохраняем обработанное изображение в указанную папку
    processed_image_path = os.path.join(output_folder, os.path.basename(image_path).replace(".", "_processed."))
    cv2.imwrite(processed_image_path, image_eq)

    return processed_image_path


def compare_faces(target_image_path, compare_image_path):
    """
    Функция сравнения двух изображений.
    """
    try:
        # Сравниваем изображения с параметром enforce_detection=False
        result = DeepFace.verify(
            target_image_path,
            compare_image_path,
            enforce_detection=False,
            model_name='Facenet',
            threshold=0.6,
            detector_backend='mtcnn'
        )
        return compare_image_path, result['verified'], result['distance']
    except Exception as e:
        return compare_image_path, False, str(e)


def batch_compare(target_image_path, folder_path):
    """
    Сравнение изображения с каждым файлом в указанной папке.
    """
    # Получаем список всех файлов в папке
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Используем многопроцессорность для ускорения сравнения
    results = []
    with ProcessPoolExecutor() as executor:
        # Создаём задачи для сравнения
        tasks = [executor.submit(compare_faces, target_image_path, img) for img in image_files]

        # Собираем результаты
        for future in tasks:
            results.append(future.result())

    return results


def process_folder_images(input_folder, output_folder):
    """
    Обрабатывает все изображения в папке, применяя предобработку ко всем из них
    и сохраняет обработанные изображения в другую папку.
    """
    # Проверяем, существует ли папка для сохранения обработанных изображений
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    processed_images = []
    # Применяем обработку к каждому изображению в папке
    for image_path in image_files:
        processed_image = preprocess_image(image_path, output_folder)
        if processed_image is not None:
            processed_images.append(processed_image)
            print(f"Обработано изображение: {image_path}")
        else:
            print(f"Ошибка обработки изображения: {image_path}")

    return processed_images


if __name__ == "__main__":
    # Указываем путь к целевому изображению и папкемриль
    target_image_path = "face_5.jpg"
    folder_path = "img"  # Путь к папке с изображениями для сравнения
    processed_folder = "processed_images"  # Папка для сохранённых обработанных изображений

    # Проверяем наличие целевого изображения и папки
    if not os.path.exists(target_image_path):
        print("Ошибка: целевое изображение не найдено.")
        exit()

    if not os.path.isdir(folder_path):
        print("Ошибка: папка для сравнения не найдена.")
        exit()

    # Применяем предобработку ко всем изображениям в папке и сохраняем их в другую папку
    #processed_images = process_folder_images(folder_path, processed_folder)

    # Запускаем сравнение
    comparison_results = batch_compare(target_image_path, processed_folder)

    # Выводим результаты
    print("Результаты сравнения:")
    for img_path, verified, distance in comparison_results:
        print(f"Изображение: {img_path} | Совпадение: {verified} | Дистанция: {distance}")
