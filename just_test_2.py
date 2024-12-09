from deepface import DeepFace
import cv2

# Указываем путь к изображениям
image_path_1 = "face_5.jpg"
image_path_2 = "face_4.jpg"  # Путь ко второму изображению для сравнения

# Загружаем изображения с проверкой
image_1 = cv2.imread(image_path_1)
image_2 = cv2.imread(image_path_2)

if image_1 is None or image_2 is None:
    print("Ошибка: одно или оба изображения не найдены по указанным путям.")
    exit()

# Выполняем анализ для первого изображения
result_1 = DeepFace.analyze(image_1, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)
print("Анализ первого лица:", result_1)


# Выводим результат сравнения
print("Сравнение лиц:", verification_result)
