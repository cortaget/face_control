from deepface import DeepFace
import cv2
import os

# Указываем путь к изображениям
image_path_1 = "face_3.jpg"
image_path_2 = "face_4.jpg"  # Путь ко второму изображению для сравнения

# Загружаем изображения с проверкой
image_1 = cv2.imread(image_path_1)
image_2 = cv2.imread(image_path_2)

if image_1 is None or image_2 is None:
    print("Ошибка: одно или оба изображения не найдены по указанным путям.")
    exit()

# Устанавливаем пути к файлам классификаторов
face_cascade_path = os.path.join("haarcascades", "haarcascade_frontalface_default.xml")
eye_cascade_path = os.path.join("haarcascades", "haarcascade_eye.xml")

# Проверяем, что файлы классификаторов существуют
if not os.path.exists(face_cascade_path) or not os.path.exists(eye_cascade_path):
    print("Ошибка: файлы классификаторов не найдены.")
    exit()

# Загружаем классификаторы вручную (если DeepFace будет использовать OpenCV бэкенд для детекции)
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Сравниваем два изображения с параметром enforce_detection=False
verification_result = DeepFace.verify(
    image_1,
    image_2,
    enforce_detection=False,
    model_name='Facenet',
    threshold=0.6
)

# Выводим результат сравнения
print("Сравнение лиц:", verification_result)
