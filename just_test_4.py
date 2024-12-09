from deepface import DeepFace
import cv2
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# Указываем путь к изображениям
image_path_1 = "face_5.jpg"
image_path_2 = "face_4.jpg"  # Путь ко второму изображению для сравнения

# Загружаем изображения с проверкой
image_1 = cv2.imread(image_path_1)
image_2 = cv2.imread(image_path_2)











if image_1 is None or image_2 is None:
    print("Ошибка: одно или оба изображения не найдены по указанным путям.")
    exit()



image_1 = cv2.resize(image_1, (224, 224))  # Например, до 224x224 для моделей
image_2 = cv2.resize(image_2, (224, 224))

image_1 = cv2.GaussianBlur(image_1, (5, 5), 0)
image_2 = cv2.GaussianBlur(image_2, (5, 5), 0)




# Устанавливаем пути к файлам классификаторов
face_cascade_path = os.path.join("haarcascades", "haarcascade_frontalface_default.xml")
eye_cascade_path = os.path.join("haarcascades", "haarcascade_eye.xml")
cascade_path = os.path.join("haarcascades", "cascade.xml")

# Проверяем, что файлы классификаторов существуют
if not os.path.exists(face_cascade_path) or not os.path.exists(eye_cascade_path) or not os.path.exists(cascade_path):
    print("Ошибка: файлы классификаторов не найдены.")
    exit()

# Загружаем классификаторы вручную (если DeepFace будет использовать OpenCV бэкенд для детекции)
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
cascade = cv2.CascadeClassifier(cascade_path)





# Сравниваем два изображения с параметром enforce_detection=False
verification_result = DeepFace.verify(
    image_1,
    image_2,
    enforce_detection=False,
    model_name='Facenet',
    threshold=0.6,
    detector_backend='mtcnn'  # Используем MTCNN вместо OpenCV
)


# Выводим результат сравнения
print("Сравнение лиц:", verification_result)
