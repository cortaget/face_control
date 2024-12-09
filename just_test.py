from deepface import DeepFace
import cv2



# Указываем путь к изображениям
image_path_1 = "face_3.jpg"
image_path_2 = "face_4.jpg"  # Путь ко второму изображению для сравнения

# Загружаем изображения с проверкой
image_1 = cv2.imread(image_path_1)
image_2 = cv2.imread(image_path_2)

if image_1 is None or image_2 is None:
    print("Ошибка: одно или оба изображения не найдены по указанным путям.")
    exit()


# Сравниваем два изображения с параметром enforce_detection=False

verification_result = DeepFace.verify(image_1, image_2, enforce_detection=False,model_name='Facenet', threshold=0.6)

# Выводим результат сравнения
print("Сравнение лиц:", verification_result)
