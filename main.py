import cv2
import sys
import datetime as dt
from time import sleep

# Путь к файлу с классификатором Хаара для распознавания лиц
cascPath = "cascade.xml"

# Загружаем классификатор
faceCascade = cv2.CascadeClassifier(cascPath)

# Проверка успешности загрузки классификатора
if faceCascade.empty():
    print("Ошибка загрузки классификатора. Проверьте путь к 'cascade.xml'.")
    sys.exit()

# Захват видео с камеры
video_capture = cv2.VideoCapture(0)
anterior = 0

# Основной цикл
while True:
    # Проверяем, открыто ли видео
    if not video_capture.isOpened():
        print('Невозможно загрузить видео с камеры!')
        sleep(5)
        continue

    # Чтение кадров с камеры
    ret, frame = video_capture.read()
    if not ret:
        print("Ошибка захвата кадра.")
        break

    # Преобразуем изображение в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на изображении
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Рисуем прямоугольники вокруг обнаруженных лиц и сохраняем их
    for i, (x, y, w, h) in enumerate(faces):
        # Рисуем прямоугольник вокруг лица
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Надпись "Лицо обнаружено"
        cv2.putText(frame, "Лицо обнаружено", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Сохраняем изображение лица
        face_img = frame[y:y + h, x:x + w]
        filename = f"/Users/Максим/PycharmProjects/pythonProject/img/face_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
        cv2.imwrite(filename, face_img)
        print(f"Сохранено лицо: {filename}")

    # Выводим количество лиц при изменении
    if anterior != len(faces):
        print(f"Обнаружено лиц: {len(faces)}")
        anterior = len(faces)

    # Отображение видео с наложенными прямоугольниками
    cv2.imshow('Video', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
video_capture.release()
cv2.destroyAllWindows()
