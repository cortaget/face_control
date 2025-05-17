import cv2
import os
import sys
import datetime as dt
import threading
import queue
import numpy as np
import time
import RPi.GPIO as GPIO  # GPIO для управления светодиодом

# Настройка GPIO
LED_PIN = 16  # GPIO16
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)  # Изначально выключен

# Пути к важным файлам и папкам
cascade_path = "cascade.xml"
reference_folder = "img"

# Проверка наличия необходимых файлов
if not os.path.exists(cascade_path):
    print("Ошибка: не найден файл классификатора.")
    sys.exit()

if not os.path.exists(reference_folder) or not os.path.isdir(reference_folder):
    print(f"Ошибка: не найдена папка с изображениями для сравнения: {reference_folder}.")
    sys.exit()

# Инициализация детектора лиц
face_cascade = cv2.CascadeClassifier(cascade_path)

# Открытие камеры
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Ошибка: не удалось открыть камеру.")
    sys.exit()

# Установка параметров камеры для лучшего качества
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Папка для сохранения захваченных изображений
output_dir = "captured"
os.makedirs(output_dir, exist_ok=True)

# Переменные для управления состоянием
result_queue = queue.Queue()
is_comparing = False
state_lock = threading.Lock()
scanning_active = False
running = True  # Флаг для управления основным циклом

# Функция управления светодиодом
def control_led(recognized=False, duration=3):
    """
    Управление светодиодом при распознавании.
    При успешном распознавании светодиод включается на указанное время.
    
    Args:
        recognized: True если лицо распознано, False в противном случае
        duration: Длительность в секундах, в течение которой светодиод будет включен
    """
    if recognized:
        print("🟢 Активация светодиода: Доступ разрешен!")
        GPIO.output(LED_PIN, GPIO.HIGH)  # Включение светодиода
        time.sleep(duration)  # Светодиод горит указанное время
        GPIO.output(LED_PIN, GPIO.LOW)   # Выключение светодиода
    else:
        # Мигание светодиодом 3 раза при неудачном распознавании
        print("🔴 Доступ запрещен!")
        for _ in range(3):
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(0.2)
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(0.2)

# Функция предобработки изображения для улучшения распознавания
def preprocess_image(img):
    # Проверка, что изображение не пустое
    if img is None or img.size == 0:
        return None
        
    # Выравнивание гистограммы для улучшения контраста
    img_equalized = cv2.equalizeHist(img)
    
    # Нормализация размера
    img_resized = cv2.resize(img_equalized, (100, 100))
    
    # Применение фильтра Гаусса для уменьшения шума
    img_filtered = cv2.GaussianBlur(img_resized, (5, 5), 0)
    
    return img_filtered

# Улучшенная функция сравнения лиц
def compare_faces_opencv(saved_path):
    global is_comparing
    with state_lock:
        is_comparing = True
    
    access_granted = False  # Флаг для управления доступом
    
    try:
        # Загрузка и предобработка захваченного изображения
        saved_img = cv2.imread(saved_path, cv2.IMREAD_GRAYSCALE)
        saved_processed = preprocess_image(saved_img)
        if saved_processed is None:
            result_queue.put("⚠️ Ошибка обработки захваченного изображения")
            return
        
        # Создаем детектор для вычисления дескрипторов SIFT
        sift = cv2.SIFT_create()
        
        # Вычисляем ключевые точки и дескрипторы для захваченного изображения
        kp1, des1 = sift.detectAndCompute(saved_processed, None)
        
        # Если не удалось найти ключевые точки
        if des1 is None or len(des1) == 0:
            result_queue.put("⚠️ Не удалось обнаружить особенности на захваченном изображении")
            return
        
        # Получаем список изображений для сравнения
        db_images = [img for img in os.listdir(reference_folder)
                     if os.path.isfile(os.path.join(reference_folder, img))
                     and img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not db_images:
            result_queue.put("⚠️ В папке эталонов нет изображений!")
            return
        
        # Создаем сопоставитель особенностей для SIFT
        bf = cv2.BFMatcher()
        
        # Отслеживаем лучшее совпадение
        best_match = None
        best_score = 0
        match_found = False
        
        for db_img in db_images:
            db_path = os.path.join(reference_folder, db_img)
            ref_img = cv2.imread(db_path, cv2.IMREAD_GRAYSCALE)
            
            if ref_img is None:
                result_queue.put(f"⚠️ Не удалось загрузить изображение: {db_img}")
                continue
            
            # Предобработка эталонного изображения
            ref_processed = preprocess_image(ref_img)
            if ref_processed is None:
                result_queue.put(f"⚠️ Ошибка обработки изображения: {db_img}")
                continue
                
            # Вычисляем ключевые точки и дескрипторы для эталонного изображения
            kp2, des2 = sift.detectAndCompute(ref_processed, None)
            
            if des2 is None or len(des2) == 0:
                result_queue.put(f"⚠️ Не удалось обнаружить особенности на изображении: {db_img}")
                continue
            
            # Одновременно используем два метода сравнения
            
            # Метод 1: SIFT с соответствием особенностей
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Применяем фильтр отношения для нахождения хороших совпадений
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:  # Тест Лоу для отбора хороших совпадений
                    good_matches.append(m)
            
            # Рассчитываем оценку на основе хороших совпадений
            match_score1 = len(good_matches) / max(len(kp1), len(kp2)) * 100 if max(len(kp1), len(kp2)) > 0 else 0
            
            # Метод 2: Сравнение гистограмм (альтернативный метод)
            hist1 = cv2.calcHist([saved_processed], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([ref_processed], [0], None, [256], [0, 256])
            
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            # Сравниваем гистограммы
            hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL) * 100  # 0-100%
            
            # Комбинированная оценка (можно настроить веса)
            combined_score = 0.7 * match_score1 + 0.3 * hist_score
            
            # Записываем результат сравнения
            result_queue.put(f"Анализ {db_img}: SIFT={match_score1:.1f}%, Гист={hist_score:.1f}%, Итог={combined_score:.1f}%")
            
            # Сохраняем лучший результат
            if combined_score > best_score:
                best_score = combined_score
                best_match = db_img
            
            # Порог для определения совпадения (можно настроить)
            if combined_score > 30:  # Этот порог менее строгий
                match_found = True
        
        # Вывод итогового результата
        if match_found:
            if best_match:
                result_queue.put(f"✅ Лучшее совпадение: {best_match} (оценка={best_score:.1f}%)")
                print("Привет 😘")
                access_granted = True  # Установка флага доступа
        else:
            result_queue.put(f"⚠️ Совпадений не найдено. Лучший результат: {best_match or 'нет'} ({best_score:.1f}%)")
            
    except Exception as e:
        result_queue.put(f"Ошибка в процессе сравнения: {str(e)}")
    finally:
        result_queue.put("DONE")
        # Активируем светодиод в зависимости от результата распознавания
        led_thread = threading.Thread(target=control_led, args=(access_granted, 5))
        led_thread.start()
        
        with state_lock:
            is_comparing = False

# Функция автоматического сканирования
def auto_scan():
    global scanning_active, running
    
    print("📊 Запущен режим автоматического сканирования")
    print("Нажмите Ctrl+C для завершения программы.")
    
    while running:
        try:
            # Если не выполняется сравнение, начинаем новый цикл сканирования
            if not is_comparing:
                with state_lock:
                    if scanning_active:
                        time.sleep(0.5)  # Ждем немного, если уже сканируем
                        continue
                    scanning_active = True
                
                print("\n🔍 Начинаем сканирование...")
                ret, frame = video_capture.read()
                if not ret:
                    print("⚠️ Ошибка захвата кадра.")
                    scanning_active = False
                    time.sleep(1)
                    continue
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                if len(faces) == 0:
                    print("⚠️ Лицо не обнаружено.")
                    scanning_active = False
                    time.sleep(1)  # Небольшая пауза перед новой попыткой
                    continue
                    
                print(f"📊 Обнаружено лиц: {len(faces)}")
                
                # Берем самое большое лицо (предположительно ближайшее к камере)
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                (x, y, w, h) = largest_face
                
                # Увеличиваем область лица для лучшего распознавания
                x_extended = max(0, x - int(w * 0.2))
                y_extended = max(0, y - int(h * 0.2))
                w_extended = min(frame.shape[1] - x_extended, int(w * 1.4))
                h_extended = min(frame.shape[0] - y_extended, int(h * 1.4))
                
                face_img = frame[y_extended:y_extended + h_extended, x_extended:x_extended + w_extended]
                
                # Сохраняем изображение лица
                timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"face_{timestamp}.jpg"
                saved_path = os.path.join(output_dir, filename)
                cv2.imwrite(saved_path, face_img)
                
                print(f"📸 Лицо сохранено: {filename}")
                print(f"🔍 Сравнение с изображениями из папки '{reference_folder}'...")
                
                # Запускаем сравнение в отдельном потоке
                comparison_thread = threading.Thread(target=compare_faces_opencv, args=(saved_path,))
                comparison_thread.daemon = True
                comparison_thread.start()
                
                # Ждём завершения и выводим результаты
                while True:
                    try:
                        result = result_queue.get(timeout=10)  # 10 секунд таймаут
                        if result == "DONE":
                            print("✅ Сравнение завершено")
                            break
                        else:
                            print(result)
                    except queue.Empty:
                        print("⚠️ Таймаут при ожидании результатов сравнения")
                        break
                
                scanning_active = False
                time.sleep(2)  # Пауза между сканированиями
            else:
                time.sleep(0.5)  # Ждем завершения текущего сравнения
        except Exception as e:
            print(f"Ошибка в цикле сканирования: {str(e)}")
            scanning_active = False
            time.sleep(1)

# Главная функция
def main():
    global running
    try:
        print("🔐 Система распознавания лиц с управлением GPIO запущена")
        print("Режим работы: автоматическое сканирование без графического интерфейса")
        
        # Запуск автоматического сканирования в отдельном потоке
        scan_thread = threading.Thread(target=auto_scan)
        scan_thread.daemon = True
        scan_thread.start()
        
        # Ждем нажатия Ctrl+C для завершения
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n✋ Программа прервана пользователем")
    finally:
        # Установка флага для завершения потоков
        running = False
        # Необходимо дать время потокам на корректное завершение
        time.sleep(1)
        # Освобождаем ресурсы
        video_capture.release()
        GPIO.cleanup()
        print("🔄 Ресурсы освобождены, программа завершена.")

if __name__ == "__main__":
    main()
