from deepface import DeepFace
import cv2





def compare_faces(image_path1, image_path2):
    # Сравниваем два изображения
    result = DeepFace.verify(image_path1, image_path2)

    # Выводим результат
    if result['verified']:
        print("Лица совпадают!")
    else:
        print("Лица разные!")

if __name__ == "__main__":
    image1 = "face_1.jpg"
    image2 = "face_2.jpg"
    compare_faces(image1, image2)
