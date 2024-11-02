from deepface import DeepFace


def face_verify(img_1,img_2):
    try:
        result_dict = DeepFace.verify(img1_path = img_1, img2_path = img_2)
    except Exception as _ex:
        return _ex