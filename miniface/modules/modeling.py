# built-in dependencies
from typing import Any

# project dependencies
from miniface.models.facial_recognition import Facenet
from miniface.models.face_detection import MediaPipe


def build_model(task: str, model_name: str) -> Any:
    """
    This function loads a pre-trained models as singletonish way
    Parameters:
        task (str): facial_recognition, facial_attribute, face_detector, spoofing
        model_name (str): model identifier
            - VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
                ArcFace, SFace, GhostFaceNet for face recognition
            - Age, Gender, Emotion, Race for facial attributes
            - opencv, mtcnn, ssd, dlib, retinaface, mediapipe, yolov8, yunet,
                fastmtcnn or centerface for face detectors
            - Fasnet for spoofing
    Returns:
            built model class
    """

    # singleton design pattern
    global cached_models

    models = {
        "facial_recognition": {
            "Facenet512": Facenet.FaceNet512dClient,
        },
        "face_detector": {
            "mediapipe": MediaPipe.MediaPipeClient,
        },
    }

    if models.get(task) is None:
        raise ValueError(f"unimplemented task - {task}")

    if not "cached_models" in globals():
        cached_models = {current_task: {} for current_task in models.keys()}

    if cached_models[task].get(model_name) is None:
        model = models[task].get(model_name)
        if model:
            cached_models[task][model_name] = model()
        else:
            raise ValueError(f"Invalid model_name passed - {task}/{model_name}")

    return cached_models[task][model_name]
