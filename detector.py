# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:46:22 2023

@author: A102919
"""


from pathlib import Path
import pickle
import face_recognition
from PIL import Image, ImageDraw
from collections import Counter
import numpy as np

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

def encode_known_faces(
    model: str = "cnn", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations, num_jitters= 100, model="large")

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

def recognize_faces(
    frame: np.array,
    # image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
    makeup: bool = False,
):
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    # input_image = face_recognition.load_image_file(image_location)
    input_image = frame

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )


    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)

    if makeup:
        face_landmarks_list = face_recognition.face_landmarks(
            input_image
        )

        for face_landmarks in face_landmarks_list:

            # Make the eyebrows into a nightmare
            draw.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
            draw.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
            draw.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
            draw.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

            # Gloss the lips
            draw.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
            draw.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
            draw.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
            draw.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

            # Sparkle the eyes
            draw.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
            draw.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

            # Apply some eyeliner
            draw.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
            draw.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)


    del draw
    # pillow_image.show()

    return pillow_image

def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding, tolerance= 0.4
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]



def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )