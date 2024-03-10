import numpy as np
import cv2
from pathlib import Path
# import cvlib as cv
# from cvlib.object_detection import draw_bbox
import matplotlib.pyplot as plt
from keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
# model = load_model(f"{BASE_DIR}/model.hdf5", compile=False)

# The list of animal classes
animal_classes = ["antelope", "bear", "boar", "deer", "eagle", "elephant", "fox", "goat", "lion", "owl", "porcupine", "reindeer", "squirrel", "swan", "tiger", "wolf"]

def predict(input):
    # img = np.array(input)

    # if len(img.shape) == 2:
    #     img = np.stack((img,) * 3, axis=-1)

    # # Detect objects in the image using cvlib
    # boxes, labels, count = cv.detect_common_objects(img)

    # # Draw bounding boxes around the detected objects
    # output = draw_bbox(img, boxes, labels, count)

    # # Store the output image
    # output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    # final_prediction = np.argmax(output, axis=2)
    # # Count the number of animals in the image
    # animal_count = 0
    # for label in labels:
    #     if label.lower() in animal_classes:
    #         animal_count += 1

    # return final_prediction, animal_count
    return None
