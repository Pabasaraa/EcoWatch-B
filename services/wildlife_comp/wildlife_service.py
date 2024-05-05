import os
import cv2
import numpy as np
# import cvlib as cv
# from cvlib.object_detection import draw_bbox
from keras.models import load_model
# from openpyxl import Workbook, load_workbook
from datetime import datetime, timedelta
from pathlib import Path


prediction_results = {}


class WildlifeService():

    def __init__(self):
        BASE_DIR = Path(__file__).resolve().parent
        mod = load_model(f"{BASE_DIR}/animal_classification_model.h5", compile=False)
        self.model = mod
        self.animal_classes = ["antelope", "bear", "boar", "deer", "eagle", "elephant", "fox", "goat", "lion", "owl", "porcupine", "reindeer", "squirrel", "swan", "tiger", "wolf"]
        self.workbook_path = 'animal_counts.xlsx'
        self.wb = None
        self.ws = None
        

    def process_video(self, video_path):
        # cap = cv2.VideoCapture(video_path)
        # frame_rate = cap.get(cv2.CAP_PROP_FPS)
        # frame_interval = int(frame_rate * 1)
        # frame_counter = 0
        # animal_id = 1
        
        # if not os.path.exists(self.workbook_path):
        #     self.wb = Workbook()
        #     self.ws = self.wb.active
        #     self.ws.append(['Animal ID', 'Frame ID', 'Animal Name', 'Animal Count', 'Frame Time'])
        # else:
        #     self.wb = load_workbook(self.workbook_path)
        #     self.ws = self.wb.active

        # # video_start_date = input("Enter video start date (YYYY-MM-DD): ")
        # # video_start_time = input("Enter video start time (HH:MM:SS): ")

        # video_start_date="2024-12-12"
        # video_start_time="12:12:00"
        # video_start_datetime = datetime.strptime(video_start_date + ' ' + video_start_time, "%Y-%m-%d %H:%M:%S")

        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
            
        #     frame_counter += 1

        #     if frame_counter % frame_interval == 0:
        #         boxes, labels, count = cv.detect_common_objects(frame)
        #         animal_counts = {}

        #         for label in labels:
        #             if label.lower() in self.animal_classes:
        #                 animal_counts[label] = animal_counts.get(label, 0) + 1

        #         output = draw_bbox(frame, boxes, labels, count)
        #         output_path = os.path.join('output_frames', 'frame_' + str(frame_counter) + '.jpg')
        #         cv2.imwrite(output_path, output)

        #         frame_time = video_start_datetime + timedelta(seconds=(frame_counter / frame_rate))
        #         frame_time_str = frame_time.strftime("%Y-%m-%d %H:%M:%S")

        #         print("Frame {}: Number of animals detected: {}".format(frame_counter, len(animal_counts)))
        #         print("Animal names and counts:", animal_counts)

        #         for animal_name, animal_count in animal_counts.items():
        #             self.ws.append([animal_id, frame_counter, animal_name.capitalize(), animal_count, frame_time_str])
        #             animal_id += 1

        # self.wb.save(self.workbook_path)
        # cap.release()
        return None