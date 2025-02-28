import ultralytics
from ultralytics import YOLO
import cv2
import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image,display
from collections import Counter

import os

def readfile():
    root_dir = './datasets/chess_pieces'
    yaml_dir = './datasets/chess_pieces/data.yaml'
    test_path = "./datasets/chess_pieces/test/images"
    output_path = "./datasets/chess_pieces/output/"

    os.makedirs(output_path, exist_ok=True)

    # print(os.path.isfile(yaml_dir))
    train_path = os.path.join(root_dir,'train','images')
    valid_path = os.path.join(root_dir,'valid','images')
    model = YOLO('yolov8s.pt')
    results = model.train(
        data=yaml_dir,
        epochs=20,
        batch=32,
        lr0=0.0001,
        lrf=0.1,
        imgsz=640,
        plots=True
    )

    # image_files = [f for f in os.listdir(test_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    # total_counts = Counter()

    # display(Image(filename='./runs/detect/train/results.png', width=1220))



if __name__ == '__main__':
    ultralytics.checks()
    readfile()