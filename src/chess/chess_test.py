import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

def check():
    # 設定測試資料夾
    test_path = "./datasets/chess_pieces/test/images"
    output_path = "./datasets/chess_pieces/output/"

    # 確保輸出資料夾存在
    os.makedirs(output_path, exist_ok=True)

    # 載入 YOLO 模型
    model = YOLO('./runs/detect/train/weights/best.pt')

    # 取得所有圖片
    image_files = [f for f in os.listdir(test_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # 遍歷所有測試圖片
    for img_name in image_files:
        img_path = os.path.join(test_path, img_name)

        # 讀取圖片
        img = cv2.imread(img_path)

        # 進行 YOLO 偵測
        results = model(img)

        # 初始化標籤計數器
        label_counts = Counter()

        # 繪製標註框
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # 取得 Bounding Box
            labels = result.boxes.cls.cpu().numpy()  # 取得標籤 ID
            scores = result.boxes.conf.cpu().numpy()  # 取得信心分數

            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(int, box)  # 取得座標 (x1, y1, x2, y2)
                label = int(label)
                label_text = f"ID:{label} {score:.2f}"

                # 計算 ID = 1 和 ID = 7 的數量
                if label in [1, 7]:
                    label_counts[label] += 1

                # 設定不同顏色 (ID=1: 藍色, ID=7: 紅色, 其他: 綠色)
                color = (0, 255, 0)  # 預設綠色
                if label == 1:
                    color = (255, 0, 0)  # 藍色
                elif label == 7:
                    color = (0, 0, 255)  # 紅色

                # 畫框
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # 標示文字 (白色底 + 黑色字)
                cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 在左上角顯示 ID=1 和 ID=7 的數量
        count_text = f"ID 1: {label_counts[1]}  ID 7: {label_counts[7]}"
        cv2.putText(img, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 儲存圖片到 output 資料夾
        output_img_path = os.path.join(output_path, img_name)
        cv2.imwrite(output_img_path, img)

    print("標示完成，結果已儲存至:", output_path)


if __name__ == '__main__':
    check()