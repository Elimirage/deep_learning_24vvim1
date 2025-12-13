#Ссылка на гугл диск https://drive.google.com/drive/folders/1YdVLzgIl2zIFg0mQ43H_SHDtKjkt-FSl?usp=sharing
# -*- coding: utf-8 -*-
"""
YOLOv8 обучение на фруктовом датасете
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


import ultralytics
ultralytics.checks()


model = YOLO("yolov8s.pt")

results = model.train(
    data="data.yaml",   # путь к YAML файлу с фруктовым датасетом
    model="yolov8s.pt", # предобученные веса
    epochs=30,
    batch=16,
    imgsz=640,
    project="fruits_runs",  # папка для результатов
    name="fruits_model",
    val=True,
    verbose=True
)


test_image = "fruits_yolo/images/val/2a0ff8ca86f4c7955b66ebf42bbc045c.jpg"
results = model(test_image)
result = results[0]


plt.imshow(result.plot()[:, :, ::-1])
plt.axis('off')
plt.show()


def draw_bboxes(image, results, conf_thresh=0.7):
    boxes = results[0].boxes.cpu()
    class_names = results[0].names
    for box in boxes:
        if box.conf < conf_thresh:
            continue
        x1, y1, x2, y2 = box.xyxy[0].numpy()
        class_idx = int(box.cls)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        cv2.putText(image, class_names[class_idx], (int(x1), int(y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    return image

 
img = cv2.imread(test_image)
annotated_img = draw_bboxes(img.copy(), results)
cv2.imshow("YOLOv8 Fruits", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
