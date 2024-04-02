from ultralytics import YOLO
model=YOLO('ch.pt')
pred=model.predict(source="0", show=True,conf=0.7)
