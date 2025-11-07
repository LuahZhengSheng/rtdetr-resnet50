from ultralytics import RTDETR

model = RTDETR("runs/detect/train/weights/last.pt")
results = model.train(resume=True)