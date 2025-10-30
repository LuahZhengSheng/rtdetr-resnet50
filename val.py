from ultralytics import RTDETR

if __name__ == "__main__":
    model = RTDETR(model="runs/train/exp/weights/best.pt")
    model.val(data="data.yaml", batch=32, device="0", imgsz=640, workers=8, split="test")
