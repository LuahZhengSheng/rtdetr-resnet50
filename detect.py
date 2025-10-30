import warnings

warnings.filterwarnings("ignore")
from ultralytics import RTDETR

if __name__ == "__main__":
    model = RTDETR("runs/train/exp/weights/best.pt")
    model.predict(source="images", imgsz=640, device="0", save=True)
