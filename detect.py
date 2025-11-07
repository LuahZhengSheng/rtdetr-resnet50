import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR(r"C:\Users\60174\Downloads\best2.pt")
    model.predict(source=r"C:\Users\60174\Downloads\images (4).jpg",
                  imgsz=640,
                  device='cpu',
                  save=True
                  )

