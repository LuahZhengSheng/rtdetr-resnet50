from ultralytics import RTDETR
import warnings
warnings.filterwarnings('ignore')

model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml')
results = model.train(
    data='data.yaml',
    epochs=150,
    imgsz=640,
    batch=8,
    device=0,
    workers=4,
    optimizer='AdamW',
    lr0=0.001,
    amp=False,
    deterministic=False,
    save=True,
    save_period=10,
    exist_ok=True,
    plots=True,
    pretrained=False
)