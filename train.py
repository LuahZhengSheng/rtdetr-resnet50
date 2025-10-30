import warnings

from ultralytics import RTDETR

warnings.filterwarnings("ignore")

model = RTDETR(r"ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml")
# model.load('rtdetr-l.pt') # 是否加载预训练权重
model.train(
    data="data.yaml",  # 训练参数均可以重新设置
    epochs=100,
    imgsz=640,
    workers=4,
    batch=16,
    device="cpu",
    optimizer="AdamW",
    amp=False,
)
