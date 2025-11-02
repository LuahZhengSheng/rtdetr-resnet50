from ultralytics import RTDETR
import warnings
warnings.filterwarnings('ignore')

# model = RTDETR(r"ultralytics/cfg/models/rt-detr/rtdetr-x.yaml")
# model.train(data='data.yaml',
#                         epochs=100,
#                         imgsz=640,
#                         workers=4,
#                         batch=16,
#                         device=0,
#                         optimizer='AdamW',
#                         amp=False,
#                         )

model = RTDETR('runs/detect/train/weights/last.pt')
results = model.train(resume=True)