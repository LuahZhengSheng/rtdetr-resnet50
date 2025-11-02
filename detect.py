import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR(r"C:\Users\60174\Downloads\best.pt")
    model.predict(source=r"C:\Users\60174\OneDrive\FYP\dataset\selected_category_dataset\images\test\7_Types_Plastic_test_1893_IMG_2826_jpg.rf.cd3d0f143c65b141c21658b2ab0142cf.jpg",
                  imgsz=640,
                  device='cpu',
                  save=True
                  )

