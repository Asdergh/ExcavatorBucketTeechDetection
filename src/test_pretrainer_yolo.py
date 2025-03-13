from ultralytics import YOLO


model = YOLO("yolo11n.pt")
results = model.train(data="C:\\Users\\1\\Downloads\\tooth excava.v2i.coco\\train\\_annotations.coco.json", epochs=10, imgsz=640)

# Run inference with the YOLO12n model on the 'bus.jpg' image
results = model()