from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import json
import uvicorn
from io import BytesIO

app = FastAPI()

@app.get("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("index.html") as file:
        return file.read()


@app.post("/detect")
async def detect(image_file: UploadFile = File(...)):
    """
    Handler of /detect POST endpoint
    Receives uploaded file with a name "image_file",
    passes it through YOLOv8 object detection
    network and returns an array of bounding boxes.
    :return: a JSON array of objects bounding
    boxes in format
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    buf = await image_file.read()
    boxes = detect_objects_on_image(Image.open(BytesIO(buf)))
    
    return boxes
    


def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param buf: Input image file stream
    :return: Array of bounding boxes in format
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    model = YOLO("\\Users\\ausawini\\Desktop\\SuperAI-SS3\\fastapi\\model\\best.pt")
    results = model.predict(buf)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
            x1, y1, x2, y2, result.names[class_id], prob
            #result.names[class_id], prob
        ])
        
    
    return output



if __name__ == "__main__":
   uvicorn.run(app, host="127.0.0.1", port=5500)