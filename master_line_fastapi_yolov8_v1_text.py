from fastapi import FastAPI
from PIL import Image
from io import BytesIO
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextMessage, TextSendMessage
import uvicorn
from ultralytics import YOLO
from fastapi import Request, Response
from PIL import ImageDraw
from linebot.models import ImageSendMessage
from linebot.models import MessageEvent, ImageMessage
app = FastAPI()

# Initialize Line bot API and webhook handler
line_bot_api = LineBotApi("") #YOUR_CHANNEL_ACCESS_TOKEN
handler = WebhookHandler("") #YOUR_CHANNEL_SECRET


@app.post("/webhook")
async def webhook(request: Request):
    signature = request.headers["X-Line-Signature"]
    body = await request.body()
    try:
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        return Response(status_code=400)
    return Response(status_code=200)


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event: MessageEvent):
    text = event.message.text
    if text.lower() == "detect":
        reply_text = "Please upload an image for detection."
    else:
        reply_text = "Invalid command. Please enter 'detect' to start object detection."
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )


@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event: MessageEvent):
    message_id = event.message.id
    message_content = line_bot_api.get_message_content(message_id)
    image_data = BytesIO(message_content.content)
    image = Image.open(image_data)

    # Perform object detection on the image
    boxes = detect_objects_on_image(image)

    # Convert the results to a text message
    reply_text = format_detection_results(boxes)
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )



def detect_objects_on_image(image):
    """
    Function receives an image,
    passes it through YOLOv5 object detection
    network and returns an array of bounding boxes.
    :param image: Input PIL image object
    :return: Array of bounding boxes in format
    [[x1, y1, x2, y2, object_type, probability], ...]
    """
    model = YOLO("\\Users\\ausawini\\Desktop\\SuperAI-SS3\\fastapi\\model\\best.pt")  # Replace with your YOLO model path
    results = model.predict(image)
    output = []
    #for box in results.xyxy[0]:
     #   x1, y1, x2, y2, conf, class_id = box.tolist()
      #  object_type = model.names[int(class_id)]
       # probability = round(conf * 100, 2)
        #output.append([x1, y1, x2, y2, object_type, probability])

    #return output

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            class_id = box.cls[0].item()
            prob = round(box.conf[0].item(), 2)
            output.append([
                x1, y1, x2, y2, result.names[class_id], prob
            ])

    return output
   
    
def format_detection_results(boxes):
    """
    Formats the object detection results into a string message.
    :param boxes: Array of bounding boxes in format
    [[x1, y1, x2, y2, object_type, probability], ...]
    :return: String message of the detection results
    """
    if not boxes:
        return "No objects detected."

    result_str = "Object detection results:\n"
    for box in boxes:
        x1, y1, x2, y2, object_type, probability = box
        result_str += f"- {object_type}: {probability}%\n"
        #result_str += f"- {object_type}: {x1} {y1} {x2} {y2} {probability}%\n"
    return result_str



    return image


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
