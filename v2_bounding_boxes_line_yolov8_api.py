from PIL import Image, ImageDraw, ImageFont
from linebot.exceptions import InvalidSignatureError
from ultralytics import YOLO
from fastapi import FastAPI, Request, Response
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, ImageMessage, TextMessage, TextSendMessage, ImageSendMessage
from io import BytesIO
import uvicorn
import base64
import os

from imgurpython import ImgurClient
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

# Function to handle image messages
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_id = event.message.id
    message_content = line_bot_api.get_message_content(message_id)

    # Read the image data from the message content
    image_data = BytesIO(message_content.content)

    # Perform object detection on the image
    boxes = detect_objects_on_image(image_data)

    # Annotate the image with bounding boxes
    annotated_image = annotate_image(image_data, boxes)

    # Send the annotated image back to the user
    send_annotated_image(event.reply_token, annotated_image)


def detect_objects_on_image(image_data):
    """
    Function receives image data,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    """
    model = YOLO("\\Users\\ausawini\\Desktop\\SuperAI-SS3\\fastapi\\model\\best.pt")
    image = Image.open(image_data)
    results = model.predict(image)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([x1, y1, x2, y2, result.names[class_id], prob])
    return output


def annotate_image(image_data, boxes):
    """
    Function annotates the image with bounding boxes.
    """
    # Specify the font file path
    font_path = "\\Users\\ausawini\\Desktop\\SuperAI-SS3\\fastapi\\font_path\\arial.ttf"

    font = ImageFont.truetype(font_path, 35)

    image = Image.open(image_data)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2, object_type, probability = box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=8)
        draw.text((x1, y1 - 10), f"{object_type} ({probability})",  outline="red", width=8,fill="red",font=font)
    return image



def send_annotated_image(reply_token, image):
    """
    Function sends the annotated image back to the user.
    :param reply_token: Reply token
    :param image: Annotated image
    """
    output_directory = "\\Users\\ausawini\\Desktop\\SuperAI-SS3\\fastapi\\image"  # Replace with the directory where you want to save the image
    os.makedirs(output_directory, exist_ok=True)
    image_path = os.path.join(output_directory, "annotated_image.png")
    image.save(image_path, format="PNG")

    # Upload the annotated image to Imgur
    client_id = ' '  # Replace with your Imgur client ID
    uploaded_image_url = upload_image_to_imgur(image_path, client_id)

    # Check if image upload was successful
    if uploaded_image_url:
        # Generate the URL for the annotated image
        annotated_image_url = uploaded_image_url

        line_bot_api.reply_message(
            reply_token,
            ImageSendMessage(
                original_content_url=annotated_image_url,
                preview_image_url=annotated_image_url
            )
        )
    else:
        print("Failed to upload annotated image to Imgur.")

# Rest of the code...


## work ##
import os
from imgurpython import ImgurClient

def upload_image_to_imgur(image_path, client_id):
    """
    Uploads an image to Imgur and returns the public URL of the uploaded image.
    :param image_path: The local path of the image file to upload.
    :param client_id: Your Imgur client ID.
    :return: The public URL of the uploaded image.
    """
    client = ImgurClient(client_id, None)
    image = client.upload_from_path(image_path, anon=True)
    return image['link']

image_path = '\\Users\\ausawini\\Desktop\\SuperAI-SS3\\fastapi\\image\\annotated_image.png'
client_id = ' '  # Replace with your Imgur client ID

uploaded_image_url = upload_image_to_imgur(image_path, client_id)
if uploaded_image_url:
    print(f"Image uploaded successfully. Public URL: {uploaded_image_url}")



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

