import cv2
from PIL import Image
import requests
import numpy as np
from io import BytesIO
from huggingface_hub import hf_hub_download
from easy_ViTPose import VitInference

MODEL_SIZE = 'b'
DATASET = 'apt36k'
MODEL_TYPE = "torch"
REPO_ID = 'JunkyByte/easy_ViTPose'
FILENAME = f'{MODEL_TYPE}/{DATASET}/vitpose-{MODEL_SIZE}-{DATASET}.pth'
MODEL_URL = f'https://huggingface.co/{REPO_ID}/resolve/main/{FILENAME}?raw=true'

print(f'Downloading model {REPO_ID}/{FILENAME}')
response = requests.get(MODEL_URL)
if response.status_code == 200:
    with open('vitpose-b-wholebody.pth', 'wb') as file:
        file.write(response.content)

# Load YOLO model
FILENAME_YOLO = 'yolov8/yolov8s.pt'
yolo_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_YOLO)

# Load ViT model
model = VitInference('vitpose-b-wholebody.pth', yolo_path, MODEL_SIZE,
                     dataset=DATASET, yolo_size=320, is_video=False)

# Run inference on example image
#url = 'https://i.ibb.co/gVQpNqF/imggolf.jpg'
#img = np.array(Image.open(BytesIO(requests.get(url).content)), dtype=np.uint8)

# Load and process the local image
image_path = 'test_kapi1.png'  # Path to your local image
img = cv2.imread(image_path)
img = cv2.resize(img, (1024, 683))  # Resize to 1024x683

frame_keypoints = model.inference(img)
img = model.draw(show_yolo=True)
cv2.imshow('Inference', img[..., ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
