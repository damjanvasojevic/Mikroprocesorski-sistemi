from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.utils.yolov8 import download_yolov8s_model
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from IPython.display import Image

# Download YOLOv8 model
yolov8_model_path = "models/yolov8s.pt"
download_yolov8s_model(yolov8_model_path)

def download_image_from_url(url, save_path):
    download_from_url(url, save_path)

def read_online_image(image_url):
    image_path = './onlineImage/online_temp.jpeg'
    download_image_from_url(image_url, image_path)
    return read_image(image_path)

# Load a pretrained YOLO model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.4,
    device="cpu"  # or 'cuda:0'
)

# Process the online image
online_image_url = 'https://www.realsimple.com/thmb/sESQaPTQu1cw7w5G8omCg_2Ghsk=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/table-setting-GettyImages-1054102340-0f81beae51704cc5b3e2e30e383ff37f.jpg'
online_image = read_online_image(online_image_url)
result_online_image = get_prediction(online_image, detection_model)
result_online_image.export_visuals(export_dir="prediction_visual_image_online")
Image("prediction_visual_image_online.png")

# Example of using the predict function (modify with actual parameters)
predict(
    model_type="yolov8",
    model_path="./yolov8n.pt",
    model_device="cpu",
    model_confidence_threshold=0.4,
    source="./onlineImage/online_temp.jpeg",
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
