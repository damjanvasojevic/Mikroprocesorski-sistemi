from sahi import AutoDetectionModel
from sahi.utils.yolov8 import download_yolov8s_model
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

# Download YOLOv8 model
yolov8_model_path = "models/yolov8s.pt"
download_yolov8s_model(yolov8_model_path)

def download_image_from_url(url, save_path):
    download_from_url(url, save_path)

def read_local_image(image_path):
    return read_image(image_path)

# Load a pretrained YOLO model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.4,
    device="cpu"  # or 'cuda:0'
)

# Process the local image
local_image_path = './localImage/slika.jpeg'  # Putanja do lokalne slike
local_image = read_local_image(local_image_path)
result_image = get_prediction(local_image, detection_model)

# Sačuvaj vizuelne rezultate detekcije
result_image.export_visuals(export_dir="C:/Users/urosm/Desktop/Jevrem MIKRO/prediction_visual_image_local")
# Prikaz vizuelnih rezultata
Image("C:/Users/urosm/Desktop/Jevrem MIKRO/prediction_visual_image_local.png")

# Process the sliced image
result_sliced = get_sliced_prediction(
    local_image,
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

# Pristupi listi predikcija objekata
object_prediction_list = result_sliced.object_prediction_list

# Konvertuj u COCO anotacije, COCO predikcije, imantics i fiftyone formate
result_sliced.to_coco_annotations()[:3]
result_sliced.to_coco_predictions(image_id=1)[:3]
result_sliced.to_imantics_annotations()[:3]
result_sliced.to_fiftyone_detections()[:3]

# Predikcija na skupu slika
predict(
    model_type="yolov8",
    model_path="./yolov8n.pt",
    model_device="cpu",
    model_confidence_threshold=0.70,
    source="./localImage/slika.jpeg",
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
