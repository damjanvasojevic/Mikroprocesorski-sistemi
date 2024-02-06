import random
import cv2
from ultralytics import YOLO

# Otvaramo datoteku u režimu čitanja
with open("utils/coco.txt", "r") as my_file:
    # Čitamo sadržaj datoteke
    class_list = my_file.read().splitlines()

# Generisanje slučajnih boja za svaku klasu
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in class_list]

# Učitavanje prethodno obučenog YOLOv8n modela
model = YOLO("./yolov8n.pt", "v8")

# Veličina okvira za video
frame_width = 640
frame_height = 480

# Otvaranje video snimka
cap = cv2.VideoCapture("./video/low_resolution_highway.mp4")

while True:
    # Čitanje frejma iz video snimka
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Predikcija na frejmu
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    # Prikaz detekcije na frejmu
    for box in detect_params[0].boxes:
        clsID = box.cls.numpy()[0]
        conf = box.conf.numpy()[0]
        bb = box.xyxy.numpy()[0]

        # Crtanje pravougaonika oko detektovanih objekata
        cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), detection_colors[int(clsID)], 3)

        # Prikaz klase i sigurnosti
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame, f"{class_list[int(clsID)]} {round(conf * 100, 2)}%", (int(bb[0]), int(bb[1]) - 10),
                    font, 1, (255, 255, 255), 2)

    # Prikaz frejma sa detekcijom
    cv2.imshow("ObjectDetection", frame)

    # Izlaz iz petlje ako je pritisnuta tipka Q
    if cv2.waitKey(1) == ord("q"):
        break

# Oslobađanje resursa i zatvaranje prozora
cap.release()
cv2.destroyAllWindows()
