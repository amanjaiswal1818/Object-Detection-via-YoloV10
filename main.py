import cv2
from ultralytics import YOLO

# Load YOLOv10n model (make sure you have yolov10n.pt downloaded)
model = YOLO("yolov10n.pt")  # You can change this to yolov10s.pt, yolov10m.pt, etc. if you want

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv10 object detection
    results = model(frame, conf=0.3)[0]

    # Draw results
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
        cls = int(box.cls[0])                  # Class ID
        conf = float(box.conf[0])              # Confidence
        label = model.names[cls]               # Class name

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show result
    cv2.imshow("YOLOv10 Object Detection", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
