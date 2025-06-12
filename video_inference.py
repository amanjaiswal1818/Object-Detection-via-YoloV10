import cv2
from ultralytics import YOLO

# Load YOLOv10n model (make sure you have yolov10n.pt downloaded)
model = YOLO("yolov10n.pt")  # Change to your model if needed

# Video input path - Use raw string to avoid path issues on Windows
input_path = r"C:\Users\Aman\Desktop\Projects\Source files\Object detection\test_vid.mp4"
cap = cv2.VideoCapture(input_path)

# Get video properties for saving output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define output video writer - saved in current folder as output.mp4
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv10 object detection
    results = model(frame, conf=0.3)[0]

    # Draw results
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        cls = int(box.cls[0])                    # Class ID
        conf = float(box.conf[0])                # Confidence score
        label = model.names[cls]                 # Class name

        # Draw rectangle and put label with confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Write the frame with detections to output video
    out.write(frame)

    # Show the frame
    cv2.imshow("YOLOv10 Video Inference", frame)
    if cv2.waitKey(1) == 27:  # Press ESC to quit
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
