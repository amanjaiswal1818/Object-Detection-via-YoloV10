import cv2
from ultralytics import YOLO

# Load YOLOv10n model (make sure you have yolov10n.pt downloaded)
model = YOLO("yolov10n.pt")  # Change to your model if needed

# Image input path - use raw string for Windows paths
input_path = r"C:\Users\Aman\Desktop\Projects\Source files\Object detection\test_image.jpg"

# Read the image
image = cv2.imread(input_path)

if image is None:
    print("Error: Image not found or path is incorrect")
    exit()

# Run YOLOv10 object detection
results = model(image, conf=0.3)[0]

# Draw results on the image
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
    cls = int(box.cls[0])                    # Class ID
    conf = float(box.conf[0])                # Confidence score
    label = model.names[cls]                 # Class name

    # Draw rectangle and label with confidence
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Show the image with detections
cv2.imshow("YOLOv10 Image Inference", image)

# Save output image in the same folder
output_path = "output_image.jpg"
cv2.imwrite(output_path, image)
print(f"Output saved to {output_path}")

cv2.waitKey(0)
cv2.destroyAllWindows()
