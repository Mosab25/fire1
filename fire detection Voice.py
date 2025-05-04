import cv2
from ultralytics import YOLO
import time
import pyttsx3

# Initialize text-to-speech engine with male voice
engine = pyttsx3.init()
voices = engine.getProperty('voices')
# Try to set a male voice (availability depends on your system)
for voice in voices:
    if 'male' in voice.name.lower():
        engine.setProperty('voice', voice.id)
        male_voice_found = True
        break

# Load your trained fire detection model
model = YOLO(r"best.pt")  # Update path as needed

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Set resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Variables for FPS calculation and voice alert cooldown
prev_time = 0
fps = 0
high_conf_detection = False
last_detection_time = 0
last_alert_time = 0
alert_cooldown = 5  # seconds between voice alerts

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection with minimum 80% confidence
    results = model(frame, conf=0.8, verbose=False)
    
    # Check for high-confidence detections
    high_conf_detection = False
    current_time = time.time()
    
    for result in results:
        for box in result.boxes:
            if box.conf.item() > 0.8:  # Confidence > 80%
                high_conf_detection = True
                last_detection_time = current_time
                # Trigger voice alert if not in cooldown
                if current_time - last_alert_time > alert_cooldown:
                    engine.say("Warning! Fire detected. Please evacuate immediately.")
                    engine.runAndWait()
                    last_alert_time = current_time
                break
    
    # Visual feedback for high confidence detections
    if high_conf_detection:
        # Add red border when fire detected
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
        # Display warning text
        cv2.putText(frame, "FIRE DETECTED!", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Draw all detections (will include those >80% confidence)
    annotated_frame = results[0].plot()
    
    # Calculate and display FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display confidence threshold
    cv2.putText(annotated_frame, "Confidence >80%", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show the processed frame
    cv2.imshow("Fire Detection (80% Confidence Threshold)", annotated_frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()