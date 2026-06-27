import cv2
from Backend.detector import SurakshaNetDetector
from Backend.compliance import classify_compliance
from Backend.alerts import SirenAlert
from Backend.db import ViolationLogger
from Backend.notifier import EmailNotifier


def run_live_pipeline():
    # 1. Initialize the ONNX model (threads=2 keeps the laptop cooler)
    detector = SurakshaNetDetector()

    # 2. Initialize support modules
    siren = SirenAlert(cooldown_seconds=5.0)
    logger = ViolationLogger()
    notifier = EmailNotifier(cooldown_seconds=60.0)  # configure via env vars

    # 3. Open the laptop webcam (0 means the default webcam)
    cap = cv2.VideoCapture(0)

    frame_count = 0
    cached_results = []

    print("SurakshaNet Live Tracker Active! Press 'q' to close it.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("ERROR: The camera feed is not available.")
            break

        frame_count += 1

        # HARDWARE OPTIMIZATION (Frame Skipping): Run the AI algorithm only on every 3rd frame.
        if frame_count % 3 == 0:
            detections = detector.detect(frame, conf_threshold=0.5)
            cached_results = classify_compliance(detections)

            breaches = [r for r in cached_results if r["status"] == "BREACH"]
            if breaches:
                # 1. Audio siren (throttled internally)
                siren.trigger()

                # 2. Persist to SQLite (every breach frame logged)
                logger.log_batch(cached_results)

                # 3. Email alert with snapshot (throttled internally)
                violations_payload = [
                    {"class": r["class"], "confidence": r["confidence"], "bbox": r["bbox"]}
                    for r in breaches
                ]
                notifier.notify(frame, violations_payload)

        # 4. UI rendering: draw boxes around detected objects
        for r in cached_results:
            x1, y1, x2, y2 = r["bbox"]
            label = f"{r['class']} ({r['confidence']:.2f})"

            if r["status"] == "BREACH":
                color = (0, 0, 255)    # Red
            elif r["status"] == "SAFE":
                color = (0, 255, 0)    # Green
            else:  # EQUIPMENT
                color = (255, 200, 0)  # Cyan-ish, neutral

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 5. Show the output in the display window
        cv2.imshow("SurakshaNet Engine - Live Preview", frame)

        # Keyboard handling: press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_pipeline()
