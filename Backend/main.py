import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from Backend.detector import SurakshaNetDetector
from Backend.compliance import classify_compliance
from Backend.db import ViolationLogger
from Backend.notifier import EmailNotifier

app = FastAPI(title="SurakshaNet AI Core Backend", version="1.1")
detector = SurakshaNetDetector()
logger = ViolationLogger()
notifier = EmailNotifier(cooldown_seconds=60.0)  # configure via env vars


@app.get("/")
def home():
    return {"status": "online", "project": "SurakshaNet Engine"}


@app.post("/detect")
async def detect_violations(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    raw_detections = detector.detect(frame, conf_threshold=0.5)
    results = classify_compliance(raw_detections)

    violations_found = [
        {
            "issue": f"Missing PPE Gear - Detected {r['class']}",
            "confidence": round(r["confidence"], 2),
            "bbox": r["bbox"]
        }
        for r in results if r["status"] == "BREACH"
    ]

    if violations_found:
        # Persist every breach to SQLite
        logger.log_batch(results)

        # Fire an email alert with the snapshot (throttled internally)
        notifier.notify(frame, [
            {"class": v["issue"].split("Detected ")[-1], "confidence": v["confidence"], "bbox": v["bbox"]}
            for v in violations_found
        ])

    return {
        "is_compliant": len(violations_found) == 0,
        "total_violations": len(violations_found),
        "violations": violations_found
    }


@app.get("/logs/recent")
def recent_logs(limit: int = 50):
    """Returns the most recent violation records for the dashboard."""
    rows = logger.get_recent(limit=limit)
    return {
        "count": len(rows),
        "logs": [
            {
                "timestamp": r[0],
                "class": r[1],
                "confidence": r[2],
                "bbox": [r[3], r[4], r[5], r[6]]
            }
            for r in rows
        ]
    }


@app.get("/logs/summary")
def logs_summary():
    """Returns aggregate violation stats for dashboard charts."""
    return logger.get_summary()
