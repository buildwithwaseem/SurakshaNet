# 🛡️ SurakshaNet: AI Site Safety & PPE Compliance Command Center

SurakshaNet is an automated, real-time computer vision system designed to monitor industrial and construction sites for safety violations, specifically detecting whether personnel are wearing mandatory Personal Protective Equipment (PPE) like safety helmets. 

By leveraging an optimized edge-computing architecture, the system operates locally with low CPU latency and streams live compliance metrics to an enterprise-grade analytics dashboard, complete with automated sirens, local persistent logging, and instant automated email breach alerts.

---

## 🚀 Key Features

* **Real-Time Local Inference:** Powered by an optimized, quantized **YOLOv8-Nano model converted to ONNX format (`best.onnx`)** to allow seamless local CPU testing without host throttling or UI freezing.
* **Gated Compliance Logic:** Employs an Intersection-over-Union (IoU) box overlap evaluation matrix to prevent false safety alerts (e.g., bare heads misclassified as helmets, or bike helmets bypassing site criteria).
* **Decoupled Microservice Architecture:** Implements a dual-engine architecture consisting of a **FastAPI backend** (exposing high-throughput stateless endpoints) and a **Streamlit front-end console**.
* **Modular Alert Pipeline:**
    * 🔊 **Audio Siren:** Non-blocking background thread-driven audio warning (`pyttsx3`) with a deliberate frame cooldown.
    * 📁 **Persistent SQLite Auditing:** Structural SQL logging of every breach timestamp, confidence index, and bounding-box coordinates.
    * 📧 **Secure Email Forwarding:** Automated SMTP transmission of breach snapshots directly to safety supervisors, throttled via a 60-second cool-down routine to avoid spamming.

---

## 📐 System Architecture Diagram

```text
               +-------------------------------------------------+
               |             Edge Camera Feed (Webcam)           |
               +-----------------------+-------------------------+
                                       |
                                       v
               +-------------------------------------------------+
               |         SurakshaNet ONNX Detector Engine        |
               |       (Class Mapping: Head, Helmet, Person)     |
               +-----------------------+-------------------------+
                                       |
                                       v
               +-------------------------------------------------+
               |       Compliance Rule Validator (IoU Check)     |
               +----------+------------+------------+------------+
                          |            |            |
            IF BREACH     v            v            v 
        +-----------------+    +-------+----+  +----+-----------+
        | Audio Siren     |    | SQLite Log |  | SMTP Email App |
        | (Thread-Pooled) |    | Audits     |  | Snapshot Alert |
        +-----------------+    +-------+----+  +----------------+
                                       |
                                       v
               +-------------------------------------------------+
               |          FastAPI Operational Backend            |
               |          (Routes: /detect, /logs/*)             |
               +-----------------------+-------------------------+
                                       | (HTTP Polling)
                                       v
               +-------------------------------------------------+
               |         Streamlit Real-Time Dashboard           |
               |       (Plotly Charts & Live Violation Feed)     |
               +-------------------------------------------------+



-------------------
📂 Repository Structure
SurakshaNet/
│
├── .gitignore             # Root level rule matrix to protect secrets/environments
├── .env                   # Local configuration variables (DO NOT COMMIT TO GITHUB)
├── requirements.txt       # Python framework dependencies
├── test_webcam.py         # Local webcam tracker pipeline script
├── main.py                # FastAPI microservice operational hub
│
├── Backend/               # Core system modules
│   ├── detector.py        # YOLOv8 ONNX runtime abstract processor
│   ├── compliance.py      # IoU bounding-box compliance rule matrix
│   ├── alerts.py          # Background audio siren engine
│   ├── db.py              # SQLite engine router
│   └── notifier.py        # Automated SMTP mail scheduler
│
└── dashboard/             # Frontend UI layer
    └── app.py             # Streamlit live charting frontend UI

---------------------

















