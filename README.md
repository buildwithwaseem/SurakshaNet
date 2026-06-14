# 🛡️ SurakshaNet — AI Site Safety & PPE Compliance Command Center

SurakshaNet is an automated AI-powered safety monitoring system designed for construction sites. It uses a fine-tuned **YOLOv8-Nano** model to detect Personal Protective Equipment (PPE) compliance in real time, identify safety violations (missing helmets), and automatically alert site supervisors through audio sirens, persistent logs, and email notifications.

---

## 🖼️ Screenshots

> Add your project screenshots below. Place image files inside the `images/` folder and update the paths accordingly.

### Live Detection — Breach Detected (No Helmet)
<!-- ![Breach Detected](images/breach_detected.png) -->

### Live Detection — Safe (Helmet Detected)
<!-- ![Safe Detection](images/safe_detection.png) -->

### Streamlit Dashboard — Overview
<!-- ![Dashboard Overview](images/dashboard_overview.png) -->

### Streamlit Dashboard — Violation Trends
<!-- ![Violation Trends](images/dashboard_trends.png) -->

### Email Alert Sample
<!-- ![Email Alert](images/email_alert.png) -->

---

## 📌 Project Overview

The system continuously monitors a video feed (webcam or uploaded image) and classifies every detected person into one of three categories:

| Class    | Meaning                                  |
|----------|-------------------------------------------|
| `Head`   | A bare head detected — no helmet present  |
| `Helmet` | A safety helmet detected                  |
| `Person` | A full person detected without PPE        |

If a `Head` or `Person` is detected **without** an overlapping `Helmet` bounding box, the system flags it as a **Safety Breach** and triggers the alert pipeline (siren, database log, and email notification).

---

## 🏗️ System Architecture

```
                    ┌─────────────────────┐
                    │   Webcam / Image     │
                    │   Input Source       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  detector.py         │
                    │  (ONNX Runtime       │
                    │   YOLOv8-Nano)       │
                    └──────────┬───────────┘
                               │  raw detections
                               ▼
                    ┌─────────────────────┐
                    │  compliance.py       │
                    │  (IoU-based SAFE /   │
                    │   BREACH logic)      │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │  alerts.py    │ │   db.py       │ │ notifier.py   │
      │  (Audio Siren)│ │  (SQLite Log) │ │ (Email Alert) │
      └──────────────┘ └──────────────┘ └──────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  main.py (FastAPI)   │
                    │  REST API Layer      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  dashboard/app.py    │
                    │  (Streamlit UI)      │
                    └─────────────────────┘
```

---

## 📂 Project Structure

```
SurakshaNet/
├── Backend/
│   ├── __init__.py
│   ├── alerts.py          # Audio siren alert module (pyttsx3-based)
│   ├── compliance.py       # IoU-based SAFE/BREACH classification logic
│   ├── db.py                # SQLite violation logging module
│   ├── detector.py          # ONNX Runtime inference pipeline (YOLOv8-Nano)
│   ├── main.py              # FastAPI backend server & REST API
│   └── notifier.py          # Email notification module (SMTP)
├── dashboard/
│   └── app.py               # Streamlit analytics dashboard
├── images/                  # Sample images / test assets
├── models/
│   └── weights/
│       ├── best.onnx        # Optimized ONNX model for CPU inference
│       └── best.pt           # Original PyTorch model weights
├── .env                      # Environment variables (credentials, not committed)
├── .gitignore
├── surakshanet_logs.db       # SQLite database (auto-generated)
├── test_webcam.py             # Live webcam testing pipeline
└── README.md
```

---

## ⚙️ Core Components

### 1. Detection Engine — `Backend/detector.py`
- Loads the fine-tuned **YOLOv8-Nano** model exported to **ONNX** format (`best.onnx`) for lightweight, CPU-friendly inference.
- Uses `onnxruntime` with `intra_op_num_threads=2` to throttle CPU usage and prevent the host machine from freezing during continuous inference.
- Performs preprocessing (resize, normalize, transpose), runs inference, and applies Non-Maximum Suppression (NMS) to filter overlapping boxes.
- Returns structured detection results: class label, confidence score, and bounding box coordinates.

### 2. Compliance Logic — `Backend/compliance.py`
- Single source of truth for converting raw detections into compliance statuses.
- Calculates **Intersection over Union (IoU)** between `Head`/`Person` boxes and `Helmet` boxes.
- Classification rules:
  - **SAFE** — A `Head`/`Person` box sufficiently overlaps a `Helmet` box.
  - **BREACH** — A `Head`/`Person` box has no overlapping helmet → triggers alerts.
  - **EQUIPMENT** — A standalone `Helmet` detection (informational only).

### 3. Alert System — `Backend/alerts.py`
- Plays a spoken audio warning using `pyttsx3` (text-to-speech) when a breach is detected.
- Falls back to a console beep if `pyttsx3` is unavailable.
- Implements a **cooldown timer** (default: 5 seconds) to prevent continuous alert spam.
- Runs asynchronously on a background thread to avoid blocking the video pipeline.

### 4. Persistent Logging — `Backend/db.py`
- Logs every confirmed breach into a local **SQLite** database (`surakshanet_logs.db`).
- Records: timestamp, detected class, confidence score, and bounding box coordinates.
- Provides query methods for recent violations and aggregate statistics (used by the dashboard).

### 5. Email Notifications — `Backend/notifier.py`
- Sends an automated email alert with a **snapshot image** attached when a breach is detected.
- Configured via environment variables for security (no hardcoded credentials).
- Implements a **cooldown timer** (default: 60 seconds) to avoid flooding inboxes.
- Runs asynchronously on a background thread.

### 6. REST API Backend — `Backend/main.py`
- Built with **FastAPI**, exposing a stateless `/detect` endpoint that accepts raw image uploads.
- Returns a structured JSON compliance report (violations, confidence scores, bounding boxes).
- Additional endpoints for dashboard integration:
  - `GET /logs/recent` — fetches the most recent violation records.
  - `GET /logs/summary` — returns aggregate violation counts by class.

### 7. Analytics Dashboard — `dashboard/app.py`
- Built with **Streamlit**, providing a real-time monitoring interface.
- Displays live violation feed, summary metrics, and interactive **Plotly** trend charts.
- Connects to the FastAPI backend over HTTP for live data streaming.

### 8. Live Webcam Pipeline — `test_webcam.py`
- Standalone script for real-time webcam-based monitoring.
- Applies **frame skipping** (processes every 3rd frame) to reduce CPU load by approximately 66%.
- Draws color-coded bounding boxes:
  - 🔴 **Red** — Breach (no helmet detected)
  - 🟢 **Green** — Safe (helmet detected and overlapping)
  - 🔵 **Cyan** — Equipment (standalone helmet detection)
- Triggers the siren, database logger, and email notifier on every breach frame.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A working webcam (for live testing)

### Installation

1. Clone the repository and navigate to the project folder.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate     # Windows
   source venv/bin/activate  # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Environment Configuration

Create a `.env` file in the project root with the following variables for email alerts:

```env
SURAKSHANET_SENDER_EMAIL=your_email@gmail.com
SURAKSHANET_SENDER_PASSWORD=your_app_password
SURAKSHANET_RECIPIENTS=supervisor1@example.com,supervisor2@example.com
```

> **Note:** For Gmail, enable 2-Factor Authentication and generate an **App Password** — standard account passwords will not work with SMTP.

---

## ▶️ Running the Application

### 1. Live Webcam Monitoring
```bash
python test_webcam.py
```
Press `q` to close the live preview window.

### 2. Start the FastAPI Backend
```bash
uvicorn Backend.main:app --reload --port 8000
```
API documentation will be available at `http://localhost:8000/docs`.

### 3. Launch the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Dashboard will be available at `http://localhost:8501`.

---

## 🔌 API Reference

| Endpoint        | Method | Description                                      |
|------------------|--------|---------------------------------------------------|
| `/`              | GET    | Health check — returns service status            |
| `/detect`        | POST   | Accepts an image file, returns compliance report |
| `/logs/recent`   | GET    | Returns the most recent violation records         |
| `/logs/summary`  | GET    | Returns aggregate violation counts by class       |

### Example `/detect` Response
```json
{
  "is_compliant": false,
  "total_violations": 1,
  "violations": [
    {
      "issue": "Missing PPE Gear - Detected Head",
      "confidence": 0.82,
      "bbox": [420, 195, 610, 430]
    }
  ]
}
```

---

## 🧠 Model Details

- **Architecture:** YOLOv8-Nano (fine-tuned on a custom construction site dataset)
- **Export Format:** ONNX (optimized for CPU inference)
- **Classes:** `Head`, `Helmet`, `Person`
- **Inference Engine:** `onnxruntime` with sequential execution mode and limited thread count for stability on consumer hardware

---

## 🛣️ Future Roadmap

- [ ] Multi-camera support for monitoring multiple site zones simultaneously
- [ ] Role-based dashboard access for site supervisors and administrators
- [ ] Cloud-based log storage and historical trend analysis
- [ ] Integration with additional PPE classes (safety vests, gloves, goggles)
- [ ] Mobile app companion for instant push notifications

---

## 📄 License

This project was developed as part of an academic sprint project. Licensing terms to be determined by the project owner.

---

## 🙏 Acknowledgements

- **YOLOv8** by Ultralytics
- **ONNX Runtime** for optimized inference
- **FastAPI** and **Streamlit** for backend and dashboard frameworks
