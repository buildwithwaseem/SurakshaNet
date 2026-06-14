"""
SurakshaNet - Email Notification Module
Sends a snapshot image + violation details to safety supervisors via SMTP
when a breach occurs. Throttled with a cooldown to avoid spamming.

NOTE: For Gmail, use an "App Password" (not your normal password) and
enable 2FA on the account first. Never hardcode credentials -- use env vars.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import smtplib
import threading
import time
import cv2
from email.message import EmailMessage
from datetime import datetime


class EmailNotifier:
    def __init__(
        self,
        sender_email: str = None,
        sender_password: str = None,
        recipient_emails: list = None,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        cooldown_seconds: float = 60.0,
    ):
        # Prefer environment variables over hardcoded values
        self.sender_email = sender_email or os.environ.get("SURAKSHANET_SENDER_EMAIL")
        self.sender_password = sender_password or os.environ.get("SURAKSHANET_SENDER_PASSWORD")
        self.recipient_emails = recipient_emails or os.environ.get("SURAKSHANET_RECIPIENTS", "").split(",")
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.cooldown = cooldown_seconds

        self._last_sent = 0.0
        self._lock = threading.Lock()

        self._configured = bool(self.sender_email and self.sender_password and self.recipient_emails[0])
        if not self._configured:
            print("⚠️ EmailNotifier not configured (missing credentials/recipients). Alerts will be skipped.")

    def _send(self, frame, violations: list):
        try:
            msg = EmailMessage()
            msg["Subject"] = f"🚨 SurakshaNet Alert: {len(violations)} PPE Violation(s) Detected"
            msg["From"] = self.sender_email
            msg["To"] = ", ".join(self.recipient_emails)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            body_lines = [f"SurakshaNet detected the following violations at {timestamp}:\n"]
            for v in violations:
                body_lines.append(
                    f"  - {v['class']} (confidence: {v['confidence']:.2f}) at bbox {v['bbox']}"
                )
            msg.set_content("\n".join(body_lines))

            # Attach the snapshot frame as JPEG
            success, encoded_img = cv2.imencode(".jpg", frame)
            if success:
                msg.add_attachment(
                    encoded_img.tobytes(),
                    maintype="image",
                    subtype="jpeg",
                    filename=f"violation_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
                )

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            print(f"📧 Alert email sent to {self.recipient_emails}")

        except Exception as e:
            print(f"❌ Failed to send alert email: {e}")

    def notify(self, frame, violations: list, force: bool = False):
        """
        Call when violations are detected. Sends asynchronously and respects
        the cooldown window. `frame` should be the raw BGR image (numpy array).
        """
        if not self._configured or not violations:
            return

        now = time.time()
        with self._lock:
            if not force and (now - self._last_sent) < self.cooldown:
                return
            self._last_sent = now

        frame_copy = frame.copy()
        threading.Thread(target=self._send, args=(frame_copy, violations), daemon=True).start()
