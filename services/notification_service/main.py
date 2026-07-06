"""
Microservices Pattern - Notification Service
=============================================
Independent service for sending alerts to at-risk students and officers.
Decoupled from prediction logic - communicates via API calls.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

app = FastAPI(
    title="Notification Service",
    description="Handles alerts for at-risk students",
    version="1.0.0"
)

# In-memory notification log (in production: use message queue + email/SMS)
notification_log = []


class AlertRequest(BaseModel):
    student: dict
    risk_score: float
    risk_level: str


class NotificationRecord(BaseModel):
    timestamp: str
    student_data: dict
    risk_score: float
    risk_level: str
    message: str
    status: str


@app.post("/alert")
def send_alert(alert: AlertRequest):
    """Send notification for at-risk students."""
    message = (
        f"ALERT: Student has a placement probability of {alert.risk_score:.1%}. "
        f"Risk Level: {alert.risk_level}. "
        f"Immediate intervention recommended."
    )

    record = NotificationRecord(
        timestamp=datetime.now().isoformat(),
        student_data=alert.student,
        risk_score=alert.risk_score,
        risk_level=alert.risk_level,
        message=message,
        status="sent"
    )

    notification_log.append(record.dict())
    print(f"[Notification] {message}")

    return {"status": "alert_sent", "message": message}


@app.get("/notifications")
def get_notifications(limit: Optional[int] = 10):
    """Retrieve recent notifications."""
    return {"notifications": notification_log[-limit:], "total": len(notification_log)}


@app.get("/health")
def health_check():
    return {"status": "healthy", "pending_notifications": len(notification_log)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
