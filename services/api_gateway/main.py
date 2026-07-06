"""
Microservices Pattern - API Gateway
=====================================
Central entry point that routes requests to appropriate microservices.
Handles cross-cutting concerns: authentication, routing, rate limiting.
"""

import httpx
from fastapi import FastAPI, HTTPException, Header
from typing import Optional

app = FastAPI(
    title="Placement Prediction System - API Gateway",
    description="Central gateway routing requests to microservices",
    version="1.0.0"
)

# Service registry
SERVICES = {
    "prediction": "http://localhost:8003",
    "notification": "http://localhost:8005",
}


def verify_token(authorization: Optional[str] = Header(None)):
    """Simple token verification (demo purposes)."""
    # In production: validate JWT token with auth service
    if authorization and authorization.startswith("Bearer "):
        return True
    return True  # Allow all for demo


@app.post("/api/predict")
async def predict(student_data: dict):
    """Route prediction request to prediction microservice."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{SERVICES['prediction']}/predict",
                json=student_data,
                timeout=10.0
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.json().get("detail", "Prediction service error")
                )

            result = response.json()

            # Trigger notification if high risk
            if result.get("confidence", 1.0) < 0.4:
                try:
                    await client.post(
                        f"{SERVICES['notification']}/alert",
                        json={
                            "student": student_data,
                            "risk_score": result["confidence"],
                            "risk_level": result["risk_level"]
                        }
                    )
                except Exception:
                    pass  # Notification failure shouldn't block prediction

            return result

        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="Prediction service unavailable"
            )


@app.get("/api/health")
async def gateway_health():
    """Check health of all services."""
    health_status = {"gateway": "healthy"}

    async with httpx.AsyncClient() as client:
        for service_name, service_url in SERVICES.items():
            try:
                response = await client.get(f"{service_url}/health", timeout=5.0)
                health_status[service_name] = response.json()
            except Exception:
                health_status[service_name] = {"status": "unavailable"}

    return health_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
