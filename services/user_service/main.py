"""
Microservices Pattern - User Service
======================================
Manages user profiles and role-based access.
Independent service with its own data store.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="User Service",
    description="Manages user profiles and authentication",
    version="1.0.0"
)

# In-memory user store (in production: use PostgreSQL)
users_db = {}


class User(BaseModel):
    user_id: str
    name: str
    role: str  # "student", "officer", "recruiter"
    email: Optional[str] = None


@app.post("/users")
def create_user(user: User):
    """Create a new user profile."""
    if user.user_id in users_db:
        raise HTTPException(status_code=409, detail="User already exists")
    users_db[user.user_id] = user.dict()
    return {"status": "created", "user": user.dict()}


@app.get("/users/{user_id}")
def get_user(user_id: str):
    """Retrieve user profile."""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return users_db[user_id]


@app.get("/users")
def list_users(role: Optional[str] = None):
    """List all users, optionally filtered by role."""
    if role:
        filtered = {k: v for k, v in users_db.items() if v["role"] == role}
        return {"users": list(filtered.values())}
    return {"users": list(users_db.values())}


@app.get("/health")
def health_check():
    return {"status": "healthy", "total_users": len(users_db)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
