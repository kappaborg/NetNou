from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import uvicorn

from app.database import get_db, engine
from app.models import models
from app.schemas import schemas
from app.services import attendance_service, auth_service
from app.core.config import settings

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Student Attendance System",
    description="An AI-powered system for managing student attendance",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:3003", "http://localhost:3004"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to AI Student Attendance System API"}

# Include routers
from app.api.endpoints import auth, students, attendance, classes

app.include_router(auth.router, prefix="/api", tags=["Authentication"])
app.include_router(students.router, prefix="/api", tags=["Students"])
app.include_router(attendance.router, prefix="/api", tags=["Attendance"])
app.include_router(classes.router, prefix="/api", tags=["Classes"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)