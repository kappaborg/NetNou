from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
from app.models.models import Student, Attendance, Class
from app.schemas.schemas import AttendanceCreate
from app.services.face_recognition_service import FaceRecognitionModel

# Initialize face recognition model
face_recognition_model = FaceRecognitionModel()

def process_attendance(
    db: Session,
    class_id: int,
    image: str
) -> Optional[Attendance]:
    """Process attendance using CNN-based face recognition"""
    # Get face embedding from image
    face_embedding = face_recognition_model.get_face_embedding(image)
    if face_embedding is None:
        return None
    
    # Get all students in the class
    class_obj = db.query(Class).filter(Class.id == class_id).first()
    if not class_obj:
        return None
    
    # Find matching student
    best_match = None
    highest_score = 0
    
    for class_student in class_obj.students:
        student = class_student.student
        if not student.face_encoding:
            continue
        
        matches, score = face_recognition_model.compare_faces(
            student.face_encoding,
            face_embedding
        )
        
        if matches and score > highest_score:
            best_match = student
            highest_score = score
    
    if best_match is None:
        return None
    
    # Create attendance record
    attendance_data = AttendanceCreate(
        class_id=class_id,
        student_id=best_match.id,
        status="present",
        confidence_score=int(highest_score)
    )
    
    # Check if attendance already exists for today
    today = datetime.now().date()
    existing_attendance = db.query(Attendance).filter(
        Attendance.class_id == class_id,
        Attendance.student_id == best_match.id,
        Attendance.date == today
    ).first()
    
    if existing_attendance:
        return existing_attendance
    
    # Create new attendance record
    new_attendance = Attendance(
        class_id=attendance_data.class_id,
        student_id=attendance_data.student_id,
        date=today,
        status=attendance_data.status,
        time_in=datetime.now(),
        confidence_score=attendance_data.confidence_score
    )
    
    db.add(new_attendance)
    db.commit()
    db.refresh(new_attendance)
    
    return new_attendance