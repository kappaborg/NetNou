"""Class service for the application."""

from ..database.class_model import (
    get_class,
    get_classes as db_get_classes,
    create_class as db_create_class,
    record_attendance as db_record_attendance,
    get_class_attendance as db_get_class_attendance
)
from ..database.student_model import get_student

def get_classes(teacher_id=None):
    """Get all classes, optionally filtered by teacher ID.
    
    Args:
        teacher_id (int, optional): Teacher ID to filter by
        
    Returns:
        list: List of classes
    """
    return db_get_classes(teacher_id)

def create_class(class_data):
    """Create a new class.
    
    Args:
        class_data (dict): Class data including name, code, etc.
        
    Returns:
        dict: Result of the operation with success status and message
    """
    # Validate required fields
    required_fields = ['name', 'code', 'teacher_id']
    for field in required_fields:
        if field not in class_data or not class_data[field]:
            return {
                'success': False,
                'message': f'Missing required field: {field}'
            }
    
    # Create class
    result = db_create_class(class_data)
    return result

def get_class_attendance(class_id, date=None):
    """Get attendance records for a class with student details.
    
    Args:
        class_id (str): The class ID
        date (str, optional): Date to filter by (YYYY-MM-DD)
        
    Returns:
        list: List of attendance records with student details
    """
    # Check if class exists
    class_obj = get_class(class_id)
    if not class_obj:
        return []
    
    # Get attendance records
    records = db_get_class_attendance(class_id, date)
    
    # Add student details to each record
    enhanced_records = []
    for record in records:
        student = get_student(record['student_id'])
        if student:
            enhanced_record = {**record}  # Copy the record
            enhanced_record['student_name'] = f"{student['first_name']} {student['last_name']}"
            enhanced_records.append(enhanced_record)
        else:
            enhanced_records.append(record)  # Keep original if student not found
    
    return enhanced_records

def record_attendance(class_id, student_id, status='present'):
    """Record attendance for a student in a class.
    
    Args:
        class_id (str): The class ID
        student_id (str): The student ID
        status (str, optional): Attendance status ('present', 'absent', 'late')
        
    Returns:
        dict: Result of the operation with success status and message
    """
    # Validate inputs
    if not class_id:
        return {'success': False, 'message': 'Class ID is required'}
    
    if not student_id:
        return {'success': False, 'message': 'Student ID is required'}
    
    # Check if class exists
    class_obj = get_class(class_id)
    if not class_obj:
        return {'success': False, 'message': 'Class not found'}
    
    # Check if student exists
    student = get_student(student_id)
    if not student:
        return {'success': False, 'message': 'Student not found'}
    
    # Record attendance
    result = db_record_attendance(class_id, student_id, status)
    
    if result['success']:
        result['student_name'] = f"{student['first_name']} {student['last_name']}"
    
    return result 