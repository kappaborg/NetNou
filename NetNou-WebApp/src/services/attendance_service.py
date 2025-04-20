"""Attendance service for the application."""

from ..database.class_model import record_attendance as db_record_attendance
from ..database.student_model import get_student

def record_attendance(class_id, student_id, status='present'):
    """Record attendance for a student in a class.
    
    Args:
        class_id (str): The class ID
        student_id (str): The student ID
        status (str, optional): Attendance status ('present', 'absent', 'late')
        
    Returns:
        dict: Result of the operation with success status and message
    """
    # Validate input
    if not class_id or not student_id:
        return {
            'success': False,
            'message': 'Class ID and Student ID are required'
        }
    
    # Validate status
    valid_statuses = ['present', 'absent', 'late']
    if status not in valid_statuses:
        return {
            'success': False,
            'message': f'Invalid status. Must be one of: {", ".join(valid_statuses)}'
        }
    
    # Get student to verify existence
    student = get_student(student_id)
    if not student:
        return {
            'success': False,
            'message': f'Student with ID {student_id} not found'
        }
    
    # Record attendance
    result = db_record_attendance(class_id, student_id, status)
    
    return result 