"""Student service for the application."""

from ..database.student_model import (
    get_student as db_get_student,
    get_students as db_get_students,
    create_student as db_create_student,
    update_student as db_update_student,
    delete_student as db_delete_student
)
from ..database.class_model import get_student_attendance
from ..services.face_service import delete_face_registration

def get_students():
    """Get all students.
    
    Returns:
        list: List of all students
    """
    students = db_get_students()
    
    # Never return sensitive data to the client
    return sanitize_students(students)

def get_student(student_id):
    """Get a student by ID.
    
    Args:
        student_id (str): The student ID
        
    Returns:
        dict: Student object if found, None otherwise
    """
    student = db_get_student(student_id)
    if not student:
        return None
    
    # Never return sensitive data to the client
    return sanitize_student(student)

def get_student_details(student_id):
    """Get detailed information about a student including attendance records.
    
    Args:
        student_id (str): The student ID
        
    Returns:
        dict: Student object with attendance data if found, None otherwise
    """
    # Get basic student info
    student = get_student(student_id)
    if not student:
        return None
    
    # Get student's attendance records
    attendance_records = get_student_attendance(student_id)
    
    # Compile complete student details
    student_details = {
        **student,
        'attendance_records': attendance_records,
        'attendance_summary': {
            'total': len(attendance_records),
            'present': sum(1 for record in attendance_records if record['status'] == 'present'),
            'absent': sum(1 for record in attendance_records if record['status'] == 'absent'),
            'late': sum(1 for record in attendance_records if record['status'] == 'late')
        }
    }
    
    return student_details

def create_student(student_data):
    """Create a new student.
    
    Args:
        student_data (dict): Student data including first_name, last_name, etc.
        
    Returns:
        dict: Result of the operation with success status and message
    """
    # Validate required fields
    required_fields = ['first_name', 'last_name', 'email']
    for field in required_fields:
        if field not in student_data or not student_data[field]:
            return {
                'success': False,
                'message': f'Missing required field: {field}'
            }
    
    # Create student
    new_student = db_create_student(student_data)
    
    if not new_student:
        return {
            'success': False,
            'message': 'Failed to create student. Student may already exist.'
        }
    
    # Never return sensitive data to the client
    return {
        'success': True,
        'message': 'Student created successfully',
        'student': sanitize_student(new_student)
    }

def update_student(student_id, updates):
    """Update a student.
    
    Args:
        student_id (str): The student ID
        updates (dict): Field updates for the student
        
    Returns:
        dict: Result of the operation with success status and message
    """
    # Validate student exists
    student = db_get_student(student_id)
    if not student:
        return {
            'success': False,
            'message': f'Student with ID {student_id} not found'
        }
    
    # Update student
    updated_student = db_update_student(student_id, updates)
    
    if not updated_student:
        return {
            'success': False,
            'message': 'Failed to update student'
        }
    
    # Never return sensitive data to the client
    return {
        'success': True,
        'message': 'Student updated successfully',
        'student': sanitize_student(updated_student)
    }

def sanitize_student(student):
    """Remove sensitive data from student object.
    
    Args:
        student (dict): Student object
        
    Returns:
        dict: Sanitized student object
    """
    # Make a copy to avoid modifying the original
    sanitized = {**student}
    
    # Remove sensitive fields if they exist
    sensitive_fields = []
    for field in sensitive_fields:
        if field in sanitized:
            del sanitized[field]
    
    return sanitized

def sanitize_students(students):
    """Remove sensitive data from a list of student objects.
    
    Args:
        students (list): List of student objects
        
    Returns:
        list: List of sanitized student objects
    """
    return [sanitize_student(student) for student in students]

def delete_student(student_id):
    """Delete a student and their face registration (if any).
    
    Args:
        student_id (str): The student ID to delete
        
    Returns:
        dict: Result of the operation with success status and message
    """
    # Validate student exists
    student = db_get_student(student_id)
    if not student:
        return {
            'success': False,
            'message': f'Student with ID {student_id} not found'
        }
    
    # Delete face registration if exists
    if student.get('has_face_registered', False):
        delete_face_registration(student_id)
    
    # Delete student
    deleted = db_delete_student(student_id)
    
    if not deleted:
        return {
            'success': False,
            'message': 'Failed to delete student'
        }
    
    return {
        'success': True,
        'message': 'Student deleted successfully'
    } 