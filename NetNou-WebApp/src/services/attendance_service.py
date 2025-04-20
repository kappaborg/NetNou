"""Attendance service for the application."""

from datetime import datetime, timedelta
from ..database.class_model import record_attendance as db_record_attendance
from ..database.student_model import get_student
from ..database.class_model import get_class_attendance, get_student_attendance
from ..services.face_service import FACE_EMBEDDINGS

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

def take_attendance(class_id, attendance_data):
    """Record attendance for multiple students in a class.
    
    Args:
        class_id (str): The class ID
        attendance_data (list): List of attendance records, each containing student_id and status
                              Example: [{"student_id": "1001", "status": "present"}, ...]
        
    Returns:
        dict: Result of the operation with success status and message
    """
    if not class_id:
        return {
            'success': False,
            'message': 'Class ID is required'
        }
        
    if not attendance_data or not isinstance(attendance_data, list):
        return {
            'success': False,
            'message': 'Attendance data must be a non-empty list'
        }
    
    # Process each attendance record
    results = []
    success_count = 0
    error_count = 0
    
    for record in attendance_data:
        student_id = record.get('student_id')
        status = record.get('status', 'present')
        
        result = record_attendance(class_id, student_id, status)
        
        if result['success']:
            success_count += 1
        else:
            error_count += 1
            
        results.append({
            'student_id': student_id,
            'status': result['success'],
            'message': result.get('message', '')
        })
    
    return {
        'success': error_count == 0,
        'message': f'Recorded {success_count} attendance records. {error_count} errors.',
        'results': results
    }

def get_recent_attendance(limit=10):
    """Get recent attendance records for all classes.
    
    Args:
        limit (int, optional): Number of records to return
        
    Returns:
        list: Recent attendance activities with formatted data for UI
    """
    # This would normally come from a database query
    # Placeholder data for demonstration
    activities = [
        {
            'id': 1,
            'student_name': 'John Smith',
            'class_name': 'Computer Science 101',
            'status': 'present',
            'timestamp': '2023-05-15 09:15:23',
            'icon': 'fa-check-circle',
            'description': 'John Smith attended Computer Science 101'
        },
        {
            'id': 2,
            'student_name': 'Maria Garcia',
            'class_name': 'Machine Learning 202',
            'status': 'late',
            'timestamp': '2023-05-15 10:05:12',
            'icon': 'fa-clock',
            'description': 'Maria Garcia was late to Machine Learning 202'
        },
        {
            'id': 3,
            'student_name': 'Ahmed Hassan',
            'class_name': 'Database Systems 301',
            'status': 'absent',
            'timestamp': '2023-05-14 14:30:00',
            'icon': 'fa-times-circle',
            'description': 'Ahmed Hassan was absent from Database Systems 301'
        },
        {
            'id': 4,
            'student_name': 'Jessica Lee',
            'class_name': 'Computer Science 101',
            'status': 'present',
            'timestamp': '2023-05-14 09:12:45',
            'icon': 'fa-check-circle',
            'description': 'Jessica Lee attended Computer Science 101'
        },
        {
            'id': 5,
            'student_name': 'Carlos Rodriguez',
            'class_name': 'Machine Learning 202',
            'status': 'present',
            'timestamp': '2023-05-14 10:02:33',
            'icon': 'fa-check-circle',
            'description': 'Carlos Rodriguez attended Machine Learning 202'
        }
    ]
    
    return activities[:limit]

def get_attendance_stats():
    """Get attendance statistics for dashboard and analytics.
    
    Returns:
        dict: Attendance statistics
    """
    # This would normally come from database queries
    # Placeholder data for demonstration
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    stats = {
        'total_students': 120,
        'attendance_today': 87,
        'average_engagement': '72%',
        'attendance_rate': '85%',
        'engagement_levels': {
            'high': 45,
            'medium': 35,
            'low': 20
        },
        'attendance_trend': {
            'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            'data': [85, 72, 78, 75, 82, 45, 20]
        },
        'engagement_trend': {
            'labels': ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
            'data': [65, 70, 68, 72]
        },
        'demographics': {
            'gender': {
                'male': 55,
                'female': 45
            },
            'age': {
                '18-20': 35,
                '21-23': 45,
                '24+': 20
            }
        }
    }
    
    return stats

def get_attendance_by_date_range(start_date, end_date, class_id=None):
    """Get attendance records filtered by date range and optionally by class.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        class_id (str, optional): Class ID to filter by
        
    Returns:
        list: Attendance records
    """
    # This would normally query the database with the date range
    # Placeholder implementation - would be replaced with actual query
    
    # Convert date strings to datetime objects
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        return {
            'success': False,
            'message': 'Invalid date format. Use YYYY-MM-DD'
        }
    
    # Mock data for now
    attendance_records = [
        {
            'date': '2023-05-15',
            'class_id': '1',
            'class_name': 'Computer Science 101',
            'total_students': 25,
            'present': 20,
            'absent': 3,
            'late': 2,
            'attendance_rate': '80%',
            'average_engagement': '75%'
        },
        {
            'date': '2023-05-14',
            'class_id': '1',
            'class_name': 'Computer Science 101',
            'total_students': 25,
            'present': 22,
            'absent': 2,
            'late': 1,
            'attendance_rate': '88%',
            'average_engagement': '78%'
        },
        {
            'date': '2023-05-15',
            'class_id': '2',
            'class_name': 'Machine Learning 202',
            'total_students': 18,
            'present': 15,
            'absent': 2,
            'late': 1,
            'attendance_rate': '83%',
            'average_engagement': '81%'
        }
    ]
    
    # Filter by class if specified
    if class_id:
        attendance_records = [record for record in attendance_records if record['class_id'] == class_id]
    
    return {
        'success': True,
        'data': attendance_records
    }

def get_student_details(student_id):
    """Get detailed information for a student including attendance history.
    
    Args:
        student_id (str): The ID of the student
        
    Returns:
        dict: Result with student details and attendance data
    """
    # Get basic student information
    student = get_student(student_id)
    if not student:
        return {
            'success': False, 
            'message': f'Student with ID {student_id} not found'
        }
    
    # Check if face is registered
    has_face_registered = False
    for embedding in FACE_EMBEDDINGS:
        if embedding['student_id'] == student_id:
            has_face_registered = True
            break
    
    # Get attendance history
    attendance_records = get_student_attendance(student_id)
    
    # Count attendance statistics
    total_classes = len(attendance_records)
    present_count = sum(1 for record in attendance_records if record['status'] == 'present')
    absent_count = total_classes - present_count
    attendance_rate = (present_count / total_classes * 100) if total_classes > 0 else 0
    
    # Format recent attendance for display
    recent_attendance = []
    for record in sorted(attendance_records, key=lambda x: x['timestamp'], reverse=True)[:5]:
        recent_attendance.append({
            'class_id': record['class_id'],
            'date': record['timestamp'],
            'status': record['status']
        })
    
    return {
        'success': True,
        'student': student,
        'attendance': {
            'total_classes': total_classes,
            'present_count': present_count,
            'absent_count': absent_count,
            'attendance_rate': f"{attendance_rate:.1f}%",
            'recent_attendance': recent_attendance
        },
        'face_registered': has_face_registered
    }

def get_attendance_for_class(class_id):
    """Get attendance records for a specific class.
    
    Args:
        class_id (str): The ID of the class
        
    Returns:
        list: Attendance records for the class
    """
    # Get attendance records from database
    attendance_records = get_class_attendance(class_id)
    
    # Format records for display
    formatted_records = []
    for record in attendance_records:
        student = get_student(record['student_id'])
        if student:
            formatted_records.append({
                'id': record['id'],
                'student_id': record['student_id'],
                'student_name': f"{student['first_name']} {student['last_name']}",
                'status': record['status'],
                'timestamp': record['timestamp'],
                'date': record['timestamp'].split(' ')[0]
            })
    
    return formatted_records 