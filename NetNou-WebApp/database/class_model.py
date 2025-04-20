"""Class model for the application."""

from datetime import datetime

# In-memory class database
CLASSES = [
    {
        'id': '101',
        'name': 'Introduction to Computer Science',
        'code': 'CS101',
        'teacher_id': 2,
        'schedule': 'Monday, Wednesday 10:00-11:30',
        'room': 'R101',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': '102',
        'name': 'Data Structures and Algorithms',
        'code': 'CS201',
        'teacher_id': 2,
        'schedule': 'Tuesday, Thursday 13:30-15:00',
        'room': 'R102',
        'created_at': datetime.now().isoformat()
    }
]

# In-memory attendance database
ATTENDANCE = []

def get_class(class_id):
    """Get a class by ID.
    
    Args:
        class_id (str): The class ID to search for
        
    Returns:
        dict: Class object if found, None otherwise
    """
    for cls in CLASSES:
        if cls['id'] == class_id:
            return cls
    return None

def get_classes(teacher_id=None):
    """Get all classes, optionally filtered by teacher ID.
    
    Args:
        teacher_id (int, optional): Teacher ID to filter by
        
    Returns:
        list: List of classes
    """
    if teacher_id is None:
        return CLASSES
    return [cls for cls in CLASSES if cls['teacher_id'] == teacher_id]

def create_class(class_data):
    """Create a new class.
    
    Args:
        class_data (dict): Class data including name, code, etc.
        
    Returns:
        dict: Created class object or None if error
    """
    # Check if class code already exists
    if any(cls['code'] == class_data.get('code') for cls in CLASSES):
        return {'success': False, 'message': 'Class code already exists'}
    
    # Generate ID if not provided
    if 'id' not in class_data:
        # Generate a unique ID
        last_id = max(int(cls['id']) for cls in CLASSES) if CLASSES else 100
        class_data['id'] = str(last_id + 1)
    
    # Create class with defaults
    new_class = {
        'id': class_data.get('id'),
        'name': class_data.get('name', ''),
        'code': class_data.get('code', ''),
        'teacher_id': class_data.get('teacher_id'),
        'schedule': class_data.get('schedule', ''),
        'room': class_data.get('room', ''),
        'created_at': datetime.now().isoformat()
    }
    
    # Add to database
    CLASSES.append(new_class)
    
    return {'success': True, 'message': 'Class created successfully', 'class': new_class}

def record_attendance(class_id, student_id, status='present'):
    """Record attendance for a student in a class.
    
    Args:
        class_id (str): The class ID
        student_id (str): The student ID
        status (str): Attendance status ('present', 'absent', 'late')
        
    Returns:
        dict: Result of the operation
    """
    # Check if class exists
    if not get_class(class_id):
        return {'success': False, 'message': 'Class not found'}
    
    # Create attendance record
    attendance_record = {
        'class_id': class_id,
        'student_id': student_id,
        'status': status,
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().strftime('%Y-%m-%d')
    }
    
    # Add to database
    ATTENDANCE.append(attendance_record)
    
    return {'success': True, 'message': 'Attendance recorded', 'record': attendance_record}

def get_class_attendance(class_id, date=None):
    """Get attendance records for a class.
    
    Args:
        class_id (str): The class ID
        date (str, optional): Date to filter by (YYYY-MM-DD)
        
    Returns:
        list: List of attendance records
    """
    if date:
        return [record for record in ATTENDANCE 
                if record['class_id'] == class_id and record['date'] == date]
    return [record for record in ATTENDANCE if record['class_id'] == class_id]

def get_student_attendance(student_id, class_id=None):
    """Get attendance records for a student.
    
    Args:
        student_id (str): The student ID
        class_id (str, optional): Class ID to filter by
        
    Returns:
        list: List of attendance records
    """
    if class_id:
        return [record for record in ATTENDANCE 
                if record['student_id'] == student_id and record['class_id'] == class_id]
    return [record for record in ATTENDANCE if record['student_id'] == student_id] 