"""User model for authentication and authorization."""

# This is a simple in-memory database for demonstration
# In a real application, this would use SQLAlchemy or similar

import hashlib
from datetime import datetime

# In-memory user database
USERS = [
    {
        'id': 1,
        'username': 'admin',
        'password': hashlib.sha256('password123'.encode()).hexdigest(),
        'first_name': 'Admin',
        'last_name': 'User',
        'email': 'admin@example.com',
        'role': 'admin',
        'created_at': datetime.now().isoformat()
    },
    {
        'id': 2,
        'username': 'teacher',
        'password': hashlib.sha256('password123'.encode()).hexdigest(),
        'first_name': 'Teacher',
        'last_name': 'User',
        'email': 'teacher@example.com',
        'role': 'teacher',
        'created_at': datetime.now().isoformat()
    }
]

def get_user_by_username(username):
    """Get a user by username.
    
    Args:
        username (str): The username to search for
        
    Returns:
        dict: User object if found, None otherwise
    """
    for user in USERS:
        if user['username'] == username:
            return user
    return None

def get_user_by_id(user_id):
    """Get a user by ID.
    
    Args:
        user_id (int): The user ID to search for
        
    Returns:
        dict: User object if found, None otherwise
    """
    for user in USERS:
        if user['id'] == user_id:
            return user
    return None

def create_user(user_data):
    """Create a new user.
    
    Args:
        user_data (dict): User data including username, password, etc.
        
    Returns:
        dict: Created user object
    """
    # Check if username already exists
    if get_user_by_username(user_data['username']):
        return None
    
    # Hash the password
    if 'password' in user_data:
        user_data['password'] = hashlib.sha256(user_data['password'].encode()).hexdigest()
    
    # Generate ID
    user_id = max(user['id'] for user in USERS) + 1 if USERS else 1
    
    # Create user with defaults
    new_user = {
        'id': user_id,
        'username': user_data.get('username'),
        'password': user_data.get('password'),
        'first_name': user_data.get('first_name', ''),
        'last_name': user_data.get('last_name', ''),
        'email': user_data.get('email', ''),
        'role': user_data.get('role', 'user'),
        'created_at': datetime.now().isoformat()
    }
    
    # Add to database
    USERS.append(new_user)
    
    # Return user without password
    return {k: v for k, v in new_user.items() if k != 'password'} 