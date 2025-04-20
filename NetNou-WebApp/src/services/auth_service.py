"""Authentication service for the application."""

from flask import session
import hashlib
from ..database.user_model import get_user_by_username

def authenticate_user(username, password):
    """Authenticate a user with username and password.
    
    Args:
        username (str): The username of the user
        password (str): The password of the user
        
    Returns:
        dict: Result of authentication with success status and message
    """
    if not username or not password:
        return {'success': False, 'message': 'Username and password are required'}
    
    # Get user from database
    user = get_user_by_username(username)
    
    if not user:
        return {'success': False, 'message': 'Invalid username or password'}
    
    # Hash the provided password
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    # Check password
    if hashed_password == user['password']:
        # Remove password from user object before returning
        user_data = {k: v for k, v in user.items() if k != 'password'}
        return {'success': True, 'message': 'Authentication successful', 'user': user_data}
    
    return {'success': False, 'message': 'Invalid username or password'}

def is_authenticated():
    """Check if the current user is authenticated.
    
    Returns:
        bool: True if authenticated, False otherwise
    """
    return 'user' in session 