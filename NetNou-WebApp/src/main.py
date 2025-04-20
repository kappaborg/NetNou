"""Command-line interface for face recognition functionality."""

import argparse
import os
import cv2
import numpy as np
from services.face_service import analyze_face
from core.nn.simple_nn import SimpleNN
from config import Config

def display_face_demo(webcam_index=0):
    """Run a simple face recognition demo using webcam.
    
    Args:
        webcam_index (int): Index of the webcam to use
    """
    print("Starting face recognition demo...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print(f"Error: Cannot open webcam at index {webcam_index}")
        return
    
    print("Camera opened successfully. Press 'q' to quit.")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break
        
        # Create a copy for display
        display_frame = frame.copy()
        
        try:
            # Analyze the frame
            result = analyze_face(frame)
            
            # If analysis succeeded, display results
            if 'error' not in result:
                # Draw info on frame
                cv2.putText(display_frame, f"Emotion: {result.get('emotion', 'unknown')}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if 'engagement' in result:
                    engagement = result['engagement']
                    cv2.putText(display_frame, 
                               f"Engagement: {engagement.get('label', 'unknown')} ({engagement.get('score', 0):.2f})", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                if 'age' in result:
                    cv2.putText(display_frame, f"Age: {result.get('age', 'unknown')}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if 'gender' in result:
                    cv2.putText(display_frame, f"Gender: {result.get('gender', 'unknown')}", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            else:
                # If analysis failed, display error
                cv2.putText(display_frame, "No face detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        except Exception as e:
            # On error, just display message
            cv2.putText(display_frame, f"Error: {str(e)[:50]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow("Face Recognition Demo", display_frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Demo stopped.")

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description='NetNou Face Recognition CLI')
    
    # Define commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run face recognition demo')
    demo_parser.add_argument('--webcam', type=int, default=0, 
                           help='Webcam index (default: 0)')
    
    # Register face command (future)
    register_parser = subparsers.add_parser('register', help='Register a face')
    register_parser.add_argument('--student', type=str, required=True, 
                                help='Student ID')
    register_parser.add_argument('--image', type=str, required=True, 
                                help='Path to image file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'demo':
        display_face_demo(args.webcam)
    elif args.command == 'register':
        print(f"Registration not implemented yet. Student ID: {args.student}, Image: {args.image}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 