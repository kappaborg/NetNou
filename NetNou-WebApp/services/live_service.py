"""Live demographic analysis service for the application."""

import cv2
import numpy as np
import threading
import queue
import time
from deepface import DeepFace
from ..config import Config
from .face_service import analyze_face, analyze_deepface_batch, load_engagement_model

class LiveDemographicAnalysis:
    """Class for handling live demographic analysis from video streams."""
    
    def __init__(self, camera_id=0, analysis_interval=1.0, batch_size=10):
        """Initialize the live demographic analysis.
        
        Args:
            camera_id (int): ID of the camera to use
            analysis_interval (float): Minimum time between analyses in seconds
            batch_size (int): Number of frames to analyze at once
        """
        self.camera_id = camera_id
        self.analysis_interval = analysis_interval
        self.batch_size = batch_size
        
        # Initialize video capture
        self.cap = None
        
        # Queues for frame processing
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue()
        
        # Threading control
        self.is_running = False
        self.threads = []
        
        # Results tracking
        self.latest_frame = None
        self.latest_results = []
        self.face_count = 0
        self.demographics = {
            'emotions': {},
            'genders': {},
            'age_groups': {},
            'engagement': {
                'high': 0,
                'medium': 0,
                'low': 0
            }
        }
        
        # Performance metrics
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        self.processing_times = []
        
        # Load engagement model
        self.engagement_model = load_engagement_model()
        
    def start(self):
        """Start the live demographic analysis."""
        if self.is_running:
            return False
            
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open camera with ID {self.camera_id}")
            
        # Set resolution if configured
        if hasattr(Config, 'CAMERA_RESOLUTION'):
            width, height = Config.CAMERA_RESOLUTION
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
        self.is_running = True
        
        # Start worker threads
        self.threads = []
        
        # Frame capture thread
        capture_thread = threading.Thread(target=self._capture_frames)
        capture_thread.daemon = True
        self.threads.append(capture_thread)
        
        # Analysis thread
        analysis_thread = threading.Thread(target=self._analyze_frames)
        analysis_thread.daemon = True
        self.threads.append(analysis_thread)
        
        # Start all threads
        for thread in self.threads:
            thread.start()
            
        return True
        
    def stop(self):
        """Stop the live demographic analysis."""
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
                
        # Release camera
        if self.cap and self.cap.isOpened():
            self.cap.release()
            
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
                
        self.threads = []
        return True
        
    def _capture_frames(self):
        """Continuously capture frames from the camera."""
        last_time = time.time()
        
        while self.is_running:
            try:
                # Capture frame
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    print("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                    
                # Update FPS calculation
                current_time = time.time()
                self.frame_count += 1
                
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time
                    
                # Store latest frame for display
                self.latest_frame = frame.copy()
                
                # Add to queue if not full
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                    
                # Control capture rate
                elapsed = time.time() - last_time
                if elapsed < 0.03:  # ~30 FPS max
                    time.sleep(0.03 - elapsed)
                    
                last_time = time.time()
                    
            except Exception as e:
                print(f"Error in frame capture: {str(e)}")
                time.sleep(0.1)
                
    def _analyze_frames(self):
        """Analyze frames from the queue."""
        last_analysis_time = 0
        frames_to_analyze = []
        
        while self.is_running:
            try:
                # Check if it's time for analysis
                current_time = time.time()
                
                if current_time - last_analysis_time < self.analysis_interval:
                    time.sleep(0.1)
                    continue
                    
                # Collect frames for batch analysis
                frames_to_analyze = []
                
                # Try to get frames up to batch_size
                for _ in range(self.batch_size):
                    try:
                        if not self.frame_queue.empty():
                            frame = self.frame_queue.get(timeout=0.1)
                            frames_to_analyze.append(frame)
                    except queue.Empty:
                        break
                        
                # If we got frames, analyze them
                if frames_to_analyze:
                    analysis_start = time.time()
                    
                    # Perform batch analysis
                    analysis_results = analyze_deepface_batch(
                        frames_to_analyze,
                        backend=Config.FACE_DETECTION_BACKEND,
                        actions=Config.FACE_ANALYZE_ACTIONS
                    )
                    
                    # Process results
                    self._process_analysis_results(analysis_results)
                    
                    # Calculate processing time
                    processing_time = time.time() - analysis_start
                    self.processing_times.append(processing_time)
                    
                    # Keep only the last 10 processing times for average calculation
                    if len(self.processing_times) > 10:
                        self.processing_times = self.processing_times[-10:]
                        
                    last_analysis_time = current_time
                    
            except Exception as e:
                print(f"Error in frame analysis: {str(e)}")
                time.sleep(0.5)
                
    def _process_analysis_results(self, analysis_results):
        """Process analysis results and update statistics.
        
        Args:
            analysis_results (list): List of DeepFace analysis results
        """
        # Reset current counts
        emotions = {}
        genders = {}
        age_groups = {}
        engagement = {'high': 0, 'medium': 0, 'low': 0}
        
        face_count = 0
        
        for result in analysis_results:
            # Skip errors
            if isinstance(result, dict) and 'error' in result:
                continue
                
            # Handle list result (DeepFace sometimes returns a list)
            if isinstance(result, list):
                for face in result:
                    self._process_single_face(face, emotions, genders, age_groups, engagement)
                    face_count += 1
            else:
                self._process_single_face(result, emotions, genders, age_groups, engagement)
                face_count += 1
                
        # Update overall statistics
        if face_count > 0:
            self.face_count = face_count
            self.demographics = {
                'emotions': emotions,
                'genders': genders,
                'age_groups': age_groups,
                'engagement': engagement
            }
            
        self.latest_results = analysis_results
        
    def _process_single_face(self, face, emotions, genders, age_groups, engagement):
        """Process a single face analysis result.
        
        Args:
            face (dict): Face analysis result from DeepFace
            emotions (dict): Emotions count dictionary to update
            genders (dict): Genders count dictionary to update
            age_groups (dict): Age groups count dictionary to update
            engagement (dict): Engagement count dictionary to update
        """
        # Process emotion
        if 'emotion' in face:
            emotion_data = face['emotion']
            dominant_emotion = max(emotion_data.items(), key=lambda x: x[1])[0]
            emotions[dominant_emotion] = emotions.get(dominant_emotion, 0) + 1
            
            # Calculate engagement based on emotion
            engagement_score = 0.5  # Default mid-level engagement
            
            if dominant_emotion.lower() in ['happy', 'surprise']:
                engagement_score = 0.8
            elif dominant_emotion.lower() in ['neutral']:
                engagement_score = 0.5
            elif dominant_emotion.lower() in ['sad', 'angry', 'fear', 'disgust']:
                engagement_score = 0.2
                
            # Get engagement label
            if engagement_score >= 0.8:
                engagement_label = 'high'
            elif engagement_score >= 0.4:
                engagement_label = 'medium'
            else:
                engagement_label = 'low'
                
            engagement[engagement_label] = engagement.get(engagement_label, 0) + 1
            
        # Process gender
        if 'gender' in face:
            gender_data = face['gender']
            if isinstance(gender_data, dict):
                gender = gender_data.get('dominant_gender', 'Unknown')
            else:
                gender = str(gender_data)
                
            genders[gender] = genders.get(gender, 0) + 1
            
        # Process age
        if 'age' in face:
            age = int(face['age'])
            
            # Group ages
            if age < 18:
                age_group = '<18'
            elif age < 25:
                age_group = '18-24'
            elif age < 35:
                age_group = '25-34'
            elif age < 45:
                age_group = '35-44'
            elif age < 55:
                age_group = '45-54'
            else:
                age_group = '55+'
                
            age_groups[age_group] = age_groups.get(age_group, 0) + 1
            
    def get_latest_frame(self):
        """Get the latest captured frame.
        
        Returns:
            numpy.ndarray: Latest frame or None
        """
        return self.latest_frame
        
    def get_latest_statistics(self):
        """Get the latest demographic statistics.
        
        Returns:
            dict: Latest demographic statistics
        """
        # Calculate average processing time
        avg_processing_time = 0
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            
        return {
            'face_count': self.face_count,
            'demographics': self.demographics,
            'performance': {
                'fps': self.fps,
                'avg_processing_time': avg_processing_time
            }
        }
        
    def get_annotated_frame(self):
        """Get the latest frame with annotations.
        
        Returns:
            numpy.ndarray: Annotated frame or None
        """
        if self.latest_frame is None:
            return None
            
        frame = self.latest_frame.copy()
        
        # Add statistics to the frame
        stats = self.get_latest_statistics()
        
        # Add face count
        cv2.putText(
            frame,
            f"Faces: {stats['face_count']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Add FPS
        cv2.putText(
            frame,
            f"FPS: {stats['performance']['fps']:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Add processing time
        cv2.putText(
            frame,
            f"Processing: {stats['performance']['avg_processing_time']*1000:.0f}ms",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Add dominant emotion if available
        emotions = stats['demographics'].get('emotions', {})
        if emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            cv2.putText(
                frame,
                f"Emotion: {dominant_emotion}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
        return frame

def analyze_live_demographics(camera_id=0, display=True, analysis_interval=1.0, max_duration=60):
    """Run live demographic analysis.
    
    Args:
        camera_id (int): ID of the camera to use
        display (bool): Whether to display the video feed
        analysis_interval (float): Minimum time between analyses in seconds
        max_duration (int): Maximum duration in seconds (0 for unlimited)
        
    Returns:
        dict: Final demographic statistics
    """
    analyzer = LiveDemographicAnalysis(
        camera_id=camera_id,
        analysis_interval=analysis_interval
    )
    
    try:
        analyzer.start()
        
        start_time = time.time()
        
        while True:
            # Check duration limit
            if max_duration > 0 and time.time() - start_time > max_duration:
                break
                
            # Display if requested
            if display:
                frame = analyzer.get_annotated_frame()
                
                if frame is not None:
                    cv2.imshow('Live Demographics', frame)
                    
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or q to exit
                    break
            else:
                # If not displaying, just sleep a bit
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        pass
    finally:
        analyzer.stop()
        
        if display:
            cv2.destroyAllWindows()
            
    return analyzer.get_latest_statistics() 