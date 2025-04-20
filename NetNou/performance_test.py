import cv2
import time
import numpy as np
from deepface import DeepFace
import os
import sys
import threading
import queue
import psutil

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, 'scratch_nn'))

# Import our modules
try:
    from scratch_nn.simple_nn import SimpleNN
    from demographic_analysis.live_demographics import get_engagement_label
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure you're running this script from the NetNou directory")
    sys.exit(1)

# --- Performance Test Configuration ---
TEST_DURATION = 30  # seconds
DETECTOR_BACKENDS = ['opencv', 'ssd', 'mediapipe']
ANALYSIS_RATES = [1, 2, 5, 10]  # analyze every N frames

# emotion_map is required for engagement prediction
emotion_map = {
    'happy': 0.9,
    'neutral': 0.5,
    'sad': 0.1,
    'angry': 0.0,
    'surprise': 0.8,
    'fear': 0.2,
    'disgust': 0.0
}
DEFAULT_EMOTION_VALUE = 0.5

def load_engagement_model():
    """Load the engagement prediction model"""
    WEIGHTS_PATH = os.path.join(script_dir, 'scratch_nn', 'engagement_nn_weights.npz')
    
    model = SimpleNN(input_size=1,
                     hidden_size=4,
                     output_size=1,
                     hidden_activation='relu', 
                     output_activation='sigmoid',
                     loss='bce'
                     )
    try:
        model.load_weights(WEIGHTS_PATH)
        return model
    except Exception as e:
        print(f"Warning: Failed to load engagement model: {e}")
        return None

def test_single_thread_performance(detector_backend, analyze_every, duration):
    """Test performance with single-threaded operation"""
    print(f"\nTesting single-thread: detector={detector_backend}, analyze_every={analyze_every}")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return None
    
    engagement_model = load_engagement_model()
    
    # Performance metrics
    start_time = time.time()
    end_time = start_time + duration
    frame_count = 0
    analysis_count = 0
    fps_values = []
    analysis_times = []
    
    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame_start = time.time()
        
        # Only analyze every N frames
        if frame_count % analyze_every == 0:
            try:
                # Resize for analysis 
                analysis_frame = cv2.resize(frame, (320, 240))
                
                # Time the analysis
                analysis_start = time.time()
                
                # Perform detection and analysis
                results = DeepFace.analyze(
                    img_path=analysis_frame,
                    actions=('emotion', 'age', 'gender'),
                    detector_backend=detector_backend,
                    enforce_detection=False,
                    silent=True
                )
                
                # Process results if needed
                face_results = []
                if isinstance(results, list):
                    face_results = results
                elif isinstance(results, dict) and 'region' in results:
                    face_results = [results]
                    
                # Calculate engagement if model is available
                if engagement_model and face_results:
                    for face_info in face_results:
                        emotion = face_info.get('dominant_emotion', 'neutral')
                        emotion_value = emotion_map.get(emotion.lower(), DEFAULT_EMOTION_VALUE)
                        nn_input = np.array([[emotion_value]])
                        engagement_score = engagement_model.predict(nn_input)[0][0]
                        engagement_label = get_engagement_label(engagement_score)
                
                analysis_time = time.time() - analysis_start
                analysis_times.append(analysis_time)
                analysis_count += 1
                
            except Exception as e:
                # Silently ignore analysis errors
                pass
                
        # Calculate FPS
        frame_time = time.time() - frame_start
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_values.append(current_fps)
        
        # Display basic info on frame
        cv2.putText(frame, f"FPS: {int(current_fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Mode: Single Thread", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"Detector: {detector_backend}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Show frame
        cv2.imshow('Performance Test (Single Thread)', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate metrics
    avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
    avg_analysis_time = sum(analysis_times) / len(analysis_times) if analysis_times else 0
    total_time = time.time() - start_time
    
    results = {
        'mode': 'single_thread',
        'detector': detector_backend,
        'analyze_every': analyze_every,
        'avg_fps': avg_fps,
        'avg_analysis_time': avg_analysis_time,
        'total_frames': frame_count,
        'analysis_count': analysis_count,
        'test_duration': total_time
    }
    
    # Print results
    print(f"Results - Single Thread:")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Average Analysis Time: {avg_analysis_time:.3f}s")
    print(f"  Frames Processed: {frame_count}")
    print(f"  Analyses Performed: {analysis_count}")
    print(f"  Total Time: {total_time:.2f}s")
    
    return results

def test_multi_thread_performance(detector_backend, analyze_every, duration):
    """Test performance with multi-threaded operation"""
    print(f"\nTesting multi-thread: detector={detector_backend}, analyze_every={analyze_every}")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return None
    
    engagement_model = load_engagement_model()
    
    # Set up queues for multi-threaded processing
    frame_queue = queue.Queue(maxsize=5)
    result_queue = queue.Queue()
    
    # Thread stop event
    thread_stop_event = threading.Event()
    
    # Analysis worker thread
    def analysis_worker():
        worker_analysis_count = 0
        worker_analysis_times = []
        
        while not thread_stop_event.is_set():
            try:
                frame_data = frame_queue.get(timeout=1.0)
                if frame_data is None:
                    continue
                    
                frame_id, analysis_frame = frame_data
                
                try:
                    # Time the analysis
                    analysis_start = time.time()
                    
                    # Perform detection and analysis
                    results = DeepFace.analyze(
                        img_path=analysis_frame,
                        actions=('emotion', 'age', 'gender'),
                        detector_backend=detector_backend,
                        enforce_detection=False,
                        silent=True
                    )
                    
                    # Process results if needed
                    face_results = []
                    if isinstance(results, list):
                        face_results = results
                    elif isinstance(results, dict) and 'region' in results:
                        face_results = [results]
                        
                    # Calculate engagement if model is available
                    if engagement_model and face_results:
                        for face_info in face_results:
                            emotion = face_info.get('dominant_emotion', 'neutral')
                            emotion_value = emotion_map.get(emotion.lower(), DEFAULT_EMOTION_VALUE)
                            nn_input = np.array([[emotion_value]])
                            engagement_score = engagement_model.predict(nn_input)[0][0]
                            engagement_label = get_engagement_label(engagement_score)
                    
                    analysis_time = time.time() - analysis_start
                    worker_analysis_times.append(analysis_time)
                    worker_analysis_count += 1
                    
                    # Put results in queue
                    result_queue.put((frame_id, analysis_time, face_results))
                    
                except Exception as e:
                    # Put empty results in queue
                    result_queue.put((frame_id, 0, []))
                
                finally:
                    frame_queue.task_done()
                    
            except queue.Empty:
                # No frames to process, continue waiting
                continue
                
        # Return statistics
        return worker_analysis_count, worker_analysis_times
    
    # Start analysis worker thread
    analysis_thread = threading.Thread(target=analysis_worker)
    analysis_thread.daemon = True
    analysis_thread.start()
    
    # Performance metrics
    start_time = time.time()
    end_time = start_time + duration
    frame_count = 0
    analysis_count = 0
    fps_values = []
    analysis_times = []
    
    # Dictionary to store results by frame ID
    results_cache = {}
    
    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame_start = time.time()
        
        # Only analyze every N frames
        if frame_count % analyze_every == 0:
            # Resize for analysis
            analysis_frame = cv2.resize(frame, (320, 240))
            
            # Add frame to queue if there's room
            if not frame_queue.full():
                frame_queue.put((frame_count, analysis_frame))
                analysis_count += 1
        
        # Process any available results
        while not result_queue.empty():
            frame_id, analysis_time, face_results = result_queue.get()
            if analysis_time > 0:
                analysis_times.append(analysis_time)
            results_cache[frame_id] = face_results
            result_queue.task_done()
        
        # Calculate FPS
        frame_time = time.time() - frame_start
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_values.append(current_fps)
        
        # Display basic info on frame
        cv2.putText(frame, f"FPS: {int(current_fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Mode: Multi Thread", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"Detector: {detector_backend}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Show frame
        cv2.imshow('Performance Test (Multi Thread)', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Stop worker thread
    thread_stop_event.set()
    if analysis_thread.is_alive():
        analysis_thread.join(timeout=1.0)
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate metrics
    avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
    avg_analysis_time = sum(analysis_times) / len(analysis_times) if analysis_times else 0
    total_time = time.time() - start_time
    
    results = {
        'mode': 'multi_thread',
        'detector': detector_backend,
        'analyze_every': analyze_every,
        'avg_fps': avg_fps,
        'avg_analysis_time': avg_analysis_time,
        'total_frames': frame_count,
        'analysis_count': analysis_count,
        'test_duration': total_time
    }
    
    # Print results
    print(f"Results - Multi Thread:")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Average Analysis Time: {avg_analysis_time:.3f}s")
    print(f"  Frames Processed: {frame_count}")
    print(f"  Analyses Performed: {analysis_count}")
    print(f"  Total Time: {total_time:.2f}s")
    
    return results

def run_performance_tests():
    """Run all performance tests and compare results"""
    print("Starting NetNou Performance Tests")
    print("=================================")
    print(f"Test duration: {TEST_DURATION} seconds per test")
    
    # Store results for comparison
    all_results = []
    
    # Run single thread tests
    for detector in DETECTOR_BACKENDS:
        for rate in ANALYSIS_RATES:
            results = test_single_thread_performance(detector, rate, TEST_DURATION)
            if results:
                all_results.append(results)
    
    # Run multi-thread tests
    for detector in DETECTOR_BACKENDS:
        for rate in ANALYSIS_RATES:
            results = test_multi_thread_performance(detector, rate, TEST_DURATION)
            if results:
                all_results.append(results)
    
    # Print comparison table
    print("\nPerformance Comparison")
    print("=====================")
    print(" Mode        | Detector  | Every N | Avg FPS | Avg Analysis Time")
    print("-------------|-----------|---------|---------|------------------")
    
    for result in all_results:
        mode = "Single Thread" if result['mode'] == 'single_thread' else "Multi Thread "
        print(f" {mode} | {result['detector']:<9} | {result['analyze_every']:<7} | {result['avg_fps']:.2f} | {result['avg_analysis_time']:.4f}s")

if __name__ == "__main__":
    run_performance_tests() 