import cv2
import numpy as np
import time


# Object size thresholds
MIN_WIDTH = 60      # Minimum width for vehicles
MIN_HEIGHT = 60     # Minimum height for vehicles
MAX_WIDTH = 400     # Maximum width (ignore too large objects)
MAX_HEIGHT = 400    # Maximum height

# Line configuration
LINE_Y = 180        # Vertical position of counting line
LINE_COLOR = (255, 255, 0)  # Cyan color
LINE_THICKNESS = 2

# Sensitivity parameters
CROSSING_OFFSET = 8      # Tolerance for line crossing (pixels)
TRACKING_DISTANCE = 70   # Maximum distance to match objects between frames
MIN_CROSSING_SPEED = 1.5 # Minimum speed to count as crossing (px/frame)
MAX_CROSSING_SPEED = 50  # Maximum speed to filter out noise

# KNN Background subtraction parameters
KNN_HISTORY = 500        # Number of frames for background modeling
KNN_DIST_THRESHOLD = 400 # Distance threshold (lower = more sensitive)
KNN_SHADOWS = True       # Detect and ignore shadows
LEARNING_RATE = 0.001    # How fast the background model adapts

# Morphology parameters (for cleaning up the detection mask)
MORPH_KERNEL_SIZE = 5    # Size of kernel for noise removal
MORPH_ITERATIONS = 2     # Number of times to apply morphology operations

# Tracking parameters
MAX_FRAMES_LOST = 15     # Remove objects not seen for this many frames
MIN_TRACK_LENGTH = 3     # Minimum frames to track before counting
# ==================================-0

# Global variables
object_tracks = []       # Stores information about each tracked object
object_id = 0            # Unique ID for each new object
object_count = 0         # Total number of vehicles counted
crossed_ids = []         # IDs of objects that have already been counted
frame_count = 0          # Current frame number
start_time = time.time() # For calculating FPS

# Initialize KNN background subtractor
knn = cv2.createBackgroundSubtractorKNN(
    history=KNN_HISTORY,
    dist2Threshold=KNN_DIST_THRESHOLD,
    detectShadows=KNN_SHADOWS
)

def rect_center(x, y, w, h):
    """Calculate the center point of a rectangle"""
    return (x + w // 2, y + h // 2)

def is_valid_object(w, h, x, y, frame_height):
    """Check if detected object is likely a vehicle based on size and aspect ratio"""
    # Size check
    size_ok = (MIN_WIDTH <= w <= MAX_WIDTH and 
               MIN_HEIGHT <= h <= MAX_HEIGHT)
    
    # Aspect ratio check (vehicles are typically wider than tall)
    aspect_ratio = w / h if h > 0 else 0
    aspect_ok = 0.5 <= aspect_ratio <= 3.0
    
    # Area check
    area = w * h
    area_ok = 2000 <= area <= 100000  # Reasonable area for vehicles
    
    return size_ok and aspect_ok and area_ok

def is_crossing_line(current_cy, prev_cy, current_cx, prev_cx):
    """
    Check if an object has crossed the counting line
    Returns: (is_crossing, direction, confidence)
    """
    # Calculate movement
    dx = current_cx - prev_cx
    dy = current_cy - prev_cy
    
    # Check for downward crossing (from above line to below)
    crossing_downward = (prev_cy < LINE_Y - CROSSING_OFFSET and 
                         current_cy >= LINE_Y - CROSSING_OFFSET)
    
    # Check for upward crossing (from below line to above)
    crossing_upward = (prev_cy > LINE_Y + CROSSING_OFFSET and 
                       current_cy <= LINE_Y + CROSSING_OFFSET)
    
    if crossing_downward:
        # For downward crossing, expect positive dy movement
        direction_confidence = max(0, dy) / max(1, abs(dy))
        return True, 1, direction_confidence  # 1 = downward
    
    elif crossing_upward:
        # For upward crossing, expect negative dy movement
        direction_confidence = max(0, -dy) / max(1, abs(dy))
        return True, -1, direction_confidence  # -1 = upward
    
    return False, 0, 0  # Not crossing

def calculate_movement_metrics(current_pos, prev_pos):
    """Calculate speed and movement direction of an object"""
    if prev_pos is None:
        return 0, 0, 0
    
    cx1, cy1 = current_pos
    cx2, cy2 = prev_pos
    
    dx = cx1 - cx2  # Horizontal movement
    dy = cy1 - cy2  # Vertical movement
    speed = np.sqrt(dx**2 + dy**2)  # Total speed in pixels/frame
    
    return speed, dx, dy

def find_best_match(current_center, current_bbox, tracks):
    """
    Find which existing track best matches the current detection
    Returns: (best_track_index, match_score)
    """
    if not tracks:
        return -1, float('inf')
    
    best_idx = -1
    min_score = float('inf')
    x, y, w, h = current_bbox
    
    for idx, track in enumerate(tracks):
        if not track['centers'] or track['frames_lost'] > 5:
            continue
            
        # Get last known position and size of this track
        last_center = track['centers'][-1]
        last_bbox = track['bboxes'][-1] if track['bboxes'] else (0, 0, 0, 0)
        lx, ly, lw, lh = last_bbox
        
        # Calculate distance between current and last position
        distance = np.sqrt((current_center[0] - last_center[0])**2 + 
                          (current_center[1] - last_center[1])**2)
        
        # Calculate size difference
        size_diff = abs(w - lw) + abs(h - lh)
        
        # Combined score (distance is more important)
        score = distance * 0.7 + size_diff * 0.3
        
        # Check if this is the best match so far
        if distance < TRACKING_DISTANCE and score < min_score:
            min_score = score
            best_idx = idx
    
    return best_idx, min_score

def update_tracks():
    """Update tracks and remove ones that haven't been seen in a while"""
    updated_tracks = []
    
    for track in object_tracks:
        # Increase the "frames lost" counter
        track['frames_lost'] = track.get('frames_lost', 0) + 1
        
        # Keep track if we haven't lost it for too long AND it hasn't been counted yet
        if track['frames_lost'] <= MAX_FRAMES_LOST and track['id'] not in crossed_ids:
            updated_tracks.append(track)
        else:
            # Remove from crossed_ids if track is deleted
            if track['id'] in crossed_ids:
                crossed_ids.remove(track['id'])
    
    # Update the global object_tracks list
    object_tracks.clear()
    object_tracks.extend(updated_tracks)

def preprocess_mask(fg_mask):
    """Clean up the foreground mask to remove noise and fill gaps"""
    # Create kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    
    # Step 1: Remove small noise (opening operation)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERATIONS)
    
    # Step 2: Fill holes in detected objects (closing operation)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
    
    # Step 3: Remove shadows (KNN marks shadows as value 127)
    fg_mask[fg_mask == 127] = 0
    
    return fg_mask

def adjust_sensitivity_interactive(key):
    """Adjust sensitivity parameters based on keyboard input"""
    if key == ord('1'):  # Increase sensitivity (detect more)
        KNN_DIST_THRESHOLD = max(100, KNN_DIST_THRESHOLD - 50)
        print(f"↑ Increased sensitivity: KNN threshold = {KNN_DIST_THRESHOLD}")
        
    elif key == ord('2'):  # Decrease sensitivity (detect less)
        KNN_DIST_THRESHOLD = min(1000, KNN_DIST_THRESHOLD + 50)
        print(f"↓ Decreased sensitivity: KNN threshold = {KNN_DIST_THRESHOLD}")
        
    elif key == ord('3'):  # Faster adaptation to changes
        LEARNING_RATE = min(0.05, LEARNING_RATE * 2)
        print(f"↑ Faster adaptation: Learning rate = {LEARNING_RATE:.4f}")
        
    elif key == ord('4'):  # Slower adaptation (more stable)
        LEARNING_RATE = max(0.0001, LEARNING_RATE / 2)
        print(f"↓ Slower adaptation: Learning rate = {LEARNING_RATE:.4f}")
        
    elif key == ord('5'):  # Wider crossing tolerance
        CROSSING_OFFSET = min(30, CROSSING_OFFSET + 2)
        print(f"↑ Wider tolerance: Offset = {CROSSING_OFFSET} pixels")
        
    elif key == ord('6'):  # Narrower crossing tolerance
        CROSSING_OFFSET = max(2, CROSSING_OFFSET - 2)
        print(f"↓ Narrower tolerance: Offset = {CROSSING_OFFSET} pixels")
        
    elif key == ord('7'):  # Larger minimum object size
        MIN_WIDTH = min(200, MIN_WIDTH + 10)
        MIN_HEIGHT = min(200, MIN_HEIGHT + 10)
        print(f"↑ Larger minimum size: {MIN_WIDTH}x{MIN_HEIGHT} pixels")
        
    elif key == ord('8'):  # Smaller minimum object size
        MIN_WIDTH = max(20, MIN_WIDTH - 10)
        MIN_HEIGHT = max(20, MIN_HEIGHT - 10)
        print(f"↓ Smaller minimum size: {MIN_WIDTH}x{MIN_HEIGHT} pixels")

# ========== MAIN PROGRAM ==========
if __name__ == "__main__":
    # Video input - CHANGE THIS TO YOUR VIDEO FILE PATH
    video_path = r'D:\pythonProject1\video\my_video.mp4'
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {video_path}")
        exit()
    
    print("=" * 60)
    print("VEHICLE COUNTING SYSTEM")
    print("=" * 60)
    print("Controls:")
    print("  q - Quit program")
    print("  u/d - Move counting line up/down")
    print("  r - Reset counter to zero")
    print("  s - Save current frame and mask")
    print("  1/2 - Increase/decrease detection sensitivity")
    print("  3/4 - Increase/decrease adaptation speed")
    print("  5/6 - Increase/decrease crossing tolerance")
    print("  7/8 - Increase/decrease minimum object size")
    print("=" * 60)
    print("Starting video processing...")
    
    # Performance tracking
    fps_list = []
    
    # Main processing loop
    while True:
        # Read next frame from video
        ret, frame = video_capture.read()
        if not ret:
            print("End of video reached")
            break
        
        frame_count += 1
        height, width = frame.shape[:2]
        
        # Step 1: Apply background subtraction
        fg_mask = knn.apply(frame, learningRate=LEARNING_RATE)
        
        # Step 2: Clean up the mask
        fg_mask = preprocess_mask(fg_mask)
        
        # Step 3: Convert to binary (black and white) mask
        _, binary_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # Step 4: Find contours (connected regions) in the mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 5: Draw counting line and zone
        cv2.line(frame, (0, LINE_Y), (width, LINE_Y), LINE_COLOR, LINE_THICKNESS)
        
        # Draw the tolerance zone around the line
        cv2.line(frame, (0, LINE_Y - CROSSING_OFFSET), 
                 (width, LINE_Y - CROSSING_OFFSET), (100, 100, 255), 1)
        cv2.line(frame, (0, LINE_Y + CROSSING_OFFSET), 
                 (width, LINE_Y + CROSSING_OFFSET), (100, 100, 255), 1)
        
        # Step 6: Process detected contours
        current_detections = []
        
        for contour in contours:
            # Get bounding rectangle of contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip objects that don't look like vehicles
            if not is_valid_object(w, h, x, y, height):
                continue
            
            # Calculate center point
            center = rect_center(x, y, w, h)
            current_detections.append((center, (x, y, w, h)))
            
            # Draw bounding box with color based on size
            if w > 150 or h > 150:  # Large vehicle (truck/bus)
                color = (255, 165, 0)  # Orange
                label = "Large"
            elif w > 100 or h > 100:  # Medium vehicle
                color = (0, 255, 255)  # Yellow
                label = "Medium"
            else:  # Small vehicle
                color = (0, 255, 0)    # Green
                label = "Small"
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)  # Red center dot
        
        # Step 7: Match detections with existing tracks
        matched_indices = set()
        
        for center, bbox in current_detections:
            best_idx, score = find_best_match(center, bbox, object_tracks)
            
            if best_idx >= 0:
                # Update existing track
                track = object_tracks[best_idx]
                track['centers'].append(center)
                track['bboxes'].append(bbox)
                track['frames_lost'] = 0  # Reset lost counter
                track['last_update'] = frame_count
                
                # Get previous position for movement calculation
                if len(track['centers']) > 1:
                    prev_center = track['centers'][-2]
                else:
                    prev_center = center
                track['prev_center'] = prev_center
                
                matched_indices.add(best_idx)
                
                # Draw object ID
                x, y, w, h = bbox
                obj_id = track['id']
                cv2.putText(frame, f'#{obj_id}', (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Draw movement trail
                centers = track['centers'][-5:]  # Last 5 positions
                for i in range(1, len(centers)):
                    cv2.line(frame, centers[i-1], centers[i], (0, 255, 255), 1)
                
                # Check for line crossing
                if obj_id not in crossed_ids and len(track['centers']) >= MIN_TRACK_LENGTH:
                    cx, cy = center
                    prev_cx, prev_cy = prev_center
                    
                    is_crossing, direction, confidence = is_crossing_line(cy, prev_cy, cx, prev_cx)
                    speed, dx, dy = calculate_movement_metrics(center, prev_center)
                    
                    # Validate crossing
                    if (is_crossing and 
                        MIN_CROSSING_SPEED <= speed <= MAX_CROSSING_SPEED and
                        confidence > 0.3):
                        
                        # Count the vehicle
                        object_count += 1
                        crossed_ids.append(obj_id)
                        
                        # Visual feedback
                        cv2.circle(frame, center, 10, (255, 0, 255), 2)
                        cv2.putText(frame, f'COUNT #{object_count}', 
                                   (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                   (255, 0, 255), 2)
                        
                        print(f"Frame {frame_count}: Vehicle #{obj_id} counted! "
                              f"Speed: {speed:.1f} px/frame, Direction: {'down' if direction > 0 else 'up'}")
            else:
                # New object - create new track
                # We can directly modify object_id since it's in the global scope
                object_id += 1
                
                object_tracks.append({
                    'id': object_id,
                    'centers': [center],
                    'bboxes': [bbox],
                    'prev_center': center,
                    'frames_lost': 0,
                    'last_update': frame_count,
                    'first_seen': frame_count
                })
                
                # Draw new object label
                x, y, w, h = bbox
                cv2.putText(frame, f'NEW #{object_id}', (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Step 8: Update tracks (remove lost ones)
        update_tracks()
        
        # Step 9: Display statistics and information
        
        # Calculate FPS every 30 frames
        if frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - start_time)
            fps_list.append(fps)
            start_time = current_time
        
        # Create statistics display
        stats_text = [
            f"COUNT: {object_count}",
            f"Tracking: {len(object_tracks)} | Detected: {len(current_detections)}",
            f"Line Y: {LINE_Y} | Tolerance: {CROSSING_OFFSET}px",
            f"Sensitivity: {KNN_DIST_THRESHOLD} | Adapt Rate: {LEARNING_RATE:.4f}",
            f"Frame: {frame_count}"
        ]
        
        # Draw semi-transparent background for stats
        stats_bg = np.zeros((120, 450, 3), dtype=np.uint8)
        frame[10:130, 10:460] = cv2.addWeighted(frame[10:130, 10:460], 0.3, stats_bg, 0.7, 0)
        
        # Draw statistics text
        y_pos = 35
        for i, text in enumerate(stats_text):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)  # Count in yellow
            font_size = 0.8 if i == 0 else 0.5
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2 if i == 0 else 1)
            y_pos += 25
        
        # Draw FPS if available
        if fps_list:
            avg_fps = np.mean(fps_list[-10:]) if len(fps_list) >= 10 else np.mean(fps_list)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (width - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display control hints
        cv2.putText(frame, "1/2:Sens 3/4:Adapt 5/6:Tol", 
                   (width - 300, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, "7/8:Size u/d:Line r:Reset", 
                   (width - 300, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Step 10: Display windows
        cv2.imshow('Vehicle Counter', frame)
        
        # Display the mask for debugging
        mask_display = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        cv2.line(mask_display, (0, LINE_Y), (width, LINE_Y), LINE_COLOR, LINE_THICKNESS)
        cv2.imshow('Detection Mask', mask_display)
        
        # Step 11: Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Quit
            break
        elif key == ord('u'):  # Move line up
            LINE_Y = max(0, LINE_Y - 5)
            print(f"Line moved UP to Y={LINE_Y}")
        elif key == ord('d'):  # Move line down
            LINE_Y = min(height - 1, LINE_Y + 5)
            print(f"Line moved DOWN to Y={LINE_Y}")
        elif key == ord('r'):  # Reset counter
            object_count = 0
            crossed_ids = []
            object_tracks = []
            object_id = 0
            print("Counter reset to 0")
        elif key == ord('s'):  # Save current frame
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f'vehicle_count_{timestamp}_frame_{frame_count}.jpg', frame)
            cv2.imwrite(f'vehicle_count_{timestamp}_mask_{frame_count}.jpg', binary_mask)
            print(f"Frame {frame_count} saved")
        elif ord('1') <= key <= ord('8'):  # Sensitivity adjustments
            adjust_sensitivity_interactive(key)
    
    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total frames processed: {frame_count}")
    print(f"Total vehicles counted: {object_count}")
    
    if fps_list:
        print(f"Average processing speed: {np.mean(fps_list):.1f} FPS")
    
    print("\nFinal settings:")
    print(f"  Sensitivity (KNN threshold): {KNN_DIST_THRESHOLD}")
    print(f"  Adaptation rate: {LEARNING_RATE:.4f}")
    print(f"  Crossing tolerance: ±{CROSSING_OFFSET} pixels")
    print(f"  Minimum object size: {MIN_WIDTH}x{MIN_HEIGHT} pixels")
    print(f"  Counting line position: Y = {LINE_Y}")
    print("=" * 60)
    print("Thank you for using the Vehicle Counting System!")
