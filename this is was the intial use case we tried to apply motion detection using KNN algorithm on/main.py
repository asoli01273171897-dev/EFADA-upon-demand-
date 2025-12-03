import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur

def motion_detect(prev_frame, next_frame, knn):
    p_gray = preprocess(prev_frame)
    n_gray = preprocess(next_frame)

    difference = cv2.absdiff(p_gray, n_gray)
    threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]

    dilated = cv2.dilate(threshold, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    valid_contours = []  # Store valid contours for drawing
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Filter out very small contours (noise)
        if area > 500:  # Minimum area threshold
            (x, y, w, h) = cv2.boundingRect(contour)
            features.append([w, h])
            valid_contours.append(contour)

    # Only make predictions if there are features
    if features:
        predictions = knn.predict(features)
    else:
        predictions = []
        return next_frame  # Return unchanged if no contours

    # Draw bounding boxes and labels
    for contour, prediction in zip(valid_contours, predictions):
        (x, y, w, h) = cv2.boundingRect(contour)
        label = 'Moving' if prediction == 1 else 'Stationary'
        
        # Draw rectangle
        color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
        cv2.rectangle(next_frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        cv2.putText(next_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 2)

    return next_frame

# Load or create sample frames for testing
def create_test_frames():
    # Create a black background
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add a stationary rectangle in both frames
    cv2.rectangle(frame1, (100, 100), (150, 150), (255, 0, 0), -1)
    cv2.rectangle(frame2, (100, 100), (150, 150), (255, 0, 0), -1)
    
    # Add a moving rectangle only in second frame
    cv2.rectangle(frame2, (300, 200), (350, 250), (0, 0, 255), -1)
    
    return frame1, frame2

# Main execution
if __name__ == "__main__":
    # Create test frames
    prev_frame, next_frame = create_test_frames()
    
    # Create better training data
    # Features: [width, height], Labels: 1=Moving, 0=Stationary
    features = [
        [50, 50],   # Large moving object
        [30, 30],   # Medium moving object  
        [15, 15],   # Small moving object
        [60, 40],   # Large stationary object
        [20, 20],   # Small stationary object
        [35, 35],   # Medium stationary object
    ]
    
    labels = [1, 1, 1, 0, 0, 0]  # 1: Moving, 0: Stationary
    
    # Create and train KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(features, labels)
    
    # Run motion detection
    output_frame = motion_detect(prev_frame.copy(), next_frame.copy(), knn)
    
    # Display results
    cv2.imshow('Frame 1 (Previous)', prev_frame)
    cv2.imshow('Frame 2 (Current)', next_frame)
    cv2.imshow('Motion Detection Output', output_frame)
    
    print("Blue rectangle: Stationary object (present in both frames)")
    print("Red rectangle: Moving object (only in current frame)")
    print("\nPress any key to exit...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()