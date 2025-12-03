# -*- coding: utf-8 -*-


import cv2
import numpy as np

class HybridGMM_KNN_Corrector:
    """
    Hybrid mask generator that combines GMM and KNN masks
    Each method corrects the other's weaknesses
    """
    
    def __init__(self):
        # Initialize both background subtractors with different strengths
        self.gmm = cv2.createBackgroundSubtractorMOG2(
            history=300,      # Medium history - adapts moderately
            varThreshold=20,  # Moderate sensitivity
            detectShadows=True
        )
        
        self.knn = cv2.createBackgroundSubtractorKNN(
            history=500,      # Longer history - more stable
            dist2Threshold=500,  # Less sensitive
            detectShadows=True
        )
        
        # Frame buffer for temporal consistency
        self.frame_buffer = []
        self.buffer_size = 5
        
        # Morphology kernels
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        self.frame_count = 0
        
    def _get_individual_masks(self, frame):
        """Get masks from both methods with different learning rates"""
        # GMM with faster adaptation
        gmm_mask_raw = self.gmm.apply(frame, learningRate=0.005)
        
        # KNN with slower adaptation (more stable)
        knn_mask_raw = self.knn.apply(frame, learningRate=0.001)
        
        # Process GMM mask
        gmm_foreground = np.zeros_like(gmm_mask_raw)
        gmm_foreground[gmm_mask_raw == 255] = 255  # Foreground only
        gmm_shadow = np.zeros_like(gmm_mask_raw)
        gmm_shadow[gmm_mask_raw == 127] = 255      # Shadows only
        
        # Process KNN mask
        knn_foreground = np.zeros_like(knn_mask_raw)
        knn_foreground[knn_mask_raw == 255] = 255  # Foreground only
        knn_shadow = np.zeros_like(knn_mask_raw)
        knn_shadow[knn_mask_raw == 127] = 255      # Shadows only
        
        return gmm_foreground, gmm_shadow, knn_foreground, knn_shadow
    
    def _analyze_mask_confidence(self, mask):
        """Analyze confidence of mask regions"""
        # Calculate gradient magnitude - edges indicate uncertainty
        edges = cv2.Canny(mask, 50, 150)
        
        # Dilate edges to get uncertainty zones
        uncertainty = cv2.dilate(edges, self.kernel_small, iterations=2)
        
        # Calculate connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 4)
        
        # Create confidence map
        confidence = np.zeros_like(mask, dtype=np.float32)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Larger regions are more confident
            if area > 50:
                region_mask = (labels == i).astype(np.uint8)
                # Confidence based on area and compactness
                confidence[region_mask > 0] = min(1.0, area / 1000.0)
        
        # Reduce confidence in uncertain areas
        confidence[uncertainty > 0] *= 0.5
        
        return confidence
    
    def _get_temporal_mask(self, frame):
        """Get temporal consistency mask using frame buffer"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(gray)
        
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        if len(self.frame_buffer) >= 3:
            # Calculate median of recent frames
            median_frame = np.median(self.frame_buffer, axis=0).astype(np.uint8)
            
            # Frame difference from median
            diff = cv2.absdiff(gray, median_frame)
            _, temporal_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            
            # Temporal mask is more confident for consistent motion
            temporal_mask = cv2.dilate(temporal_mask, self.kernel_small, iterations=1)
            return temporal_mask
        
        return np.zeros_like(gray)
    
    def generate_mask(self, frame):
        """
        Generate hybrid mask where GMM and KNN correct each other
        """
        self.frame_count += 1
        height, width = frame.shape[:2]
        
        # Get individual masks
        gmm_fg, gmm_shadow, knn_fg, knn_shadow = self._get_individual_masks(frame)
        
        # Get temporal consistency mask
        temporal_mask = self._get_temporal_mask(frame)
        
        # Analyze confidence of each mask
        gmm_confidence = self._analyze_mask_confidence(gmm_fg)
        knn_confidence = self._analyze_mask_confidence(knn_fg)
        
        # Step 1: Areas where both agree on foreground (HIGH CONFIDENCE)
        both_foreground = cv2.bitwise_and(gmm_fg, knn_fg)
        
        # Step 2: Areas where GMM says foreground but KNN says shadow (GMM may be detecting noise)
        gmm_fg_knn_shadow = cv2.bitwise_and(gmm_fg, knn_shadow)
        
        # Step 3: Areas where KNN says foreground but GMM says shadow (KNN may be detecting ghosts)
        knn_fg_gmm_shadow = cv2.bitwise_and(knn_fg, gmm_shadow)
        
        # Step 4: Intelligent combination
        hybrid_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Case A: Both agree on foreground - definitely foreground
        hybrid_mask[both_foreground > 0] = 255
        
        # Case B: GMM foreground, KNN shadow - use confidence to decide
        gmm_confident = gmm_confidence > 0.7
        gmm_fg_areas = (gmm_fg > 0) & (gmm_confident) & (gmm_fg_knn_shadow == 0)
        hybrid_mask[gmm_fg_areas] = 255
        
        # Case C: KNN foreground, GMM shadow - use confidence to decide
        knn_confident = knn_confidence > 0.7
        knn_fg_areas = (knn_fg > 0) & (knn_confident) & (knn_fg_gmm_shadow == 0)
        hybrid_mask[knn_fg_areas] = 255
        
        # Case D: Temporal consistency - if motion is consistent, trust it
        hybrid_mask[temporal_mask > 0] = 255
        
        # Case E: Areas with high confidence in either method
        high_confidence = (gmm_confidence > 0.8) | (knn_confidence > 0.8)
        high_conf_fg = ((gmm_fg > 0) | (knn_fg > 0)) & high_confidence
        hybrid_mask[high_conf_fg] = 255
        
        # Step 5: Post-processing
        
        # Fill holes in foreground objects
        hybrid_mask = cv2.morphologyEx(hybrid_mask, cv2.MORPH_CLOSE, self.kernel_medium)
        
        # Remove small noise
        hybrid_mask = cv2.morphologyEx(hybrid_mask, cv2.MORPH_OPEN, self.kernel_small)
        
        # Remove very small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hybrid_mask, 4)
        if num_labels > 1:
            cleaned = np.zeros_like(hybrid_mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] > 50:  # Minimum area
                    cleaned[labels == i] = 255
            hybrid_mask = cleaned
        
        # Ensure binary
        _, hybrid_mask = cv2.threshold(hybrid_mask, 127, 255, cv2.THRESH_BINARY)
        
        return hybrid_mask


class SmartHybridCorrector:
    """
    Simplified but smart hybrid corrector
    Focuses on the key correction logic
    """
    
    def __init__(self):
        self.gmm = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16)
        self.knn = cv2.createBackgroundSubtractorKNN(history=400, dist2Threshold=400)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
    def generate_mask(self, frame):
        # Get masks from both methods
        gmm_mask = self.gmm.apply(frame, learningRate=0.01)
        knn_mask = self.knn.apply(frame, learningRate=0.005)
        
        # Binary masks (ignore shadows)
        _, gmm_bin = cv2.threshold(gmm_mask, 250, 255, cv2.THRESH_BINARY)
        _, knn_bin = cv2.threshold(knn_mask, 250, 255, cv2.THRESH_BINARY)
        
        # CORRECTION LOGIC:
        
        # 1. Areas where both agree - keep them
        both_agree = cv2.bitwise_and(gmm_bin, knn_bin)
        
        # 2. GMM says foreground but KNN doesn't - could be GMM noise
        gmm_only = cv2.subtract(gmm_bin, knn_bin)
        
        # 3. KNN says foreground but GMM doesn't - could be KNN catching something GMM missed
        knn_only = cv2.subtract(knn_bin, gmm_bin)
        
        # Apply corrections:
        hybrid_mask = both_agree.copy()
        
        # For GMM-only areas, check if they're large enough (not noise)
        gmm_contours, _ = cv2.findContours(gmm_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in gmm_contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Large enough, probably real
                cv2.drawContours(hybrid_mask, [contour], -1, 255, -1)
        
        # For KNN-only areas, check temporal consistency
        knn_contours, _ = cv2.findContours(knn_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in knn_contours:
            area = cv2.contourArea(contour)
            if area > 150:  # KNN tends to be more conservative, so trust larger areas
                cv2.drawContours(hybrid_mask, [contour], -1, 255, -1)
        
        # Clean up
        hybrid_mask = cv2.morphologyEx(hybrid_mask, cv2.MORPH_CLOSE, self.kernel)
        hybrid_mask = cv2.medianBlur(hybrid_mask, 3)
        
        return hybrid_mask


class WeightedHybridCorrector:
    """
    Weighted combination of GMM and KNN masks
    """
    
    def __init__(self):
        self.gmm = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=18)
        self.knn = cv2.createBackgroundSubtractorKNN(history=350, dist2Threshold=450)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Weights can be adjusted based on scene
        self.gmm_weight = 0.6
        self.knn_weight = 0.4
        
    def generate_mask(self, frame):
        # Get masks
        gmm_mask = self.gmm.apply(frame, learningRate=0.005)
        knn_mask = self.knn.apply(frame, learningRate=0.002)
        
        # Binary masks
        _, gmm_bin = cv2.threshold(gmm_mask, 240, 255, cv2.THRESH_BINARY)
        _, knn_bin = cv2.threshold(knn_mask, 240, 255, cv2.THRESH_BINARY)
        
        # Convert to float for weighted combination
        gmm_float = gmm_bin.astype(np.float32) / 255.0
        knn_float = knn_bin.astype(np.float32) / 255.0
        
        # Weighted combination
        combined_float = (gmm_float * self.gmm_weight + knn_float * self.knn_weight)
        
        # Normalize and threshold
        combined_float = cv2.normalize(combined_float, None, 0, 1, cv2.NORM_MINMAX)
        
        # Adaptive threshold based on combined confidence
        threshold = 0.5 * (self.gmm_weight + self.knn_weight)
        _, hybrid_mask = cv2.threshold((combined_float * 255).astype(np.uint8), 
                                      threshold * 255, 255, cv2.THRESH_BINARY)
        
        # Post-processing
        hybrid_mask = cv2.morphologyEx(hybrid_mask, cv2.MORPH_CLOSE, self.kernel)
        
        return hybrid_mask


# Test function
def test_hybrid_corrector():
    """Test the hybrid corrector"""
    print("Testing GMM-KNN Hybrid Corrector")
    print("Press 'q' to quit, '1-3' to switch methods")
    
    # Create correctors
    correctors = {
        "Smart Corrector": SmartHybridCorrector(),
        "Weighted Corrector": WeightedHybridCorrector(),
        "Advanced Corrector": HybridGMM_KNN_Corrector()
    }
    
    current_method = "Smart Corrector"
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Generate mask with current method
        mask = correctors[current_method].generate_mask(frame)
        
        # Get individual masks for comparison
        gmm = cv2.createBackgroundSubtractorMOG2(history=200).apply(frame)
        knn = cv2.createBackgroundSubtractorKNN(history=400).apply(frame)
        
        _, gmm_bin = cv2.threshold(gmm, 250, 255, cv2.THRESH_BINARY)
        _, knn_bin = cv2.threshold(knn, 250, 255, cv2.THRESH_BINARY)
        
        # Create comparison display
        display = np.zeros((240*2, 320*2, 3), dtype=np.uint8)
        
        # Row 1: Original and GMM
        display[0:240, 0:320] = frame
        display[0:240, 320:640] = cv2.cvtColor(gmm_bin, cv2.COLOR_GRAY2BGR)
        
        # Row 2: KNN and Hybrid
        display[240:480, 0:320] = cv2.cvtColor(knn_bin, cv2.COLOR_GRAY2BGR)
        display[240:480, 320:640] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        cv2.putText(display, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "GMM Mask", (330, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "KNN Mask", (10, 270), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Hybrid ({current_method})", (330, 270), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('GMM-KNN Hybrid Comparison', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_method = "Smart Corrector"
        elif key == ord('2'):
            current_method = "Weighted Corrector"
        elif key == ord('3'):
            current_method = "Advanced Corrector"
    
    cap.release()
    cv2.destroyAllWindows()


# Simple demo
def simple_demo():
    """Simple demo showing correction in action"""
    print("Simple GMM-KNN Correction Demo")
    print("Press 'q' to quit")
    
    corrector = SmartHybridCorrector()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get hybrid mask
        mask = corrector.generate_mask(frame)
        
        # Overlay mask on original
        overlay = frame.copy()
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_color[:, :, 0] = 0  # Remove blue
        mask_color[:, :, 1] = 0  # Remove green
        
        # Blend
        overlay = cv2.addWeighted(overlay, 0.7, mask_color, 0.3, 0)
        
        cv2.imshow('Hybrid Correction Result', overlay)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("GMM-KNN Hybrid Mask Corrector")
    print("=" * 50)
    print("1. Test with comparison view")
    print("2. Simple demo")
    
    choice = input("Enter choice (1-2): ")
    
    if choice == '1':
        test_hybrid_corrector()
    elif choice == '2':
        simple_demo()