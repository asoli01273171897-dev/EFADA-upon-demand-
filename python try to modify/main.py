# -- coding: utf-8 --


import cv2
import numpy as np
import time
import math
import sys

# -------------------------
# PARAMETERS (tune these)
# -------------------------
MAX_WIDTH = 320             # resized width (keep small for speed)
MAX_HEIGHT = 240            # resized height (keep small for speed)
ALPHA = 0.002               # learning rate for weights (small)
INIT_VARIANCE = 225.0       # initial variance for new components (15^2)
MATCH_THRESH = 9.0          # multiplier on variance for match (squared Mahalanobis threshold)
MAX_COMPONENTS = 4          # maximum Gaussians per pixel
BG_WEIGHT_THRESHOLD = 0.7   # cumulative weight threshold to consider background
MIN_VARIANCE = 4.0          # min allowed variance
MAX_VARIANCE = 400.0        # max allowed variance
MIN_WEIGHT_PRUNE = 1e-6     # weights below this will be pruned
SHOW_FPS = True
# -------------------------

# -------------------------
# Linked-list Gaussian class
# -------------------------
class GaussianNode:
    __slots__ = ("mean", "variance", "weight", "Next", "Previous")
    def __init__(self, mean=None, variance=INIT_VARIANCE, weight=0.0):
        # mean: numpy array length 3 (B,G,R)
        self.mean = np.array(mean, dtype=np.float32) if mean is not None else np.zeros((3,), dtype=np.float32)
        self.variance = float(variance)
        self.weight = float(weight)
        self.Next = None
        self.Previous = None

# -------------------------
# Per-pixel linked list container
# -------------------------
class PixelNode:
    __slots__ = ("pixel_s", "pixel_r", "no_of_components")
    def __init__(self):
        # pixel_s: head (start) node
        # pixel_r: tail (rear) node
        self.pixel_s = None
        self.pixel_r = None
        self.no_of_components = 0

    # Insert gaussian at the end (tail)
    def insert_end(self, gnode: GaussianNode):
        if self.pixel_s is None:
            self.pixel_s = self.pixel_r = gnode
            gnode.Previous = None
            gnode.Next = None
        else:
            tail = self.pixel_r
            tail.Next = gnode
            gnode.Previous = tail
            gnode.Next = None
            self.pixel_r = gnode
        self.no_of_components += 1

    # Remove a node (unlink from list)
    def remove_node(self, gnode: GaussianNode):
        if gnode.Previous is None and gnode.Next is None:
            # only node
            self.pixel_s = None
            self.pixel_r = None
        elif gnode.Previous is None:
            # head
            self.pixel_s = gnode.Next
            self.pixel_s.Previous = None
        elif gnode.Next is None:
            # tail
            self.pixel_r = gnode.Previous
            self.pixel_r.Next = None
        else:
            # middle
            gnode.Previous.Next = gnode.Next
            gnode.Next.Previous = gnode.Previous
        gnode.Next = None
        gnode.Previous = None
        self.no_of_components -= 1

    # Normalize all weights to sum to 1 (if sum>0)
    def normalize_weights(self):
        s = 0.0
        ptr = self.pixel_s
        while ptr is not None:
            s += ptr.weight
            ptr = ptr.Next
        if s > 0:
            ptr = self.pixel_s
            while ptr is not None:
                ptr.weight = ptr.weight / s
                ptr = ptr.Next

    # Sort nodes by (weight/variance) descending by swapping nodes in linked list
    def resort_by_score(self):
        # Convert to list, sort, rebuild linked list for simplicity and safety
        nodes = []
        p = self.pixel_s
        while p is not None:
            nodes.append(p)
            p = p.Next
        if not nodes:
            return
        nodes.sort(key=lambda x: (x.weight / (x.variance + 1e-9)), reverse=True)
        # rebuild
        self.pixel_s = nodes[0]
        self.pixel_s.Previous = None
        for i in range(len(nodes)-1):
            nodes[i].Next = nodes[i+1]
            nodes[i+1].Previous = nodes[i]
        self.pixel_r = nodes[-1]
        self.pixel_r.Next = None

# -------------------------
# Helper functions
# -------------------------
def create_gaussian(pixel, weight=0.001):
    # pixel: [B,G,R] array or list
    return GaussianNode(mean=np.array(pixel, dtype=np.float32), variance=INIT_VARIANCE, weight=weight)

def denoise(frame, max_w=MAX_WIDTH, max_h=MAX_HEIGHT):
    h, w = frame.shape[:2]
    scale_w = max_w / float(w)
    scale_h = max_h / float(h)
    scale = min(scale_w, scale_h, 1.0)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    if (new_w, new_h) != (w, h):
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    frame = cv2.medianBlur(frame, 3)
    frame = cv2.GaussianBlur(frame, (3,3), 0)
    return frame

# Compute squared Mahalanobis distance assuming diagonal covariance with identical variance
def mahalanobis_sq(comp: GaussianNode, pixel):
    d = pixel.astype(np.float32) - comp.mean
    return float((d * d).sum())  # sum squared diff across channels

# -------------------------
# Frame processing (per-pixel linked list update)
# -------------------------
class LinkedListGMM:
    def __init__(self, height, width):
        self.h = height
        self.w = width
        # create grid of PixelNode
        self.grid = [[PixelNode() for _ in range(width)] for __ in range(height)]

    def process_frame(self, frame_bgr):
        """
        frame_bgr: numpy array shape (h,w,3), dtype=uint8
        returns: mask (h,w) uint8: 255 foreground, 0 background
        """
        h, w = frame_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # iterate rows (cache friendly)
        for y in range(h):
            row = frame_bgr[y]
            grid_row = self.grid[y]
            for x in range(w):
                pixel = row[x]  # BGR uint8
                pn = grid_row[x]
                self.update_pixel(pn, pixel, mask, y, x)
        return mask

    def update_pixel(self, pn: PixelNode, pixel, mask, y, x):
        """
        Update the pixel's Gaussian linked-list with new pixel value.
        Decide if pixel is background or foreground and set mask accordingly.
        """
        pixel_f = pixel.astype(np.float32)
        matched = None
        ptr = pn.pixel_s

        # Step 1: If no components yet, create first component with weight alpha
        if ptr is None:
            g = create_gaussian(pixel_f, weight=ALPHA)
            pn.insert_end(g)
            # normalize
            pn.normalize_weights()
            mask[y, x] = 255  # newly created component => foreground (not stable yet)
            return

        # Step 2: Decay all weights by (1 - ALPHA)
        ptr = pn.pixel_s
        while ptr is not None:
            ptr.weight *= (1.0 - ALPHA)
            ptr = ptr.Next

        # Step 3: Try to match any component
        ptr = pn.pixel_s
        while ptr is not None:
            mdsq = mahalanobis_sq(ptr, pixel_f)
            # match condition: squared distance < MATCH_THRESH * variance
            if mdsq < (MATCH_THRESH * ptr.variance):
                matched = ptr
                break
            ptr = ptr.Next

        # Step 4: Update matched component or create new one
        if matched is not None:
            # compute rho = ALPHA / matched.weight_clamped to adapt mean/var
            # but to avoid huge rho we clamp denominator
            denom = max(matched.weight, 1e-6)
            rho = ALPHA / denom

            # update mean
            matched.mean = (1.0 - rho) * matched.mean + rho * pixel_f

            # update variance using squared distance (instantaneous estimate)
            d2 = mahalanobis_sq(matched, pixel_f)
            matched.variance = (1.0 - rho) * matched.variance + rho * d2
            # clamp variance
            matched.variance = max(MIN_VARIANCE, min(matched.variance, MAX_VARIANCE))

            # increase matched weight
            matched.weight += ALPHA
        else:
            # create new component with small weight ALPHA
            new_g = create_gaussian(pixel_f, weight=ALPHA)
            pn.insert_end(new_g)
            matched = new_g

        # Step 5: Prune tiny-weight components
        # compute sum for renormalize later
        # remove nodes with weight < MIN_WEIGHT_PRUNE
        ptr = pn.pixel_s
        to_remove = []
        while ptr is not None:
            nxt = ptr.Next
            if ptr.weight < MIN_WEIGHT_PRUNE:
                to_remove.append(ptr)
            ptr = nxt
        for r in to_remove:
            pn.remove_node(r)

        # Step 6: Normalize weights (so sum = 1)
        pn.normalize_weights()

        # Step 7: Enforce max components: drop smallest-weight components
        while pn.no_of_components > MAX_COMPONENTS:
            # find node with smallest weight and remove it
            ptr = pn.pixel_s
            smallest = ptr
            while ptr is not None:
                if ptr.weight < smallest.weight:
                    smallest = ptr
                ptr = ptr.Next
            pn.remove_node(smallest)

        # Step 8: Resort nodes by score (weight/variance)
        pn.resort_by_score()

        # Step 9: Decide background membership:
        # accumulate weights in sorted order until cumulative >= BG_WEIGHT_THRESHOLD
        cum = 0.0
        bg_nodes = set()
        idx = 0
        ptr = pn.pixel_s
        while ptr is not None:
            cum += ptr.weight
            bg_nodes.add(ptr)  # store ptr object
            ptr = ptr.Next
            idx += 1
            if cum >= BG_WEIGHT_THRESHOLD:
                break

        # If the matched component (the one adjusted or created) is inside bg_nodes => background
        if matched in bg_nodes:
            mask[y, x] = 0
        else:
            mask[y, x] = 255

# -------------------------
# Main program
# -------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera.")
        return

    # grab a frame to determine resized dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: cannot read camera frame.")
        cap.release()
        return

    frame = denoise(frame, MAX_WIDTH, MAX_HEIGHT)
    h, w = frame.shape[:2]
    print(f"Working on frame size: {w}x{h}")

    # Create model
    model = LinkedListGMM(h, w)

    fps_counter = 0
    start_time = time.time()

    print("Press ESC to exit.")
    while True:
        ret, raw = cap.read()
        if not ret:
            print("Frame read failed - exiting.")
            break

        frame_small = denoise(raw, MAX_WIDTH, MAX_HEIGHT)
        # process per-pixel
        mask = model.process_frame(frame_small)

        # morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        cv2.imshow("Frame (scaled)", frame_small)
        cv2.imshow("MOG Linked-List Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        # fps counting
        fps_counter += 1
        if SHOW_FPS and fps_counter % 30 == 0:
            elapsed = time.time() - start_time
            fps = fps_counter / elapsed if elapsed > 0 else 0.0
            cv2.setWindowTitle("MOG Linked-List Mask", f"MOG Linked-List Mask - approx FPS: {fps:.1f}")
            fps_counter = 0
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
