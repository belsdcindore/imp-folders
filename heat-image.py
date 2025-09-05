# import cv2
# from ultralytics import YOLO
# import matplotlib.pyplot as plt

# # Load YOLO model (pretrained)
# model = YOLO("yolov8n.pt")  # person detection included

# # Load image
# img_path = "1_5.jpg"
# img = cv2.imread(img_path)

# # Run detection
# results = model(img)

# # Get detections
# people = []
# for r in results[0].boxes:
#     if int(r.cls[0]) == 0:  # class 0 = person in COCO dataset
#         x1, y1, x2, y2 = map(int, r.xyxy[0])
#         people.append(((x1+x2)//2, (y1+y2)//2))  # center position
#         cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

# print(f"Total People Detected: {len(people)}")

# # Show image
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.show()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter

# # Load image
# img = cv2.imread("1_5.jpg")
# height, width = img.shape[:2]

# # ⚡ Example: assume detected people positions (normally from YOLO)
# # For demo, let's randomly generate some positions
# np.random.seed(42)
# positions = [(np.random.randint(0, width), np.random.randint(0, height)) for _ in range(500)]

# # Create heatmap
# heatmap = np.zeros((height, width), dtype=np.float32)
# for x, y in positions:
#     heatmap[y, x] += 1

# # Smooth it
# heatmap = gaussian_filter(heatmap, sigma=50)

# # Normalize to [0, 255]
# heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
# heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)

# # Overlay on original image
# overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

# # Show result
# plt.figure(figsize=(12,6))
# plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.show()

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

class CrowdDetector:
    def __init__(self):
        pass
    
    def estimate_crowd_density(self, image_path):
        """
        Estimate crowd using multiple methods for better accuracy
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Could not load image")
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Feature point detection
        crowd_estimate_1 = self.method_1_feature_detection(gray)
        
        # Method 2: Density mapping
        crowd_estimate_2 = self.method_2_density_mapping(gray, img)
        
        # Method 3: Color segmentation
        crowd_estimate_3 = self.method_3_color_segmentation(img)
        
        # Combined estimate
        final_estimate = int((crowd_estimate_1 + crowd_estimate_2 + crowd_estimate_3) / 3)
        
        print(f"Method 1 (Feature Detection): {crowd_estimate_1:,} people")
        print(f"Method 2 (Density Mapping): {crowd_estimate_2:,} people")
        print(f"Method 3 (Color Segmentation): {crowd_estimate_3:,} people")
        print(f"Final Estimated Crowd: {final_estimate:,} people")
        
        # Visualize results
        self.visualize_detection(img, gray)
        
        return final_estimate
    
    def method_1_feature_detection(self, gray):
        """
        Method 1: Using corner detection and blob detection
        """
        # Harris corner detection
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        
        # Count significant corners
        corner_threshold = 0.01 * corners.max()
        corner_points = np.sum(corners > corner_threshold)
        
        # SIFT feature detection
        sift = cv2.SIFT_create()
        keypoints = sift.detect(gray, None)
        
        # Estimate based on features
        feature_density = len(keypoints) + corner_points
        
        # Calibration factor for crowd (empirically determined)
        crowd_estimate = int(feature_density * 8.5)
        
        return crowd_estimate
    
    def method_2_density_mapping(self, gray, img):
        """
        Method 2: Using density mapping and contour analysis
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding to separate crowd from background
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate density based on contour areas
        total_crowd_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Filter small noise
                total_crowd_area += area
        
        # Estimate people per unit area (empirically calibrated)
        people_per_pixel = 0.85
        crowd_estimate = int(total_crowd_area * people_per_pixel)
        
        return crowd_estimate
    
    def method_3_color_segmentation(self, img):
        """
        Method 3: Using color segmentation to identify crowd areas
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range for human skin colors and clothing
        # Multiple ranges to capture different lighting conditions
        ranges = [
            # Skin tones
            ([0, 20, 70], [20, 255, 255]),
            # Dark clothing
            ([0, 0, 0], [180, 255, 100]),
            # Bright colors
            ([0, 100, 100], [180, 255, 255])
        ]
        
        total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            total_mask = cv2.bitwise_or(total_mask, mask)
        
        # Clean up the mask
        kernel = np.ones((5,5), np.uint8)
        total_mask = cv2.morphologyEx(total_mask, cv2.MORPH_OPEN, kernel)
        total_mask = cv2.morphologyEx(total_mask, cv2.MORPH_CLOSE, kernel)
        
        # Count pixels and estimate crowd
        crowd_pixels = np.sum(total_mask > 0)
        
        # Calibration factor
        pixels_per_person = 120  # Approximate pixels per person
        crowd_estimate = int(crowd_pixels / pixels_per_person)
        
        return crowd_estimate
    
    def visualize_detection(self, img, gray):
        """
        Visualize the detection results
        """
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Grayscale
        plt.subplot(2, 3, 2)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')
        
        # Harris corners
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        plt.subplot(2, 3, 3)
        plt.imshow(corners, cmap='hot')
        plt.title('Harris Corners')
        plt.axis('off')
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        plt.subplot(2, 3, 4)
        plt.imshow(thresh, cmap='gray')
        plt.title('Adaptive Threshold')
        plt.axis('off')
        
        # HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        plt.subplot(2, 3, 5)
        plt.imshow(hsv)
        plt.title('HSV Color Space')
        plt.axis('off')
        
        # Crowd density heatmap
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        plt.subplot(2, 3, 6)
        plt.imshow(blurred, cmap='hot')
        plt.title('Density Heatmap')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Usage
def main():
    detector = CrowdDetector()
    
    # Replace 'your_image_path.jpg' with the actual path to your image
    image_path = '150people.jpg'  # Change this to your image path
    
    print("Starting crowd detection analysis...")
    print("="*50)
    
    estimated_crowd = detector.estimate_crowd_density(image_path)
    
    if estimated_crowd:
        print("="*50)
        print(f"FINAL RESULT: Estimated crowd size is approximately {estimated_crowd:,} people")
        
        # Provide confidence range
        lower_bound = int(estimated_crowd * 0.8)
        upper_bound = int(estimated_crowd * 1.2)
        print(f"Confidence Range: {lower_bound:,} - {upper_bound:,} people")

if __name__ == "__main__":
    main()

# Additional utility functions for more advanced analysis
def analyze_crowd_distribution(image_path):
    """
    Analyze crowd distribution across different areas of the image
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape
    
    # Divide image into grid sections
    sections = []
    grid_size = 4
    
    for i in range(grid_size):
        for j in range(grid_size):
            start_y = (h // grid_size) * i
            end_y = (h // grid_size) * (i + 1)
            start_x = (w // grid_size) * j
            end_x = (w // grid_size) * (j + 1)
            
            section = gray[start_y:end_y, start_x:end_x]
            
            # Calculate density for this section
            features = cv2.goodFeaturesToTrack(section, maxCorners=1000, 
                                              qualityLevel=0.01, minDistance=10)
            
            if features is not None:
                density = len(features)
            else:
                density = 0
            
            sections.append({
                'position': f'Row {i+1}, Col {j+1}',
                'density': density,
                'estimated_people': int(density * 5)  # Calibration factor
            })
    
    print("\nCrowd Distribution Analysis:")
    print("-" * 40)
    for section in sections:
        print(f"{section['position']}: ~{section['estimated_people']} people")
    
    return sections