import mss
import numpy as np
import cv2

def capture_minimap(region):
    """
    Capture and process the minimap region from the screen.
    
    Args:
        region (dict): Dictionary with keys 'top', 'left', 'width', 'height' defining the screen region
        
    Returns:
        numpy.ndarray: Processed edge-detected minimap image
    """
    with mss.mss() as sct:
        screenshot = sct.grab(region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Remove alpha channel
        
        # Create a circular mask
        height, width = img.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        center = (width // 2, height // 2)
        radius = min(width, height) // 2  # Use the smaller dimension for radius
        cv2.circle(mask, center, radius, 255, -1)  # Draw filled circle
        
        # Apply mask to the image
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        
        # Process the masked image
        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        return edges