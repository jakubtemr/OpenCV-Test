import cv2
import numpy as np

# Load image
image = cv2.imread('data/2.jpg')

def detect_quadrilaterals(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=4)
    eroded = cv2.erode(dilated, kernel, iterations=2)
    
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    def is_quadrilateral(approx):
        return len(approx) == 4 and cv2.contourArea(approx) > 100
    
    quadrilateral_contours = [cnt for cnt in contours if is_quadrilateral(cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True))]
    
    return quadrilateral_contours, thresh

def find_holes(image, quadrilateral_contours, thresh):
    network_mask = np.zeros_like(thresh)
    cv2.drawContours(network_mask, quadrilateral_contours, -1, 255, thickness=cv2.FILLED)
    potential_holes_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    holes = []

    for hole in potential_holes_contours:
        hole_area = cv2.contourArea(hole)
        if hole_area == 0:
            continue
        # Create a mask for the current hole
        hole_mask = np.zeros_like(thresh)
        cv2.drawContours(hole_mask, [hole], -1, 255, thickness=cv2.FILLED)
        # Check if the hole is completely within the network
        if cv2.countNonZero(cv2.bitwise_and(hole_mask, network_mask)) > 0:
            if hole_area > 10000: 
                holes.append(hole)
    
    return holes


quadrilateral_contours, thresh = detect_quadrilaterals(image)
holes = find_holes(image, quadrilateral_contours, thresh)
output_image = image.copy()

# Draw quadrilateral contours (network) in green
for cnt in quadrilateral_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Draw holes in red
for cnt in holes:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Save the resulting image
cv2.imwrite('data/out.jpg', output_image)
