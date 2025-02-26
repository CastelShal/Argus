import cv2
   
def draw_rect(frame, bbox, color=(0, 255, 0)):
  x, y, r, b = bbox
  cv2.rectangle(frame, (x, y), (r, b), color, 4)

def rgb_pre_processing(image):
      """Performs CLAHE on each RGB components and rebuilds final
      normalised RGB image - side note: improved face detection not recognition"""
      clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
      (B, G, R) = cv2.split(image)
      R = clahe.apply(R)
      G = clahe.apply(G)
      B = clahe.apply(B)
    
      filtered = cv2.merge([R, G, B])
      return filtered