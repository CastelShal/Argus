import cv2

def get_bbox(detection, w, h):
  face = detection.location_data.relative_bounding_box
  xmin, ymin, width, height = face.xmin, face.ymin, face.width, face.height
  if xmin < 0 or ymin < 0 or width < 0 or height < 0:
     print("Faulty face box")
     return None
  
  xmin = int(xmin * w)
  ymin = int(ymin * h)
  width = int(w * width)
  height = int(h * height)
  return [xmin, ymin, width, height]
   
def draw_rect(frame, bbox, color=(0, 255, 0)):
  # x, y, r, b = int(bbox.left()), int(bbox.top()), int(bbox.right()), int(bbox.bottom())
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