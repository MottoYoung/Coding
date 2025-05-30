def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)
def get_bbox_width(bbox):
    x1, y1, x2, y2 = bbox
    return int(x2 - x1)
def measure_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
def measure_xy_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return x2 - x1, y2 - y1

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2),int(y2)