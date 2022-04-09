# Measures the angle between 2 lines
import cv2
import numpy as np
import math

def find_angle():
    a = points[0]
    b = points[1]
    c = points[2]
    m1 = slope(a, b)
    m2 = slope(a, c)

    # Use if 1st/green arrow is on the left
    # angle = math.atan((m1 - m2) / (1 + m1 * m2))
    # Use if 2nd/blue arrow is on the left
    angle = math.atan((m2 - m1) / (1 + m1 * m2))

    angle = round(math.degrees(angle))
    if angle < 0:
        angle = 180 + angle
    cv2.putText(image, str(angle), (a[0]-40,a[1]+40), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
    print(f'Angle = {angle} degrees')
    return angle

def slope(p1, p2):
   return (p2[1]-p1[1])/(p2[0]-p1[0])

image = cv2.imread('mtre4800-kawasaki-project/protractor.jpeg')

color_line_1 = (255,0,0)
color_line_2 = (0,255,0)
window_name = 'Angle Test'
cv2.namedWindow(window_name)

protractor_center = [364,365]
x1 = 193    # 95        Blue
y1 = 158    # 158
x2 = 638    # 95 599   Green
y2 = 366    # 366 230

points = [protractor_center, [x1, y1], [x2, y2]] # [start, 1st set, 2nd set] 30 deg
print("points:", points)

cv2.arrowedLine(image, tuple(points[0]), (x1, y1), color_line_1, 3)
cv2.arrowedLine(image, tuple(points[0]), (x2, y2), color_line_2, 3)
angle = find_angle()    # Degrees

cv2.imshow(window_name, image)
cv2.waitKey(0)
# cv2.destroyAllWindows()