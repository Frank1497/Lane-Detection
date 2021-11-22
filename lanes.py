import cv2 as cv
import numpy as np


def canny(image):                                           #for edge detection
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)       #easier for AI to understand
    blur = cv.GaussianBlur(gray, (5, 5), 0)                 #reason to blur is to smoothen the edges
    canny = cv.Canny(blur, 50, 150)                         #canny is a edge detection function
    return canny


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = (image.shape[0])
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_interest(image, averaged_lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    if len(left_fit) and len(right_fit):
        left_fit_average  = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line  = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        averaged_lines = [left_line, right_line]
        return averaged_lines


def display_lines(image, lines):                            #to put lines equivalent to lane lines
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    mask = np.zeros_like(image)

    triangle = np.array([[(200, height), (550, 250), (1100, height),]], np.int32)
    cv.fillPoly(mask, triangle, 255)                         #highlighting triangle
    masked_image = cv.bitwise_and(image, mask)               #using binary numbers able to crop necessary section
    return masked_image

# image = cv.imread('test_image.jpg')
# lane_image = np.copy(image)                                   #copying image kinda useful optional
# canny = canny(lane_image)
# cropped_image = region_of_interest(canny)
# lines = cv.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_interest(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combine = cv.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv.imshow('result', cropped_image)
# cv.imshow('resul', canny)
# cv.waitKey(0)


cap = cv.VideoCapture('test.mp4')
while(cap.isOpened()):
    _, frame = cap.read()
    cann = canny(frame)
    cropped_image = region_of_interest(cann)
    lines = cv.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_interest(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combine = cv.addWeighted(frame, 0.8, line_image, 1, 1)
    cv.imshow('result', combine)
    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


    