import cv2
import numpy as np
import sys


def finding_lanes(img):
    # canny
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)

    low_threshold = 50
    high_threshold = 150
    canny = cv2.Canny(gray, low_threshold, high_threshold)

    # region of interest
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)
    region = np.array([[
        (0, height),
        (width / 2, height / 2),
        (width, height), ]], np.int32)

    match_mask_color = 255
    cv2.fillPoly(mask, region, match_mask_color)
    cropped_img = cv2.bitwise_and(canny, mask)

    # HoughLines parameters
    lines = cv2.HoughLinesP(
        cropped_img,
        rho=2,
        theta=np.pi / 180,
        threshold=115,
        lines=np.array([]),
        minLineLength=30,
        maxLineGap=25
    )

    # average_slope_intercept
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # add more weight to longer lines
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_points(img, left_fit_average)
    right_line = make_points(img, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines


def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1*3/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]


def draw_lines(img, lines):
    detected_lines = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(detected_lines, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return detected_lines


def main(videoName):
    video = cv2.VideoCapture(videoName)
    while(video.isOpened()):
        _, frame = video.read()
        lines = finding_lanes(frame)
        detected_lines = draw_lines(frame, lines)
        combo_image = cv2.addWeighted(frame, 0.8, detected_lines, 1, 1)
        cv2.imshow("result", combo_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    videoName = sys.argv[1]
    main(videoName)
