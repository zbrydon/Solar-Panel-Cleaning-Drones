from collections import defaultdict
import cv2
import numpy as np
import copy
import math
import glob
import pandas



def main():
    global MIN_EXPECTED_CONTOR_COUNT

    global MIN_AREA

    global THRESHOLD
    global BLUR_VALUE

    # HSV colour thresholds for blue 
    global BLUE_MIN
    global BLUE_MAX

    global HOUGH_RHO
    global HOUGH_THETA
    global HOUGH_THRESHOLD
    global HOUGH_MIN_LINE_LENGTH
    global HOUGH_MAX_LINE_GAP
    MIN_EXPECTED_CONTOR_COUNT = 50

    MIN_AREA = 5000

    THRESHOLD = 60  # BINARY threshold
    BLUR_VALUE = 7  # GaussianBlur parameter

    # HSV colour thresholds for blue 
    BLUE_MIN = np.array([100, 120, 0],np.uint8)
    BLUE_MAX = np.array([140, 255, 255],np.uint8)

    HOUGH_RHO = 1
    HOUGH_THETA = math.pi/180.0
    HOUGH_THRESHOLD = 25
    HOUGH_MIN_LINE_LENGTH = 0 
    HOUGH_MAX_LINE_GAP = 0
    # file_name = './input/panel_09_100.jpg'

    # import_img = cv2.imread(file_name)

    # cell_count = file_name.split('_')[2].split('.')[0]

    # if not isinstance(int(cell_count), int):
    #     print("Invalid cell count")
    #     exit()
    # x = read_labels('./input/labels.csv')
    # print(x)

    # exit()

    images = [cv2.imread(file) for file in glob.glob("./input/*.jpg")]

    count = 0
    for import_img in images:
        # if count == 1:
        #     break
        file_name = glob.glob("./input/*.jpg")[count]
        cell_count = file_name.split('_')[2].split('.')[0]
        count += 1

        if not isinstance(int(cell_count), int):
            print("Invalid cell count")
            exit()

        cv2.imshow('Original', import_img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        blue_threshold = get_blue_component(import_img)
        blured_image = convert_image_to_binary(blue_threshold, True)
        threshold_image = get_threshold_image(blured_image)
        contours, length = get_contours(threshold_image)

        largest_contour = find_largest_contour(contours, length)
        filtered_contours = filter_contours(contours, MIN_AREA)

        iteration_count = 0
        while len(filtered_contours) < MIN_EXPECTED_CONTOR_COUNT:
            if iteration_count > 10:
                break
            iteration_count += 1

            MIN_AREA = MIN_AREA - 1000

            blured_image = convert_image_to_binary(blue_threshold, False)
            threshold_image = get_threshold_image(blured_image)
            contours, length = get_contours(threshold_image)

            largest_contour = find_largest_contour(contours, length)

            filtered_contours = filter_contours(contours, MIN_AREA)

        original_image_contours = draw_contours(import_img, filtered_contours)

        contour_image = draw_contours(np.zeros(import_img.shape, np.uint8), filtered_contours)

        result = create_mask(import_img, filtered_contours)

        cv2.imshow("Binary", threshold_image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


        cv2.imshow("Result", result)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        cv2.imwrite("contours.jpg", original_image_contours)

        final_image = draw_contours(result, filtered_contours)

        corners = get_corners(import_img, filtered_contours)
        if corners is not None:            
            for points in corners:
                for p in points:
                    cv2.circle(final_image, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        print(file_name)
        print('Observed cell count: ', len(filtered_contours))
        print('Expected cell count: ', cell_count)
        print('Cell count difference: ', int(cell_count) - len(filtered_contours) )
        print('Cell count detection percentage: ' + str(len(filtered_contours)/int(cell_count) * 100) + '%')
        print('====================================')

        cv2.imshow('', final_image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        
        cv2.destroyAllWindows()


def read_labels(file_name):
    """Reads in the labels from the given file."""
    column_names = ['filename', 'file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes', 'region_attributes']
    data = pandas.read_csv(file_name, names=column_names)
    return data


# Line intersection code from:
# https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv 

def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """ 

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections

def get_blue_component(img):
    # Convert image to HSV
    hsv_image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    cv2.imwrite('hsv_image.jpg', hsv_image)
    cv2.imwrite('extracted_blue.jpg', cv2.inRange(hsv_image, BLUE_MIN, BLUE_MAX))
    # Apply Blue threshold to identify blue colour in the image
    blue_threshold = cv2.cvtColor(cv2.inRange(hsv_image, BLUE_MIN, BLUE_MAX), cv2.COLOR_GRAY2RGB)
    cv2.imwrite('binary.jpg', blue_threshold)
    # blue_threshold = cv2.inRange(hsv_image, BLUE_MIN, BLUE_MAX)
    
    return blue_threshold

def convert_image_to_binary(img, blur = True):
    # convert the image into binary image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if blur:
        # Apply GaussianBlur to reduce noise
        blured_image = cv2.GaussianBlur(gray_image, (BLUR_VALUE, BLUR_VALUE), 0)
        return blured_image
    return gray_image

def get_threshold_image(img):
    #thresholding the frame
    ret, thresh = cv2.threshold(img, THRESHOLD, 255, cv2.THRESH_BINARY) 
    return thresh

def get_contours(img):
    # Find contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, len(contours)

def find_largest_contour(contours, length):
    if not length > 0:
        return None
    max_area = -1
    for i in range(length):
        area = cv2.contourArea(contours[i])
        if area > max_area:
            max_area = area
            current = i
    return contours[current]

def filter_contours(contours, min_area = 5000):
    # Filter contours by area
    filtered_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            filtered_contours.append(c)
    return filtered_contours

def draw_contours(img, contours, color = (0, 255, 0), thickness = 2):
    contour_image = copy.deepcopy(img)
    # Draw contours
    for c in contours:
        hull = cv2.convexHull(c)
        cv2.drawContours(contour_image, [hull], -1, color, thickness)
    return contour_image
        
def create_mask(import_img, contours):
    if len(contours) == 0:
        return import_img

    
    # Create a mask of the image
    mask = np.zeros(import_img.shape, np.uint8)
    for c in contours:
        new_image = cv2.drawContours(mask,[cv2.convexHull(c)], -1, (255,255,255), -1)
        new_image_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(new_image_gray, 100, 255, cv2.THRESH_BINARY)
        final = cv2.bitwise_and(import_img, import_img, mask = thresh)
    return final

def get_hough_lines(import_img, contour):
    # Apply Hough Transform

    temp_image = np.zeros(import_img.shape, np.uint8)
    cv2.drawContours(temp_image, [cv2.convexHull(contour)], -1, (0,255,0), 1)
    
    edges = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(edges, THRESHOLD, 255, cv2.THRESH_BINARY)
    
    lines = cv2.HoughLines(thresh, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD, np.array([]), HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP)
    
    return lines

def draw_hough_lines(img, lines):
    # Draw Hough Lines
    # final_lines = copy.deepcopy(result)
    # for line in lines:
    a,b,c = lines.shape
    for i in range(a):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0, y0 = a*rho, b*rho
        pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
        pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
        cv2.line(img, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
    return img

# Multiple lines and therefore multiple intersections are possible
# This code finds the groups of intersections that are k euclidean distance away from each other
# it returns a list of those groups
def find_intersection_groups(points, k):
    adj = defaultdict(list)
    n = len(points)
    groups = []
    
    for j in range(n):
        for i in range(j):
            
            x1, y1 = points[i]
            x2, y2 = points[j]
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 <= k ** 2:
                adj[i].append(j)
                adj[j].append(i)
                groups.append([points[i], points[j]])

    seen = set()
    def dfs(i):
        if i in seen:
            return
        seen.add(i)
        for nb in adj[i]:
            
            dfs(nb)

    ans = 0
    for i in range(n):
        if i not in seen:
            
            ans += 1
        dfs(i)
    return groups
# Calculates the center point of a group of intersections
def find_center(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length      

def get_corners(import_img, filtered_contours):
    corners = []
    for c in filtered_contours:

        lines = get_hough_lines(import_img, c)
        if lines is None or len(lines) == 0:
            return

        # final_lines = draw_hough_lines(contour_image, lines)

        # Groups lines into horizontal and vertical lines
        segmented = segment_by_angle_kmeans(lines)
        # Finds the intersections between groups of lines
        intersections = segmented_intersections(segmented)

        points = []
        for x in intersections:
            points.append(x[0])
        groups = find_intersection_groups(points, 10)

        final_points = []
        for g in groups:
            final_points.append(find_center(np.array(g)))
        corners.append(final_points)

    return corners

if __name__ == '__main__':
    main()