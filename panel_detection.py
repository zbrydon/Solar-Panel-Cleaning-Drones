from collections import defaultdict
import cv2
import numpy as np
import copy
import math
import glob
import pandas
from functools import reduce
import operator
import math


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
    MIN_EXPECTED_CONTOR_COUNT = 5

    MIN_AREA = 1000

    CORNER_GROUP = 20

    THRESHOLD = 60  # BINARY threshold
    BLUR_VALUE = 7  # GaussianBlur parameter

    # HSV colour thresholds for blue 
    BLUE_MIN = np.array([70, 70, 70],np.uint8)
    BLUE_MAX = np.array([140, 255, 240],np.uint8)

    HOUGH_RHO = 1
    HOUGH_THETA = math.pi/180.0
    HOUGH_THRESHOLD = 25
    HOUGH_MIN_LINE_LENGTH = 0 
    HOUGH_MAX_LINE_GAP = 0

    images = [cv2.imread(file) for file in glob.glob("./input/*.jpg")]

    indexs = [2,5,9]

    count = 0
    for import_img in images:
        file_name = glob.glob("./input/*.jpg")[count]
        panel_count = file_name.split('_')[2].split('.')[0]
        file_no = file_name.split('_')[1]
        if int(file_no) not in indexs:
            
            count += 1
            continue

        count += 1

        if not isinstance(int(panel_count), int):
            print("Invalid cell count")
            exit()

        cv2.imshow('Original', import_img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        blue_threshold = get_blue_component(import_img)
        # cv2.imwrite("./output/last.jpg", blue_threshold)
        blured_image = convert_image_to_binary(blue_threshold, False)
        threshold_image = get_threshold_image(blured_image)
        contours, length = get_contours(threshold_image)

        largest_contour = find_largest_contour(contours, length)
        filtered_contours = filter_contours(contours, MIN_AREA)

        iteration_count = 0
        while len(filtered_contours) < MIN_EXPECTED_CONTOR_COUNT:
            if iteration_count > 10:
                break
            iteration_count += 1

            MIN_AREA = MIN_AREA - 500

            blured_image = convert_image_to_binary(blue_threshold, True)
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


        cv2.imshow("original_image_contours", original_image_contours)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        final_image = draw_contours(result, filtered_contours)

        corners = get_corners_v2(import_img, filtered_contours,CORNER_GROUP)
        
        if corners is not None:      
            for points in corners:
                for i in range(1, len(points)):
                    cv2.circle(final_image, (int(points[i][0]), int(points[i][1])), 3, (0, 0, 255), -1)

        print(file_name)
        print('Observed panel count: ', len(filtered_contours))
        print('Expected panel count: ', panel_count)
        print('Panel count difference: ', int(panel_count) - len(filtered_contours) )
        print('Panel count detection percentage: ' + str(len(filtered_contours)/int(panel_count) * 100) + '%')
        print('====================================')

        cv2.imshow('Final', final_image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        # extract_panels(corners, import_img)


        fn = "./output/panel_%d_original.jpg" % count
        fn1 = "./output/panel_%d_blue_threshold.jpg" % count
        fn2 = "./output/panel_%d_blured.jpg" % count
        fn3 = "./output/panel_%d_threshold.jpg" % count
        fn4 = "./output/panel_%d_contours.jpg" % count
        fn5 = "./output/panel_%d_final.jpg" % count
        
        # cv2.imwrite(fn, import_img)
        # cv2.imwrite(fn1, blue_threshold)
        # cv2.imwrite(fn2, blured_image)
        # cv2.imwrite(fn3, threshold_image)
        # cv2.imwrite(fn4, original_image_contours)
        # cv2.imwrite(fn5, final_image)
        
        cv2.destroyAllWindows()


def sort_corners(corners):
    """Sorts the corners in clockwise order."""

    # FIX ME: This is a very naive way of sorting the corners.

    coords = [(0, 1), (1, 0), (1, 1), (0, 0)]
    coords = corners
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    print(sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360))
    return sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    

def extract_panels(corners, img):
    """Extracts the panels from the given image."""
    count = 0
    colours = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(255,255,255)]
    #           Blue,       Green,      Red,    Yellow,         White  
    # for points in corners:
    
    points = corners[5]

    test = np.zeros(img.shape, np.uint8)
    print(points)
    for i in range(1, len(points)):
        print(i)
        print(colours[i])
        cv2.circle(test, (int(points[i][0]), int(points[i][1])), 3, colours[i], -1)

    cv2.imshow("test", test)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    if len(points[-4:]) == 4:
        #---- 4 corner points of the bounding box

        """Need to find the correct order of the points"""
        #---- top left point
        # Sort the points clockwise
        sorted_points = sort_corners(points[-4:])

        for i in range(len(sorted_points)):
            cv2.circle(test, (int(sorted_points[i][0]), int(sorted_points[i][1])), 3, colours[i], -1)
        
        cv2.imshow("test", test)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


        pts_src = np.array(sorted_points)

        print(pts_src)

        #---- 4 corner points of the black image you want to impose it on
        pts_dst = np.array([[0.0,0.0],[200.0, 0.0],[ 0.0,300.0],[200.0, 300.0]])

        #---- forming the black image of specific size
        im_dst = np.zeros((300, 200, 3), np.uint8)

        #---- Framing the homography matrix
        h, status = cv2.findHomography(pts_src, pts_dst)

        #---- transforming the image bound in the rectangle to straighten
        im_out = cv2.warpPerspective(img, h, (im_dst.shape[1],im_dst.shape[0]))
        
        cv2.imshow('Panel ' + str(count), im_out)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        # file_name = './output/' + str(count) + '.jpg'
        # cv2.imwrite(file_name, im_out)
        count += 1


def get_corners_v2(img, contours, CORNER_GROUP):
    """Returns the corners of the panel."""
    all_corners  = []
    for c in contours:
        temp_image = np.zeros(img.shape, np.uint8)
        cv2.drawContours(temp_image, [cv2.convexHull(c)], -1, (255,255,255), 2)
        
        edges = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
        
        ret, thresh = cv2.threshold(edges, THRESHOLD, 255, cv2.THRESH_BINARY)

        dst= cv2.cornerHarris(thresh, 3, 3, 0.05)

        kernel= np.ones((7,7), np.uint8)

        dst = cv2.dilate(dst, kernel, iterations= 2)

        ret, dst = cv2.threshold(dst,0.2*dst.max(),255,0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(thresh,np.float32(centroids),(CORNER_GROUP,CORNER_GROUP),(-1,-1),criteria)

        all_corners.append(corners)
    return all_corners

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
    # cv2.imwrite("panel_1_hsv.jpg", hsv_image)
    # cv2.imwrite('hsv_image.jpg', hsv_image)
    # cv2.imwrite('extracted_blue.jpg', cv2.inRange(hsv_image, BLUE_MIN, BLUE_MAX))
    # Apply Blue threshold to identify blue colour in the image
    blue_threshold = cv2.cvtColor(cv2.inRange(hsv_image, BLUE_MIN, BLUE_MAX), cv2.COLOR_GRAY2RGB)
    # cv2.imwrite('binary.jpg', blue_threshold)
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
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    cv2.drawContours(temp_image, [cv2.convexHull(contour)], -1, (255,255,255), 1)
    
    edges = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(edges, THRESHOLD, 255, cv2.THRESH_BINARY)

    dst= cv2.cornerHarris(thresh, 3, 3, 0.05)

    # print(dst)

    kernel= np.ones((7,7), np.uint8)

    # dst = cv2.dilate(harris_corners, kernel, iterations= 2)

    ret, dst = cv2.threshold(dst,0.2*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(thresh,np.float32(centroids),(5,5),(-1,-1),criteria)
    for i in range(1, len(corners)):
        print(corners[i])

    temp_image[dst>0.2*dst.max()]=[0,0,255]
    cv2.imshow('CORNERS', temp_image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    # temp_image[harris_corners > 0.025 * harris_corners.max()]= [255,127,127]

    cv2.imshow('Harris Corners', temp_image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    
    
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
    print(arr)
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length      

def get_corners(import_img, filtered_contours, k):
    corners = []
    # result = []
    for c in filtered_contours:

        lines = get_hough_lines(import_img, c)
        if lines is None or len(lines) == 0:
            return

        # final_lines = draw_hough_lines(import_img, lines)
        

        # Groups lines into horizontal and vertical lines
        segmented = segment_by_angle_kmeans(lines)
        # Finds the intersections between groups of lines
        intersections = segmented_intersections(segmented)

        

        points = []
        for x in intersections:
            points.append(x[0])
        groups = find_intersection_groups(points, k)

        
        for p in points:
            cv2.circle(import_img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
        
        cv2.imshow("final_lines", import_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        result = []
        removed = []
        for index, [x, y] in enumerate(points):
            if not [x, y] in result and not [x, y] in removed:
                removed += list(filter(lambda c: abs(c[0] - x) < k, points))
                result.append([x, y])
        # print(result)
        # print('\n')
        # print(groups)
        # print("Intersections  |  Point  |  Group  ")
        # for p in range(10):
        #     print(str(intersections[p]) + "  |  " +str(points[p]) + "  |  " + str(groups[p]))

        # for g in range(10):        
        #     print(groups[g])

        for p in result:
            cv2.circle(import_img, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)

        # final_points = []
        # for g in result:
        #     final_points.append(find_center(np.array(g)))
        # corners.append(final_points)
        if len(result) == 4:
            corners.append(result)
        else:
            print(result)

    return corners

if __name__ == '__main__':
    main()