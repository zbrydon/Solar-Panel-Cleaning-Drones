import cv2
import numpy as np
import math
import copy
import csv
from scipy.spatial import distance as dist

class Detector:
    def __init__(self, frame, min_area , threshold, blur_value, colour_min, colour_max, corner_group, enable_blur = True, array_detection = False, cell_detection = False, expected_cell_count = 0, display = False):
        self.frame = frame
        self.min_area = min_area
        self.threshold = threshold
        self.blur_value = blur_value
        self.colour_min = colour_min
        self.colour_max = colour_max
        self.corner_group = corner_group
        self.enable_blur = enable_blur
        self.array_detection = array_detection
        self.cell_detection = cell_detection
        self.expected_cell_count = expected_cell_count
        self.display = display
        self.extracted = []
        self.type_detected = ''

    def find_contours(self):
        if self.display:
            cv2.imshow("import_img", self.frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

        blue_threshold = self.get_colour_component(self.frame)
        blured_image = self.convert_image_to_binary(blue_threshold)
        threshold_image = self.get_threshold_image(blured_image)

        if self.cell_detection:
           threshold_image = self.refine_cells(threshold_image)


        contours, length = self.get_contours(threshold_image)

        if self.array_detection:
            try:
                c = max(contours, key = cv2.contourArea)
                filtered_contours = [c]
            except ValueError:
                filtered_contours = []

        else:
            filtered_contours = self.filter_contours(contours)

        original_image_contours = self.draw_contours(self.frame, filtered_contours)

        result = self.create_mask(self.frame, filtered_contours)

        if self.display:
            threshold_image = cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2BGR)
            numpy_horizontal_concat = np.concatenate((threshold_image, original_image_contours), axis=1)
            cv2.imshow("original_image_contours", numpy_horizontal_concat)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

        final_image = self.draw_contours(result, filtered_contours)

        if self.cell_detection:
            return filtered_contours

        else:
            corners = self.get_corners(self.frame, filtered_contours)

            if corners is not None:
                for points in corners:

                    if len(points) == 5:
                        points = self.order_points(points[1:])

                        self.output_result(points, filtered_contours)
                    elif self.array_detection:
                        points = self.filter_points(points[1:])

                        try:
                            points = self.order_points(points)
                        except ValueError:
                            continue

                        self.output_result(points, filtered_contours)

        if self.display:
            cv2.imshow('Final', final_image)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()


        return self.extracted

    def output_result(self, points, filtered_contours):
        height = math.sqrt((points[0][0] - points[3][0])**2 + (points[0][1] - points[3][1])**2)
        width = math.sqrt((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)

        # Assumes that arrays are orientated landscape in the image
        # -----------------------------
        # |   |   |   |   |   |   |   |
        # |   |   |   |   |   |   |   |
        # -----------------------------
        # |   |   |   |   |   |   |   |
        # |   |   |   |   |   |   |   |
        # -----------------------------

        if height > width:
            self.type_detected = 'panel'
        else:
            self.type_detected = 'array'

        out = self.warpImage(self.frame, points, height, width)

        self.extracted.append({'type': self.type_detected, 'extracted_img': out, 'contours': filtered_contours})

    def refine_cells(self, threshold_image):
        rows,cols = threshold_image.shape
        for i in range(rows):
            black_count = 0
            for j in range(cols):
                k = threshold_image[i,j]
                if k == 0:
                    black_count += 1
            if black_count > cols*0.8:
                for j in range(cols):
                    threshold_image[i,j] = 0

        # Reverse

        for i in range(cols):
            black_count = 0
            for j in range(rows):
                k = threshold_image[j,i]
                if k == 0:
                    black_count += 1
            if black_count > rows*0.7:
                for j in range(rows):
                    threshold_image[j,i] = 0

        return threshold_image



    # Remove points on the same approximate line
    def filter_points(self, points):

        # print(points)

        threshold = 20
        points_to_remove = []


        for i in range(len(points)):
            x_line = []
            y_line = []
            x_line.append(points[i])
            y_line.append(points[i])
            for j in range(len(points)):
                if self.is_equal(points[i], points[j]):
                    continue
                if self.is_same_line(1, points[i], points[j], threshold):
                    x_line.append(points[j])
                if self.is_same_line(0, points[i], points[j], threshold):
                    y_line.append(points[j])

            x_line.sort(key=lambda x: x[0])
            y_line.sort(key=lambda y: y[1])

            if len(x_line) > 2:
                for p in x_line[1:-1]:
                    if not self.contains(p, points_to_remove): points_to_remove.append(p)

            if len(y_line) > 2:
                for p in y_line[1:-1]:
                    if not self.contains(p, points_to_remove): points_to_remove.append(p)

        result = []

        for p in points:
            if not self.contains(p, points_to_remove):
                result.append(p)

        return np.array(result)



    def contains(self, point, points):
        for p in points:
            if self.is_equal(point, p):
                return True
        return False

    def is_equal(self, a, b):
        if len(a) != len(b):
            return False
        else:
            for i in range(len(a)):
                if a[i] != b[i]:
                    return False
            return True

    def is_same_line(self, index, a, b, threshold):
        if len(a) != len(b):
            return False
        else:
            if abs(a[index] - b[index]) < threshold:
                return True

    # https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    def order_points(self, pts):
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]
        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        # now that we have the top-left coordinate, use it as an
        # anchor to calculate the Euclidean distance between the
        # top-left and right-most points; by the Pythagorean
        # theorem, the point with the largest distance will be
        # our bottom-right point
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]
        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        return np.array([tl, tr, br, bl], dtype="float32")

    def warpImage(self, image, corners, height, width):

        # 5% of the height and width of the image
        h_percent = 0.05 * height
        w_percent = 0.05 * width


        pts1 = np.float32([[corners[0][0] - w_percent, corners[0][1] - h_percent], [corners[1][0] + w_percent, corners[1][1] - h_percent], [corners[2][0] + w_percent, corners[2][1] + h_percent], [corners[3][0] - w_percent, corners[3][1] + h_percent]])
        pts2 = np.float32([[0,0],[width,0],[width,height],[0,height]])
        M = cv2.getPerspectiveTransform(pts1,pts2)

        out = cv2.warpPerspective(image,M,(int(width),int(height)))
        return out


    def get_colour_component(self, input_image):
        # Convert image to HSV
        hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        if self.display:
            cv2.imshow("hsv_image", hsv_image)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        # Apply Blue threshold to identify blue colour in the image
        blue_threshold = cv2.cvtColor(cv2.inRange(hsv_image, self.colour_min, self.colour_max), cv2.COLOR_GRAY2RGB)

        if self.display:
            numpy_horizontal_concat = np.concatenate((input_image, hsv_image, blue_threshold), axis=1)
            cv2.imshow("blue_threshold", numpy_horizontal_concat)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

        return blue_threshold

    def convert_image_to_binary(self, input_image):
        # convert the image into binary image
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        if self.enable_blur:
            # Apply GaussianBlur to reduce noise
            blured_image = cv2.GaussianBlur(gray_image, (self.blur_value, self.blur_value), 0)
            return blured_image
        return gray_image

    def get_threshold_image(self , input_image):
        #thresholding the frame
        ret, thresh = cv2.threshold(input_image, self.threshold, 255, cv2.THRESH_BINARY)
        return thresh

    def get_contours(self, img):
        # Find contours
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours, len(contours)

    def filter_contours(self, contours):
        # Filter contours by area
        filtered_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > self.min_area:
                filtered_contours.append(c)
        return filtered_contours

    def draw_contours(self, img, contours, color = (0, 255, 0), thickness = 2):
        contour_image = copy.deepcopy(img)
        # Draw contours
        for c in contours:
            hull = cv2.convexHull(c)
            cv2.drawContours(contour_image, [hull], -1, color, thickness)
        return contour_image

    def create_mask(self, input_image, contours):
        if len(contours) == 0:
            return input_image
        # Create a mask of the image
        mask = np.zeros(input_image.shape, np.uint8)
        for c in contours:
            new_image = cv2.drawContours(mask,[cv2.convexHull(c)], -1, (255,255,255), -1)
            new_image_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(new_image_gray, 100, 255, cv2.THRESH_BINARY)
            final = cv2.bitwise_and(input_image, input_image, mask = thresh)
        return final

    def get_corners(self, input_image, contours):
        """Returns the corners of the panel."""
        all_corners  = []
        for c in contours:
            temp_image = np.zeros(input_image.shape, np.uint8)
            cv2.drawContours(temp_image, [cv2.convexHull(c)], -1, (255,255,255), 2)

            edges = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)

            ret, thresh = cv2.threshold(edges, self.threshold, 255, cv2.THRESH_BINARY)

            dst= cv2.cornerHarris(thresh, 3, 3, 0.05)

            kernel= np.ones((7,7), np.uint8)

            dst = cv2.dilate(dst, kernel, iterations= 2)

            ret, dst = cv2.threshold(dst,0.2*dst.max(),255,0)
            dst = np.uint8(dst)
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            try:
                corners = cv2.cornerSubPix(thresh,np.float32(centroids),(self.corner_group, self.corner_group),(-1,-1),criteria)
            except Exception as e:
                self.draw_contours(input_image, contours, color = (0, 0, 255), thickness = 2)
                cv2.imshow("original", input_image)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                cv2.imshow("thresh", thresh)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                print(e)
                corners = []

            all_corners.append(corners)
        return all_corners
