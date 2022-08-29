import Detector
import Options
import numpy as np

class ArrayDetection:
    def __init__(self, frame, panel_type):
        self.frame = frame

        self.min_area = 2000
        self.threshold = 60
        self.blur_value = 15

        if panel_type == Options.PanelType.BLUE.value:
            self.colour_min = np.array([70, 60, 80],np.uint8)
            self.colour_max = np.array([140, 255, 255],np.uint8)
        elif panel_type == Options.PanelType.BLACK.value:
            self.colour_min = np.array([100, 25 , 35],np.uint8)
            self.colour_max = np.array([130, 80, 80],np.uint8)
        else:
            print("Panel type not recognised.")
            return None

        self.corner_group = 50
        self.array_detection = True
        self.enable_blur = True

        self.cell_detection = False
        self.expected_cell_count = 0
        self.display = False

        self.detector = Detector.Detector(self.frame, self.min_area, self.threshold, self.blur_value, self.colour_min, self.colour_max,   self.corner_group, self.enable_blur, self.array_detection, self.cell_detection, self.expected_cell_count, self.display)

        self.extracted = self.detector.find_contours()

    def get_extracted(self):
        return self.extracted

    def get_array_contours(self):
        if len(self.extracted) == 0:
            return self.frame
        return self.detector.draw_contours(self.frame, self.extracted[0]['contours'])


