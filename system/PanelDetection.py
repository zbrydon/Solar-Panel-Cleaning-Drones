import Detector
import Options
import numpy as np

class PanelDetection:
    def __init__(self, arrays, panel_type, no_panels = None):
        self.arrays = arrays

        self.min_area = 3000
        self.threshold = 60
        self.blur_value = 7

        if no_panels != None:
            self.min_no_of_panels = no_panels
        else:
            self.min_no_of_panels = 2

        if panel_type == Options.PanelType.BLUE.value:
            self.colour_min = np.array([70, 70, 60],np.uint8)
            self.colour_max = np.array([140, 255, 240],np.uint8)
        elif panel_type == Options.PanelType.BLACK.value:
            self.colour_min = np.array([100, 20 , 40],np.uint8)
            self.colour_max = np.array([130, 85, 140],np.uint8)
        else:
            print("Panel type not recognised.")
            return None

        self.corner_group = 20
        self.array_detection = False
        self.enable_blur = False

        self.panels = []

        self.cell_detection = False
        self.expected_cell_count = 0
        self.display = False

        for array in arrays:
            if array['extracted_img'] is None: continue
            if array['type'] != 'array':
                self.panels.append(array['extracted_img'])
                continue
            if no_panels != None:
                self.min_area = (
                    (array['extracted_img'].shape[0] *
                    array['extracted_img'].shape[1])/
                    self.min_no_of_panels) - (
                        (array['extracted_img'].shape[0] *
                        array['extracted_img'].shape[1])/
                        self.min_no_of_panels) * 0.7

            detector = Detector.Detector(array['extracted_img'], self.min_area, self.threshold, self.blur_value, self.colour_min, self.colour_max,   self.corner_group, self.enable_blur, self.array_detection, self.cell_detection, self.expected_cell_count, self.display)

            extracted = detector.find_contours()

            for panel in extracted:
                self.panels.append(panel)



    def get_extracted(self):
        return self.panels


