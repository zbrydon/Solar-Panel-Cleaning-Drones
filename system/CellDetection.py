import Detector
import Options
import numpy as np

class CellDetection:
    def __init__(self, panels, panel_type, no_cells_per_panel):
        self.panels = panels

        self.min_area = 5000
        self.threshold = 70
        self.blur_value = 7

        if no_cells_per_panel != None:
            self.expected_cell_count = no_cells_per_panel
        else:
            self.expected_cell_count = 60

        if panel_type == Options.PanelType.BLUE.value:
            self.colour_min = np.array([100, 100, 100],np.uint8)
            self.colour_max = np.array([140, 200, 235],np.uint8)
        elif panel_type == Options.PanelType.BLACK.value:
            self.colour_min = np.array([100, 25 , 35],np.uint8)
            self.colour_max = np.array([130, 80, 80],np.uint8)
        else:
            print("Panel type not recognised.")
            return None

        self.corner_group = 1
        self.array_detection = False
        self.enable_blur = False

        self.cells = []

        self.cell_detection = True
        self.display = False

        for panel in panels:
            if not isinstance(panel, dict): continue
            if panel['extracted_img'] is None: continue

            self.min_area = ((panel['extracted_img'].shape[0] * panel['extracted_img'].shape[1])/self.expected_cell_count) - ((panel['extracted_img'].shape[0] * panel['extracted_img'].shape[1])/self.expected_cell_count)*0.7

            detector = Detector.Detector(panel['extracted_img'], self.min_area, self.threshold, self.blur_value, self.colour_min, self.colour_max,   self.corner_group, self.enable_blur, self.array_detection, self.cell_detection, self.expected_cell_count, self.display)

            self.cells.append(detector.find_contours())


    def get_extracted(self):
        return self.cells
