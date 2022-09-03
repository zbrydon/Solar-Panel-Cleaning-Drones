import time, cv2
from threading import Thread
from djitellopy import Tello
import ArrayDetection
import PanelDetection
import CellDetection
import Statistics

class Drone:
    def __init__(self, save_video):
        self.tello = Tello()
        self.save_video = save_video

    def display_video(self, panel_type, no_panels, no_arrays, no_cells_per_panel, no_cells_covered, output_path):
        self.tello.connect()
        self.tello.streamon()
        frame_read = self.tello.get_frame_read()

        while True:
            frame_count = 0
            # ret, frame = self.video_capture.read()
            frame = frame_read.frame

            frame = ArrayDetection.ArrayDetection(frame, panel_type).get_array_contours()

            array_detection = ArrayDetection.ArrayDetection(frame, panel_type)

            arrays = array_detection.get_extracted()

            panel_detection = PanelDetection.PanelDetection(arrays, panel_type, no_panels)

            panels = panel_detection.get_extracted()

            cell_detection = CellDetection.CellDetection(panels, panel_type, no_cells_per_panel)

            cells = cell_detection.get_extracted()

            count = 0
            for panel in cells:
                count += len(panel)

            statistics = Statistics.Statistics()
            statistics.print_image_stats(output_path, len(arrays), no_arrays, len(panels), no_panels, count, no_cells_per_panel, no_cells_covered)

            cv2.imshow("video", frame)
            if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
                break
            frame_count += 1

        cv2.destroyAllWindows()

