import cv2
import os
import glob
import json
import ArrayDetection
import PanelDetection
import CellDetection
import Statistics

class Input:
    def __init__(self) -> None:
        pass

    def get_images(self, path, file_type):
        os.chdir(path)
        images = []


        for file in glob.glob("*." + file_type):
            file_name = str(file)

            try:
                data = file_name[:-4].split("-")
                d = dict([(k, v) for k,v in zip (data[::2], data[1::2])])
                num_arrays = int(d['arrays'])
                num_panels = int(d['panels'])
                num_cells_per_panel = int(d['cells_per_panel'])
                cells_covered = int(d['cells_covered'])

                import_img = self.get_image(file_name)

                image_dict = {'num_arrays': num_arrays, 'num_panels': num_panels, 'num_cells_per_panel': num_cells_per_panel, 'cells_covered': cells_covered, 'image': import_img}

                images.append(image_dict)
            except Exception as e:
                print("Warning: Invalid image name")


        return images

    def get_image(self, path):
        import_img = cv2.imread(path)
        return import_img

    def get_video(self, path):
        self.video_capture = cv2.VideoCapture(path)
        self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.video_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.video_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)


    def show_video(self, panel_type, no_panels, no_arrays, no_cells_per_panel, no_cells_covered, output_path):
        while True:
            frame_count = 0
            ret, frame = self.video_capture.read()
            if ret:
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
                if cv2.waitKey(int(1000 / self.video_fps)) & 0xFF == ord('q'):
                    break
                frame_count += 1
            else:
                break
        self.video_capture.release()
        cv2.destroyAllWindows()

    def get_frames(self):
        frames = []
        while True:
            ret, frame = self.video_capture.read()
            if ret:
                frames.append(frame)
            else:
                break

        return frames

    def read_json_config(self, config_file):
        # Test if config file exists
        if os.path.isfile(config_file):
            with open(config_file) as f:
                data = json.load(f)
            return data
        else:
            print("Error: Config file does not exist")
            exit()
