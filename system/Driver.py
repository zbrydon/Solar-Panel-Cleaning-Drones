import Input
import ArrayDetection
import PanelDetection
import CellDetection
import Options
import Statistics
import Drone
import os
import sys


class Driver:
    def __init__(self):
        self.input_type = None
        self.input_path = "./input"
        self.output_path = "./output"
        self.file_type = None
        self.panel_type = None
        self.no_arrays = None
        self.no_panels = None
        self.no_cells_per_panel = None
        self.no_cells_covered = None

    def start(self):
        if len(sys.argv) > 1:
            if sys.argv[1] == "-help":
                print("Usage: python3 Driver.py")
                print("Options:")
                print("-help: Displays this message")
                print("-gui: Starts the GUI")
                print("-input ./input.json: Reads the path to a JSON file containing the system configuration")
                return
            elif sys.argv[1] == "-input":
                input_data = Input.Input()
                data = input_data.read_json_config(sys.argv[2])
                self.input_type = data['input_type']
                self.input_path = os.path.dirname(__file__) + data['input_path']
                self.output_path = os.path.dirname(__file__) + data['output_path']
                self.file_type = data['file_type']
                self.panel_type = data['panel_type']
                self.no_arrays = data['no_arrays']
                self.no_panels = data['no_panels']
                self.no_cells_per_panel = data['no_cells_per_panel']
                self.no_cells_covered = data['no_cells_covered']

                self.path_existence_check(self.output_path)

                if self.input_type == Options.InputType.IMAGE.value:
                    self.image_detection()
                elif self.input_type == Options.InputType.IMAGES.value:
                    self.images_detection()
                elif self.input_type == Options.InputType.VIDEO.value:
                    self.video_detection()
                elif self.input_type == Options.InputType.DRONE.value:
                    self.drone_detection()
                else:
                    print("Invalid input type")
                    return


            elif sys.argv[1] == "-gui":
                self.input_type = self.menu(Options.InputType, "input")
                self.panel_type = self.menu(Options.PanelType, "panel")
                self.file_type = self.menu(Options.FileType, "image")

                if self.input_type == Options.InputType.IMAGE.value:
                    print("Input type: IMAGE")


                    self.input_path = os.path.dirname(__file__) + self.get_paths('input', self.input_path)
                    self.output_path = os.path.dirname(__file__) +  self.get_paths('output', self.output_path)

                    print("Input path: " + self.input_path)
                    print("Output path: " + self.output_path)

                    image_name = self.get_image_name()

                    if self.no_arrays == None:
                        self.get_image_stats()

                    print("Number of arrays: " + str(self.no_arrays))
                    print("Number of panels: " + str(self.no_panels))
                    print("Number of cells per panel: " + str(self.no_cells_per_panel))
                    print("Number of cells covered: " + str(self.no_cells_covered))

                    print(self.input_path + image_name)
                    self.image_detection(self.input_path + "/" +image_name)


                elif self.input_type == Options.InputType.IMAGES.value:
                    print("Input type: IMAGES")

                    self.input_path = '/input/input'

                    self.input_path = os.path.dirname(__file__) + self.get_paths('input', self.input_path)
                    self.output_path = os.path.dirname(__file__) +  self.get_paths('output', self.output_path)

                    self.images_detection()

                elif self.input_type == Options.InputType.VIDEO.value:
                    print("Input type: VIDEO")

                    self.input_path = os.path.dirname(__file__) + self.get_paths('input', self.input_path)
                    self.output_path = os.path.dirname(__file__) +  self.get_paths('output', self.output_path)

                    print("Input path: " + self.input_path)
                    print("Output path: " + self.output_path)

                    video_name = self.get_video_name()

                    self.video_detection(self.input_path + "/" + video_name)

            else:
                print("Invalid argument")
                print( "Try: `python3 Driver.py -help` for more information.")
                return
        else:
            print("Invalid argument")
            print( "Try: `python3 Driver.py -help` for more information.")
            return

    def image_detection(self, image_name=None):
        image_input = Input.Input()

        if image_name == None:
            if not os.path.isfile(self.input_path):
                print("Invalid image name")
                exit(1)
            image = image_input.get_image(self.input_path)
        else:
            image = image_input.get_image(image_name)

        self.detection(image)

    def images_detection(self):
        if not os.path.isdir(self.input_path):
            print("Input path is invalid")
            exit

        image_input = Input.Input()

        images = image_input.get_images(self.input_path, str(Options.FileType(self.file_type)).split(".")[1].lower())

        if len(images) == 0:
            print("No images found")
            exit

        for image in images:

            self.no_arrays = image['num_arrays']
            self.no_panels = image['num_panels']
            self.no_cells_per_panel = image['num_cells_per_panel']
            self.no_cells_covered = image['cells_covered']

            self.detection(image['image'])

    def video_detection(self, video_name=None):
        video_input = Input.Input()

        if video_name == None:
            if not os.path.isfile(self.input_path):
                print("Invalid image name")
                exit(1)
            video = video_input.get_video(self.input_path)
        else:
            video = video_input.get_video(video_name)

        video_input.show_video(self.panel_type, self.no_panels, self.no_arrays, self.no_cells_per_panel, self.no_cells_covered, self.output_path)

    def drone_detection(self):
        drone_input = Drone.Drone(False)
        drone_input.display_video(self.panel_type, self.no_panels, self.no_arrays, self.no_cells_per_panel, self.no_cells_covered, self.output_path)


    def detection(self, image):
        array_detection = ArrayDetection.ArrayDetection(image, self.panel_type).get_extracted()

        panel_detection = PanelDetection.PanelDetection(array_detection, self.panel_type, self.no_panels).get_extracted()

        cell_detection = CellDetection.CellDetection(panel_detection, self.panel_type, self.no_cells_per_panel).get_extracted()

        count = 0
        for panel in cell_detection:
            count += len(panel)

        statistics = Statistics.Statistics()
        statistics.print_image_stats(self.output_path, len(array_detection), self.no_arrays, len(panel_detection), self.no_panels, count, self.no_cells_per_panel, self.no_cells_covered)

    def path_existence_check(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def menu(self, enum, name):
        print("Please select an " + name + " type:")
        x = 1
        for type in (enum):
            print(str(x) + ": " + str(type.name))
            x += 1

        choice = input("Choice: ")

        # Check that choice is valid
        try:
            choice = int(choice)
        except ValueError:
            print("Invalid choice")
            print("Please select a number between 1 and " + str(len(enum)))
            return self.menu(enum, name)

        if choice < 1 or choice > len(enum):
            print("Invalid choice")
            print("Please select a number between 1 and " + str(len(enum)))
            return self.menu(enum, name)

        return choice

    def get_image_name(self):
        print("Enter the name of the image you wish to process:")
        print("Note: If the image name follows the following format:")
        print("arrays-<NUMBER OF ARRAYS>-panels-<NUMBER OF PANELS>-cells_per_panel-<NUMBER OF CELLS PER PANELS>-cells_covered-<NUMBER OF CELLS COVERED>")
        print("The image statistics will be automatically detected.")
        image_name = input("Image name: ")

        file_extension = str(Options.FileType(self.file_type)).split(".")[1].lower()

        if image_name == "":
            print("Invalid image name")
            return self.get_image_name()
        if not os.path.isfile(self.input_path + "/" + image_name + "." + file_extension):
            print("Invalid image name")
            return self.get_image_name()

        try:
            data = image_name.split("-")

            d = dict([(k, v) for k,v in zip (data[::2], data[1::2])])
            num_arrays = int(d['arrays'])
            num_panels = int(d['panels'])
            num_cells_per_panel = int(d['cells_per_panel'])
            cells_covered = int(d['cells_covered'])

            self.no_panels = num_panels
            self.no_arrays = num_arrays
            self.no_cells_per_panel = num_cells_per_panel
            self.no_cells_covered = cells_covered

            return image_name + "." + file_extension
        except:
            print("Stats not found in image name")
            return image_name + "." + file_extension

    def get_video_name(self):
        print("Enter the name of the video you wish to process:")

        video_name = input("Video name: ")

        file_extension = str(Options.FileType(self.file_type)).split(".")[1].lower()

        if video_name == "":
            print("Invalid image name")
            return self.get_image()
        if not os.path.isfile(self.input_path + "/" + video_name + "." + file_extension):
            print("Invalid image name")
            return self.get_image()
        else:
            return video_name + "." + file_extension

    def get_paths(self, type, path):
        if type == 'input' or type == 'output':
            print("The default " + type + " path is: " + os.path.dirname(__file__) + path)
            print("Enter a new " + type + " path or press enter to use the default path.")
            new_path = input("New path: ")

            if new_path == "":
                return path
            if not os.path.isdir(new_path):
                print("Input path is invalid")
                return self.get_paths(type, path)
            if new_path != "":
                return new_path
        else:
            print("Invalid type")

    def get_image_stats(self):
        print("Enter the number of arrays in the image:")
        no_arrays = input("Number of arrays: ")

        print("Enter the number of panels per array:")
        no_panels = input("Number of panels: ")

        print("Enter the number of cells per panel:")
        no_cells_per_panel = input("Number of cells per panel: ")

        print("Enter the number of cells covered by soiling:")
        no_cells_covered = input("Number of cells covered: ")

        try:
            no_arrays = int(no_arrays)
            no_panels = int(no_panels)
            no_cells_per_panel = int(no_cells_per_panel)
            no_cells_covered = int(no_cells_covered)
        except ValueError:
            print("Invalid number")
            return self.get_image_stats()

        self.no_arrays = no_arrays
        self.no_panels = no_panels
        self.no_cells_per_panel = no_cells_per_panel
        self.no_cells_covered = no_cells_covered



def main():
    driver = Driver()
    driver.start()

if __name__ == "__main__":
    main()

