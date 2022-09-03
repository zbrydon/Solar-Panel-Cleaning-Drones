import csv
import os

class Statistics:
    def __init__(self) -> None:
        pass

    def print_image_stats(self, output, number_of_arrays, NUMBER_OF_ARRAYS, panels, NUMBER_OF_PANELS, cells, NUMBER_OF_CELLS_PER_PANEL, num_covered, itt = 0):
        # Print summary of detected cells
        print("\n== Summary ==\n")
        print("Number of Arrays Detected: {}".format(number_of_arrays))
        if NUMBER_OF_ARRAYS != None:
            print("Number of Arrays Expected: {}".format(NUMBER_OF_ARRAYS))
            print('Percentage of Cells Detected: ' + str(round(number_of_arrays/NUMBER_OF_ARRAYS * 100, 2) ) + '%')

        print("Number of Panels Detected: {}".format(panels))
        if NUMBER_OF_PANELS != None:
            print("Number of Panels Expected: {}".format(NUMBER_OF_PANELS))
            print('Percentage of Cells Detected: ' + str(round(panels/NUMBER_OF_PANELS * 100, 2) ) + '%')

        print("Number of Cells Detected: {}".format(cells))
        if NUMBER_OF_CELLS_PER_PANEL != None:
            cells_to_detect = (NUMBER_OF_PANELS * NUMBER_OF_CELLS_PER_PANEL) - num_covered
            percentage = round(cells/(NUMBER_OF_CELLS_PER_PANEL * NUMBER_OF_PANELS) * 100, 2)
            print("Number of Cells Expected: {}".format(cells_to_detect))
            print('Percentage of Cells Detected: ' + str(percentage) + '%')
            error = round((abs(cells - cells_to_detect)/cells_to_detect) * 100, 2)
            print('Error: {}%'.format(error))

        print("====================================")

        if NUMBER_OF_ARRAYS != None and NUMBER_OF_PANELS != None and NUMBER_OF_CELLS_PER_PANEL != None:
            file_name = output + '/' + 'summary.csv'
            self.write_to_csv_error(file_name, num_covered, cells, cells_to_detect, error, itt)

    def write_to_csv_error(self, file_name, num_covered, num_detected, cells_to_detect, error, itt):
        print(os.getcwd())

        # Write to CSV
        try:
            with open(file_name, 'a') as csvfile:

                fieldnames = ['Number of cells covered', 'Number of cells un-covered', 'Number of cells detected', 'Error', 'Iteration']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if itt == 0: writer.writeheader()


                writer.writerow({'Number of cells covered': num_covered, 'Number of cells un-covered':cells_to_detect, 'Number of cells detected': num_detected, 'Error': error, 'Iteration': itt})
        except FileNotFoundError:
            print(os.getcwd())
            with open(file_name, 'w') as csvfile:

                fieldnames = ['Number of cells covered', 'Number of cells un-covered', 'Number of cells detected', 'Error', 'Iteration']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if itt == 0: writer.writeheader()


                writer.writerow({'Number of cells covered': num_covered, 'Number of cells un-covered':cells_to_detect, 'Number of cells detected': num_detected, 'Error': error, 'Iteration': itt})

    def dir_test(self):
        return os.getcwd().split('/')[-1] == 'output'
