import os
import numpy as np
from numpy import random as rand
rand.seed(2)  # Seed is chosen so that tumor doesn't go extinct
import matplotlib.pyplot as plt
import imageio

SAVE_DIR = '/Users/Andrew/PycharmProjects/Tumor_Simulation/output/'

DIM = 3  # Number of DIMensions
DX = 1.  # Grid spacing
D = .05  # Diffusion constant
tau_D = 1 / (2 * DIM) * DX ** 2 / D  # Diffusion time constant

CELL_TYPES_LIST = ('A', 'B', 'C')
CELL_COLORS = {'A': 'm', 'B': 'g', 'C': 'y'}

RATE_B = {'A': 0.1, 'B': 0.15, 'C': 0.2}
RATE_D = 0.05  # Death rate
CARRYING_CAPACITY = 20  # Number of sustainable cells at each gridpoint

k1 = (1 - 0.33) / np.log(2)
k1_p = (1 + 0.33) / np.log(2)
k2 = 2
k2_p = 3

MUTATION_PROBS = {'AB': 0.05, 'AC': 0.01, 'BC': 0.10}

MAX_CELL_COUNT = 1000
SEEDING_COUNT_THRESHOLD = 200

class Cell:
    """A cell object contains the information about a tumor cell.
        Instance variables:
        - coords
        - cell_type
        Methods:
        - move(new_coords)
        - brownian_diffuse(dt)
        - equals(cell)
        - copy()
        - to_string()
    """

    def __init__(self, coords=np.zeros(DIM, dtype=np.float_), cell_type='A'):
        """Coords can be a tuple or an numpy array
        Tested Th 6/28 8:45pm"""
        if len(coords) == DIM:
            self.coords = np.array(coords, dtype=np.float_)
        else:
            raise ValueError('$Dimensions mismatch')
        if cell_type in CELL_TYPES_LIST:
            self.cell_type = cell_type
        else:
            raise ValueError('$Cell type is invalid')

    def move(self, new_coords):
        """Sets cell coordinates to a new numpy array
        Tested Th 6/28 8:49pm"""
        self.coords = np.array(new_coords, dtype=np.float_)

    def brownian_diffuse(self, dt):
        """Diffuses cell using Smoluchowski diffusion
        Tested Th 6/28 9:13pm"""
        new_coords = self.coords + np.sqrt(2 * D * dt) * rand.normal(0, 1, len(self.coords))
        self.move(new_coords)

    def equals(self, cell):
        """Tests whether two cells have the same coordinates and cell_type
        Tested Th 6/28 10:43pm"""
        return all(self.coords == cell.coords) and self.cell_type == cell.cell_type

    def copy(self):
        """Returns another cell object with identical properties.
        Tested Th 7/5 9:55pm"""
        return Cell(self.coords, self.cell_type)

    def to_string(self):
        """For debugging
        Tested Th 6/28 8:45pm"""
        s = ''
        for i in range(len(self.coords)):
            s = s + ' ' + str(self.coords[i])
        return '(Type ' + self.cell_type + ')' + s


class Grid:
    """Contains information about all of the cells (location, quantity, and types.)
        Instance variables:
        - dictionary
        - scale
        - total_cell_count
        - cell_types_count
        Methods:
        - cell_count(gridpoint)
        - add_cell(coords, cell_type)
        - remove_cell(coords, cell_type)
        - update_gridpoints()
        - copy()
        - choose_random_cell(gridpoint, cell_type)
        - birth_propensity(gridpoint, cell_type)
        - death_propensity(gridpoint, cell_type)
        - propensities(gridpoint)
        - net_propensity(gridpoint)
        - T_R(gridpoint)
        - T_R_min()
        - plot()
        - print()
    """

    def __init__(self, scale=1 / DX):
        """Grid is a dictionary with tuples representing discrete lattice points
            and values representing sets of cells nearest to that point
        Scale is the number of gridpoints per unit
        Tested Thu 6/28 9:16pm"""
        self.dictionary = {}
        self.scale = scale
        self.total_cell_count = 0
        self.cell_types_count = {}
        for cell_type in CELL_TYPES_LIST:
            self.cell_types_count[cell_type] = 0

    def cell_count(self, gridpoint, cell_type=None):
        """Returns the number of cells at a given gridpoint
        If cell_type is not specified, then the total number of cells
            at that gridpoint is returned.
        Tested Sun 8/12 11:28pm"""
        if gridpoint not in self.dictionary:
            return 0
        elif cell_type == None:
            return len(self.dictionary[gridpoint])
        else: # Return number of cells of the specified type
            count = 0
            for cell in self.dictionary[gridpoint]:
                if cell.cell_type == cell_type:
                    count += 1
            return count

    def add_cell(self, coords=np.zeros(DIM), cell_type='A'):
        """Adds cell as a value to key corresponding to discrete gridpoint
           If no gridpoint is in dictionary, create a list containing that cell
           Otherwise, add it to the existing value list.
        Tested Sun 8/12 11:25pm"""
        if len(coords) != DIM:
            raise ValueError('$Dimension mismatch')
        gridpoint = tuple([round(i * self.scale) / self.scale for i in coords])
        if gridpoint not in self.dictionary:
            self.dictionary[gridpoint] = [Cell(coords, cell_type)]
        else:
            self.dictionary[gridpoint].append(Cell(coords, cell_type))
        self.total_cell_count += 1
        self.cell_types_count[cell_type] += 1

    def remove_cell(self, coords, cell_type):
        """Removes a cell from the grid.
        Does not remove empty gridpoints.
        Tested 8/12 11:25pm"""
        gridpoint = tuple([round(i * self.scale) / self.scale for i in coords])
        if gridpoint not in self.dictionary:
            raise KeyError('No such cell to remove.')
        else:
            for cell in self.dictionary[gridpoint]:
                if all(np.equal(cell.coords, coords)) and cell.cell_type == cell_type:
                    self.dictionary[gridpoint].remove(cell)
                    self.total_cell_count -= 1
                    self.cell_types_count[cell_type] -= 1
                    break

    def update_gridpoints(self):
        """Remove any gridpoints which do not contain any cells.
        Tested Sun 8/12 11:25pm"""
        null_gridpoints = []
        for gridpoint in self.dictionary:
            if len(self.dictionary[gridpoint]) == 0:
                null_gridpoints.append(gridpoint)
        for gridpoint in null_gridpoints:
            del self.dictionary[gridpoint]

    def copy(self):
        """Returns an identical copy of the grid.
        This method is slow and should be avoided if possible.
        Tested Thu 7/5 9:55pm"""
        new_grid = Grid()
        for gridpoint in self.dictionary:
            for cell in self.dictionary[gridpoint]:
                new_grid.add_cell(cell.coords, cell.cell_type)
        return new_grid

    def choose_random_cell(self, gridpoint, cell_type=None):
        """Returns coordinates of a random cell at a gridpoint with the specified cell type.
        If no cell type is specified, then it selects an arbitrary random cell from the gridpoint.
        Tested Sun 8/12 11:25pm"""
        if cell_type == None:
            random_cell = rand.choice(tuple(self.dictionary[gridpoint]))
            return random_cell.coords
        else:
            cell_subset = []
            for cell in self.dictionary[gridpoint]:
                if cell.cell_type == cell_type:
                    cell_subset.append(cell)
            random_cell = rand.choice(cell_subset)
            return random_cell.coords

    # Propensity functions:
    def birth_propensity(self, gridpoint, cell_type):
        """Returns the birth propensity of the specified
        cell type at the specified gridpoint.
        Tested Sun 8/12 11:25pm"""
        return self.cell_count(gridpoint, cell_type) * RATE_B[cell_type] * (
                    1 - self.cell_count(gridpoint) / CARRYING_CAPACITY)

    def death_propensity(self, gridpoint, cell_type):
        """Returns the death propensity of the specified
        cell type at the specified gridpoint.
        Tested Sun 8/12 11:25pm"""
        return self.cell_count(gridpoint, cell_type) * RATE_D

    def propensities(self, gridpoint):
        """Produces a list of propensities evaluated at the gridpoint: [birth(A), death(A), birth(B), death(B), ...]
           propensities_list[2*i] is the birth propensity of cell type i
           propensities_list[2*i + 1] is the death propensity of cell type i
           Tested Sun 8/12 11:30pm"""
        propensities_list = []
        for cell_type in CELL_TYPES_LIST:
            propensities_list.append(self.birth_propensity(gridpoint, cell_type))
            propensities_list.append(self.death_propensity(gridpoint, cell_type))
        return propensities_list

    def net_propensity(self, gridpoint):
        # return sum(f(cell_count) for f in [birth_propensity, death_propensity])
        # Tested Sun 8/12 11:30pm
        return sum(self.propensities(gridpoint))

    def T_R(self, gridpoint):
        """Returns the macroscopic time constant for a compartment.
        Tested Sun 8/12 11:25pm"""
        try:
            return 1. / (self.net_propensity(gridpoint))
        except ZeroDivisionError:
            return float('Inf')

    def T_R_min(self):
        """Returns T_R^min, as described in Choi's thesis.
        Tested Mon 7/2 6:30pm"""
        return min([self.T_R(gridpoint) for gridpoint in self.dictionary])

    def plot_and_save(self, filename, file_type='.png',
             output_dir=SAVE_DIR,
             x_min=-30, x_max=30, y_min=-30, y_max=30):
        """Plots and saves the xy cross section of the tumor.
        Tested Sun 8/12 11:32pm"""
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        for gridpoint in self.dictionary:
            for cell in self.dictionary[gridpoint]:
                if x_min <= cell.coords[0] <= x_max and y_min <= cell.coords[1] <= y_max:
                    ax.plot(cell.coords[0], cell.coords[1],
                            marker='.', markerfacecolor=CELL_COLORS[cell.cell_type],
                            markersize=8, markeredgewidth=0.3, markeredgecolor='k')
        ax.set_aspect('equal')
        fig.savefig(output_dir + filename + file_type)

    def print(self):
        """Print function for debugging.
        Tested Fri 6/29 6:13am"""
        for gridpoint in self.dictionary:
            print(gridpoint, ':')
            for cell in self.dictionary[gridpoint]:
                print(cell.to_string())

# Global functions
def possible_mutations(cell_type):
    subdict = {}
    for mutation in MUTATION_PROBS:
        if mutation[0] == cell_type:
            subdict[mutation] = MUTATION_PROBS[mutation]
    return subdict

# Operator-splitting algorithm implementation.
# Initialize grid:
grid = Grid()
grid.add_cell()
t = 0
iteration = 0
images = []
added_subclone = False

while grid.total_cell_count < MAX_CELL_COUNT:
    iteration += 1
    print('Time: ', t, 'A:', grid.cell_types_count['A'], 'B:', grid.cell_types_count['B'])
    # (a) Determine system state type:
    try:
        F = float(grid.T_R_min() / tau_D)
    except ValueError:
        print('ERROR: All cells have died under this initial seed.')
        print('Simulation discontinued')
        break
    # (b) Compute time-step:
    if F < k1:
        dt = k2 * tau_D
    elif F > k1_p:
        dt = 10 * tau_D
    else:  # k1 ≤ F ≤ k1_p
        dt = k2_p * tau_D
    # (c) Reset time:
    t_old = t
    # (d) Perform diffusion and reaction steps:
    #    (i) Diffusion step:
    temp_grid = Grid()
    for gridpoint in grid.dictionary:
        for cell in grid.dictionary[gridpoint]:
            cell.brownian_diffuse(dt)
            temp_grid.add_cell(cell.coords, cell.cell_type)
    grid = temp_grid
    del temp_grid
    #    (ii) Reaction step:
    for gridpoint in grid.dictionary:
        while t < t_old + dt:
            r1 = rand.sample()
            r2 = rand.sample()
            tau_R = grid.T_R(gridpoint) * np.log(1. / r1)
            if tau_R <= dt:  # Reaction occurs
                propensities_list = grid.propensities(gridpoint)
                for i in range(2 * len(CELL_TYPES_LIST)):
                    if r2 < sum(propensities_list[:i + 1]) / grid.net_propensity(gridpoint):
                        # Corresponding reaction occurs
                        if i % 2 == 0:
                            # birth occurs
                            reacting_cell_type = CELL_TYPES_LIST[int(i / 2)]
                            random_coords = grid.choose_random_cell(gridpoint, reacting_cell_type)
                            if not added_subclone and grid.total_cell_count > SEEDING_COUNT_THRESHOLD:
                                grid.add_cell(random_coords, 'B')
                                added_subclone = True
                            else:
                                grid.add_cell(random_coords, reacting_cell_type)
                        else:
                            # death occurs
                            reacting_cell_type = CELL_TYPES_LIST[int((i - 1) / 2)]
                            random_coords = grid.choose_random_cell(gridpoint, reacting_cell_type)
                            grid.remove_cell(random_coords, reacting_cell_type)
                        break
                t = t + tau_R
            else:
                t = t_old + dt
        t = t_old
    grid.update_gridpoints()
    # (e) Synchronize t across all cells
    t = t_old + dt
    # (f) Save grid as jpeg
    filename = 'output-graphic-' + str(iteration).zfill(3)
    grid.plot_and_save(filename)
    filepath = os.path.join(SAVE_DIR, filename + '.png')
    images.append(imageio.imread(filepath))

imageio.mimsave(SAVE_DIR + 'animation.gif', images, duration=5./len(images))  # Creates a 5 second gif
print('Animation successfully assembled')
