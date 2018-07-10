import math
import random
import numpy as np
from numpy import random as rand

rand.seed(0)

dim = 3  # Number of dimensions
dx = 1  # Grid spacing
D = 1  # Diffusion constant
tau_D = 1 / (2 * dim) * dx ** 2 / D  # Diffusion time constant

rate_b = 2  # Birth rate of cell A
# TO DO: for multiple cell types, rate_b will be a list or a tuple indicating birth rates of each cell
rate_d = 1  # Death rate
carrying_capacity = 20  # Number of sustainable cells at each gridpoint

k1 = (1 - 0.33) / np.log(2)
k1_p = (1 + 0.33) / np.log(2)
k2 = 2
k2_p = 3

cell_types_list = ('A',)

t_final = 100.


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

    def __init__(self, coords=np.zeros(dim, dtype=np.float_), cell_type='A'):
        # Coords can be a tuple or an numpy array
        # Tested Th 6/28 8:45pm
        if len(coords) == dim:
            self.coords = np.array(coords, dtype=np.float_)
        else:
            raise ValueError
        if cell_type in cell_types_list:
            self.cell_type = cell_type
        else:
            raise ValueError('Cell type is invalid')

    def move(self, new_coords):
        # Sets cell coordinates to a new numpy array
        # Tested Th 6/28 8:49pm
        self.coords = np.array(new_coords, dtype=np.float_)

    def brownian_diffuse(self, dt):
        # Diffuses cell using Smoluchowski diffusion
        # Tested Th 6/28 9:13pm
        new_coords = self.coords + np.sqrt(2 * D * dt) * rand.normal(0, 1, len(self.coords))
        self.move(new_coords)

    def equals(self, cell):
        # Tested Th 6/28 10:43pm
        return all(self.coords == cell.coords) and self.cell_type == cell.cell_type

    def copy(self):
        # Tested Th 7/5 9:55pm
        return Cell(self.coords, self.cell_type)

    def to_string(self):
        # For debugging
        # Tested Th 6/28 8:45pm
        s = ''
        for i in range(len(self.coords)):
            s = s + ' ' + str(self.coords[i])
        return '(Type ' + self.cell_type + ')' + s


class Grid:
    """Contains information about all of the cells (location, quantity, and types.)
        Instance variables:
        - dictionary
        - scale
        Methods:
        - cell_count(gridpoint)
        - cell_type_count(gridpoint, cell_type)
        - add_cell(coords, cell_type)
        - remove_cell(coords, cell_type)
        - copy()
        - birth_propensity(gridpoint, cell_type)
        - death_propensity(gridpoint, cell_type)
        - net_propensity(gridpoint, cell_type)
        - T_R(gridpoint)
        - T_R_min()
        - plot()
        - print()
    """

    def __init__(self, scale=1 / dx):
        # Grid is a dictionary with tuples representing discrete lattice points
        ### and values representing sets of cells nearest to that point
        # Scale is the number of gridpoints per unit
        # Tested Thu 6/28 9:16pm
        self.dictionary = {}
        self.scale = scale

    def cell_count(self, gridpoint):
        count = 0
        for _ in self.dictionary[gridpoint]:
            count += 1
        return count

    def cell_type_count(self, gridpoint, cell_type):
        count = 0
        for cell in self.dictionary[gridpoint]:
            if cell.cell_type == cell_type:
                count += 1
        return count

    def add_cell(self, coords=np.zeros(dim), cell_type='A'):
        gridpoint = tuple([round(i * self.scale) / self.scale for i in coords])
        if gridpoint not in self.dictionary:
            self.dictionary[gridpoint] = [Cell(coords, cell_type)]
        else:
            self.dictionary[gridpoint].append(Cell(coords, cell_type))

    def remove_cell(self, coords, cell_type):
        gridpoint = tuple([round(i * self.scale) / self.scale for i in coords])
        if gridpoint not in self.dictionary:
            raise KeyError('No such cell to remove')
        else:
            for cell in self.dictionary[gridpoint]:
                if all(np.equal(cell.coords, coords)) and cell.cell_type == cell_type:
                    self.dictionary[gridpoint].remove(cell)
                    if len(self.dictionary[gridpoint]) == 0:
                        del self.dictionary[gridpoint]
                    break

    def copy(self):
        new_grid = Grid()
        for gridpoint in self.dictionary:
            for cell in self.dictionary[gridpoint]:
                new_grid.add_cell(cell.coords, cell.cell_type)
        return new_grid

    # Propensity functions:
    def birth_propensity(self, gridpoint, cell_type):
        # Edited Fri 7/6 6:51pm
        return self.cell_type_count(gridpoint, cell_type) * rate_b * (
                    1 - self.cell_count(gridpoint) / carrying_capacity)

    def death_propensity(self, gridpoint, cell_type):
        # Edited Fri 7/6 6:53pm
        return self.cell_type_count(gridpoint, cell_type) * rate_d

    def net_propensity(self, gridpoint):
        # Edited Sat 7/7 3:18pm
        total = 0
        for cell_type in cell_types_list:
            total += self.birth_propensity(gridpoint, cell_type) + self.death_propensity(gridpoint, cell_type)
        return total

    # Macroscopic time constant
    def T_R(self, gridpoint):
        # Tested Mon 7/2 6:30pm
        # Edited Sat 7/7 3:22pm
        return 1. / (self.net_propensity(gridpoint))

    def T_R_min(self):
        # Tested Mon 7/2 6:30pm
        return min([self.T_R(gridpoint) for gridpoint in self.dictionary])

    # def plot(self, )

    def print(self):
        # Print function for debugging.
        # Tested Fri 6/29 6:13am
        for gridpoint in self.dictionary:
            print(gridpoint, ':')
            for cell in self.dictionary[gridpoint]:
                print(cell.to_string())

# Comment