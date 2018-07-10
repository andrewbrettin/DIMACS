import random
import numpy as np
# import parameters.py as p
from numpy import random as rand
rand.seed(0)

# Parameter values
dim = 3  # Number of dimensions
dx = 1  # Grid spacing
D = 1  # Diffusion constant
tau_D = 1/(2*dim) * dx**2/D  # Diffusion time constant

rate_b = 2  # Birth rate of cell A
# TO DO: for multiple cell types, rate_b will be a list or a tuple indicating birth rates of each cell
rate_d = 1  # Death rate
carrying_capacity = 20  # Number of sustainable cells at each gridpoint

k1 = (1 - 0.33)/np.log(2)
k1_p = (1 + 0.33)/np.log(2)
k2 = 2
k2_p = 3

cell_types_list = frozenset({'A'})

t_final = 100

# Class Definitions


class Cell:
    """A cell object contains the information about a tumor cell"""

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

    def brownian_diffuse(self, time):
        # Diffuses cell using Smoluchowski diffusion
        # Tested Th 6/28 9:13pm
        new_coords = self.coords + np.sqrt(2 * D * time) * rand.normal(0, 1, len(self.coords))
        self.move(new_coords)

    def equals(self, cell):
        # Tested Th 6/28 10:43pm
        return all(self.coords == cell.coords) and self.cell_type == cell.cell_type

    def to_string(self):
        # For debugging
        # Tested Th 6/28 8:45pm
        s = ''
        for i in range(len(self.coords)):
            s = s + ' ' + str(self.coords[i])
        return s


class Grid:
    """Contains information about all of the cells (location, quantity, and)"""

    def __init__(self, scale=1/dx):
        # Grid is a dictionary with tuples representing discrete lattice points
        #   and values representing sets of cells nearest to that point
        # Scale is the number of gridpoints per unit
        # Tested Th 6/28 9:16pm
        self.dictionary = {}
        self.scale = scale

    def cell_count(self, gridpoint):
        # Returns the number of cells at a given gridpoint
        # Tested M 7/2 12:42pm
        count = 0
        for cell in self.dictionary[gridpoint]:
            count = count + 1
        return count

    def cell_type_count(self, gridpoint, cell_type):
        # Returns the number of cells of a specified type at a given gridpoint
        # Tested M 7/2 12:43pm
        count = 0
        for cell in self.dictionary[gridpoint]:
            if cell.cell_type == cell_type:
                count = count + 1
        return count

    def add_cell(self, cell=Cell()):
        # Adds cell as a value to key corresponding to discrete gridpoint
        # If no gridpoint is in dictionary, create a set containing that cell
        # Otherwise, add it to the existing value set.
        # Tested Th 6/28 9:54pm
        gridpoint = tuple([round(i * self.scale) / self.scale for i in cell.coords])
        if gridpoint not in self.dictionary:
            self.dictionary[gridpoint] = {cell}
        else:
            self.dictionary[gridpoint].add(cell)

    def remove_cell(self, cell):
        # Tested Th 6/28 11:11am
        gridpoint = tuple([round(i * self.scale) / self.scale for i in cell.coords])
        if gridpoint not in self.dictionary:
            raise KeyError('No such cell to remove.')
        else:
            for cell_i in self.dictionary[gridpoint]:
                if cell.equals(cell_i):
                    self.dictionary[gridpoint].remove(cell_i)
                    if len(self.dictionary[gridpoint]) == 0:
                        self.dictionary.pop(gridpoint)
                    break

    # Propensity functions:
    def birth_propensity(self, gridpoint):
        return rate_b * (1 - self.cell_count(gridpoint) / carrying_capacity)

    def death_propensity(self, gridpoint):
        return rate_d

    def net_propensity(self, gridpoint):
        # return sum(f(cell_count) for f in [birth_propensity, death_propensity])
        return self.birth_propensity(gridpoint) - self.death_propensity(gridpoint)

    # Macroscopic time constant
    def T_R(self, gridpoint):
        # Tested M 7/2 6:30pm
        return 1. / (self.net_propensity(gridpoint))

    def T_R_min(self):
        # Tested M 7/2 6:30pm
        return min([self.T_R(gridpoint) for gridpoint in self.dictionary])

    def print(self):
        # Print function for debugging.
        # Tested F 6/29 6:13am
        for gridpoint in self.dictionary:
            print(gridpoint, ':')
            for cell in self.dictionary[gridpoint]:
                print(cell.to_string())

### Simulation script ###

# Operator-splitting algorithm implementation.
grid = Grid()
grid.add_cell()
t = 0
while t < t_final:
    # (a) Determine system state type:
    F = float(grid.T_R_min() / tau_D)
    # (b) Compute time-step:
    if F < k1:
        dt = k2 * tau_D
    elif F > k1_p:
        dt = 10 * tau_D
    else: # k1 ≤ F ≤ k1_p
        dt = k2_p * tau_D
    # (c) Reset time:
    t_old = t
    # (d) Perform diffusion and reaction steps:
    #    (i) Diffusion:
    temp_grid = Grid(grid.scale)
    for gridpoint in grid.dictionary:
        for cell in grid.dictionary[gridpoint]:
            cell.brownian_diffuse(dt)
            temp_grid.add_cell(cell)
    grid = temp_grid
    del temp_grid
    #    (ii) Reaction:
    for gridpoint in grid.dictionary:
        while t < t_old + dt:
            r1 = rand.sample()
            r2 = rand.sample()
            tau_R = grid.T_R(gridpoint) * np.log(1./r1)
            if tau_R <= dt:
                if r2 < grid.birth_propensity(gridpoint) / grid.net_propensity(gridpoint):
                    # Birth occurs at location of a randomly selected existing cell
                    grid.add_cell(random.choice(tuple(grid.dictionary[gridpoint])))
                else:
                    # One randomly chosen cell at the gridpoint dies
                    grid.remove_cell(random.choice(tuple(grid.dictionary[gridpoint])))
            else:
                t = t_old + dt
        t = t_old
    # (e) Synchronize t across all cells
    t = t_old + dt

