import os
import random
import numpy as np
from numpy import random as rand
rand.seed(2) # Seed is chosen so that tumor doesn't go extinct
import matplotlib.pyplot as plt
import imageio

save_dir = '/Users/Andrew/PycharmProjects/Tumor_Simulation/output/'

dim = 3  # Number of dimensions
dx = 1  # Grid spacing
D = .05  # Diffusion constant
tau_D = 1/(2*dim) * dx**2/D  # Diffusion time constant

rate_b = .1  # Birth rate of cell A
# TO DO: for multiple cell types, rate_b will be a dictionary indicating birth rates of each cell
rate_d = 0.05  # Death rate
carrying_capacity = 20  # Number of sustainable cells at each gridpoint

k1 = (1 - 0.33)/np.log(2)
k1_p = (1 + 0.33)/np.log(2)
k2 = 2
k2_p = 3

cell_types_list = ('A',)
cell_colors = {'A':'b'}

t_final = 180.

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

    def __init__(self, scale=1/dx):
        # Grid is a dictionary with tuples representing discrete lattice points
        #    and values representing sets of cells nearest to that point
        # Scale is the number of gridpoints per unit
        # Tested Thu 6/28 9:16pm
        self.dictionary = {}
        self.scale = scale

    def cell_count(self, gridpoint):
        # Returns the number of cells at a given gridpoint
        # Tested Mon 7/2 12:42pm
        count = 0
        for cell in self.dictionary[gridpoint]:
            count += 1
        return count

    def cell_type_count(self, gridpoint, cell_type):
        # Returns the number of cells of a specified type at a given gridpoint
        # Tested Mon 7/2 12:43pm
        count = 0
        for cell in self.dictionary[gridpoint]:
            if cell.cell_type == cell_type:
                count = count + 1
        return count

    def add_cell(self, coords=np.zeros(dim), cell_type='A'):
        # Adds cell as a value to key corresponding to discrete gridpoint
        #    If no gridpoint is in dictionary, create a list containing that cell
        #    Otherwise, add it to the existing value list.
        # Tested Thu 6/28 9:54pm
        # Edited Fri 7/6 3:31pm
        gridpoint = tuple([round(i * self.scale) / self.scale for i in coords])
        if gridpoint not in self.dictionary:
            self.dictionary[gridpoint] = [Cell(coords, cell_type)]
        else:
            self.dictionary[gridpoint].append(Cell(coords, cell_type))

    def remove_cell(self, coords, cell_type):
        # Removes a cell from the grid.
        # Does not remove empty gridpoints.
        # Tested Thu 6/28 11:11am
        # Edited Fri 7/9 5:38pm
        gridpoint = tuple([round(i * self.scale) / self.scale for i in coords])
        if gridpoint not in self.dictionary:
            raise KeyError('No such cell to remove.')
        else:
            for cell in self.dictionary[gridpoint]:
                if all(np.equal(cell.coords, coords)) and cell.cell_type == cell_type:
                    self.dictionary[gridpoint].remove(cell)
                    break

    def update_gridpoints(self):
        # Remove any gridpoints which do not contain any cells.
        null_gridpoints = []
        for gridpoint in self.dictionary:
            if len(self.dictionary[gridpoint]) == 0:
                null_gridpoints.append(gridpoint)
        for gridpoint in null_gridpoints:
            del self.dictionary[gridpoint]

    def copy(self):
        # Tested Thu 7/5 9:55pm
        new_grid = Grid()
        for gridpoint in self.dictionary:
            for cell in self.dictionary[gridpoint]:
                new_grid.add_cell(cell.coords, cell.cell_type)
        return new_grid

    # Propensity functions:
    def birth_propensity(self, gridpoint, cell_type):
        # Edited Fri 7/6 6:51pm
        return self.cell_type_count(gridpoint, cell_type) * rate_b * (1 - self.cell_count(gridpoint) / carrying_capacity)

    def death_propensity(self, gridpoint, cell_type):
        # Edited Fri 7/6 6:53pm
        return self.cell_type_count(gridpoint, cell_type) * rate_d

    def net_propensity(self, gridpoint):
        # return sum(f(cell_count) for f in [birth_propensity, death_propensity])
        # Edited Sat 7/7 3:18pm
        total = 0
        for cell_type in cell_types_list:
            total += self.birth_propensity(gridpoint, cell_type) + self.death_propensity(gridpoint, cell_type)
        return total

    # Macroscopic time constant
    def T_R(self, gridpoint):
        # Tested Mon 7/2 6:30pm
        # Edited Sat 7/7 3:22pm
        try:
            return 1. / (self.net_propensity(gridpoint))
        except ZeroDivisionError:
            return float('Inf')

    def T_R_min(self):
        # Tested Mon 7/2 6:30pm
        return min([self.T_R(gridpoint) for gridpoint in self.dictionary])

    def plot(self, filename, file_type='.png',
             output_dir=save_dir,
             x_min=-30, x_max=30, y_min=-30, y_max=30):
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        for gridpoint in self.dictionary:
            for cell in self.dictionary[gridpoint]:
                if (x_min <= cell.coords[0] <= x_max and y_min <= cell.coords[1] <= y_max):
                    ax.plot(cell.coords[0], cell.coords[1],
                            marker='o', markerfacecolor='b', markeredgecolor='k')
        ax.set_aspect('equal')
        fig.savefig(output_dir + filename + file_type)

    def print(self):
        # Print function for debugging.
        # Tested Fri 6/29 6:13am
        for gridpoint in self.dictionary:
            print(gridpoint, ':')
            for cell in self.dictionary[gridpoint]:
                print(cell.to_string())

# Operator-splitting algorithm implementation.
# Initialize grid:
grid = Grid()
grid.add_cell()
t = 0
iteration = 0
images = []

while t < t_final:
    iteration += 1
    print('Time: ', t)
    # (a) Determine system state type:
    F = float(grid.T_R_min() / tau_D)
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
            tau_R = grid.T_R(gridpoint) * np.log(1./r1)
            if tau_R <= dt:  # Reaction occurs
                random_cell = random.choice(grid.dictionary[gridpoint])  # Select random cell to "react"
                if r2 < grid.birth_propensity(gridpoint, 'A') / grid.net_propensity(gridpoint):
                    # Birth occurs at the exact location of a randomly selected existing cell
                    grid.add_cell(random_cell.coords, random_cell.cell_type)
                else:
                    # One randomly chosen cell at the gridpoint dies
                    grid.remove_cell(random_cell.coords, random_cell.cell_type)
                t = t + tau_R
            else:
                t = t_old + dt
        t = t_old
    grid.update_gridpoints()
    # (e) Synchronize t across all cells
    t = t_old + dt
    # (f) Save grid as jpeg
    filename = 'output-graphic-' + str(iteration).zfill(3)
    grid.plot(filename)
    filepath = os.path.join(save_dir, filename + '.png')
    images.append(imageio.imread(filepath))

imageio.mimsave('/Users/Andrew/PycharmProjects/Tumor_Simulation/output/animation.gif', images, duration = 0.2)

