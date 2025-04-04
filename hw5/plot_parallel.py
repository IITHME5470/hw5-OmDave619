import numpy as np
import matplotlib.pyplot as plt
import glob
import re

# ----- User-Defined Parameters -----
# Time-step identifier to plot (must match the output file naming)
tid = 12768
# Processor grid dimensions (must match the ones used during the run)
px = 2  # number of processors in x-direction
py = 2  # number of processors in y-direction

# ----- Get List of Files -----
# Files are named like: T_x_y_003185_0000_par.dat, T_x_y_003185_0001_par.dat, etc.
file_pattern = f"T_x_y_{tid:06d}_*_par.dat"
file_list = sorted(glob.glob(file_pattern))
expected_files = px * py
if len(file_list) != expected_files:
    raise ValueError(f"Expected {expected_files} files but found {len(file_list)}.")

# ----- Read Local Data from Each File -----
# We'll store each rank's data (its unique x, y arrays and the local T array) in a dictionary.
local_data = {}
# Regular expression to extract the rank number from the filename.
# The expected pattern is: T_x_y_{tid:06d}_{rank:04d}_par.dat
rank_regex = re.compile(r'T_x_y_\d{6}_(\d{4})_par\.dat')

local_nx = None
local_ny = None

for fname in file_list:
    match = rank_regex.search(fname)
    if not match:
        print(f"Warning: file {fname} does not match expected pattern.")
        continue
    rank = int(match.group(1))
    # Load the data (each file is assumed to have three columns: x, y, T)
    data = np.loadtxt(fname)
    # Determine the local grid size from the unique x and y values
    x_local = np.unique(data[:, 0])
    y_local = np.unique(data[:, 1])
    if local_nx is None:
        local_nx = len(x_local)
        local_ny = len(y_local)
    else:
        if len(x_local) != local_nx or len(y_local) != local_ny:
            raise ValueError(f"Inconsistent local grid size in file {fname}.")
    # Reshape the T values (the files are written with an outer loop over i and inner loop over j)
    T_local = data[:, 2].reshape((local_nx, local_ny))
    local_data[rank] = {'x': x_local, 'y': y_local, 'T': T_local}

# ----- Assemble Global Data -----
# Global grid dimensions: assume the total number of x points is px * local_nx and y points is py * local_ny.
global_nx = px * local_nx
global_ny = py * local_ny

global_T = np.zeros((global_nx, global_ny))
global_x = np.zeros(global_nx)
global_y = np.zeros(global_ny)

# Combine the data from each rank.
# According to get_processor_grid_ranks, the rank's coordinates are:
#   rank_y = rank // px,  rank_x = rank % px.
for rank, block in local_data.items():
    rank_y = rank // px
    rank_x = rank % px
    # Compute the global indices for this block.
    i_start = rank_x * local_nx
    i_end = (rank_x + 1) * local_nx
    j_start = rank_y * local_ny
    j_end = (rank_y + 1) * local_ny
    # Place the local T block into the global T array.
    global_T[i_start:i_end, j_start:j_end] = block['T']
    # For the x-coordinates, assume all blocks in the same processor column share the same x values.
    global_x[i_start:i_end] = block['x']
    # For the y-coordinates, assume all blocks in the same processor row share the same y values.
    global_y[j_start:j_end] = block['y']

# ----- Create Meshgrid for Plotting -----
# Note: The output from the C code uses x for the first index and y for the second.
# We create a meshgrid with 'ij' indexing to preserve that order.
X, Y = np.meshgrid(global_x, global_y, indexing='ij')

# ----- Plot Global Contour -----
plt.figure()
contour = plt.contourf(X, Y, global_T.T, levels=50, cmap='jet')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Parallel solution at t = {tid:06d}')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.clim([-0.05, 1.05])
plt.colorbar(contour)
plt.savefig(f'cont_T_parallel_{tid:06d}.png', dpi=300)
plt.show()
