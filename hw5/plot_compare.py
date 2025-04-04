#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

def read_serial_solution(tid):
    filename = f"T_x_y_{tid:06d}.dat"
    data = np.loadtxt(filename)
    n = int(np.sqrt(data.shape[0]))
    x = data[0::n, 0]
    y = data[:n, 1]
    T = data[:, 2].reshape((n, n))
    return x, y, T

def read_parallel_solution(tid, px, py):
    file_pattern = f"T_x_y_{tid:06d}_*_par.dat"
    file_list = sorted(glob.glob(file_pattern))
    if len(file_list) != px * py:
        raise ValueError(f"Expected {px*py} files, but found {len(file_list)}.")
    local_data = {}
    rank_regex = re.compile(r'T_x_y_\d{6}_(\d{4})_par\.dat')
    local_nx, local_ny = None, None
    for fname in file_list:
        match = rank_regex.search(fname)
        if not match:
            continue
        rank = int(match.group(1))
        data = np.loadtxt(fname)
        x_local = np.unique(data[:, 0])
        y_local = np.unique(data[:, 1])
        if local_nx is None:
            local_nx = len(x_local)
            local_ny = len(y_local)
        T_local = data[:, 2].reshape((local_nx, local_ny))
        local_data[rank] = {'x': x_local, 'y': y_local, 'T': T_local}
    global_nx = px * local_nx
    global_ny = py * local_ny
    global_T = np.zeros((global_nx, global_ny))
    global_x = np.zeros(global_nx)
    global_y = np.zeros(global_ny)
    for rank, block in local_data.items():
        rank_y = rank // px
        rank_x = rank % px
        i_start = rank_x * local_nx
        i_end = (rank_x + 1) * local_nx
        j_start = rank_y * local_ny
        j_end = (rank_y + 1) * local_ny
        global_T[i_start:i_end, j_start:j_end] = block['T']
        global_x[i_start:i_end] = block['x']
        global_y[j_start:j_end] = block['y']
    return global_x, global_y, global_T

# ----- Main Comparison Plot -----
if __name__ == '__main__':
    tid = 12769         # Use the desired time step id (not t=0)
    px = 2           # Set processor grid dimensions as used in the parallel run
    py = 2

    # Read solutions
    x_serial, y_serial, T_serial = read_serial_solution(tid)
    x_parallel, y_parallel, T_parallel = read_parallel_solution(tid-1, px, py)

    # Extract a mid-y profile (choose the middle index)
    mid_index_serial = len(y_serial) // 2
    mid_index_parallel = len(y_parallel) // 2

    T_profile_serial = T_serial[:, mid_index_serial]
    T_profile_parallel = T_parallel[:, mid_index_parallel]

    # Plot comparison
    plt.figure()
    plt.plot(x_serial, T_profile_serial, 'b-', label='Serial')
    plt.plot(x_parallel, T_profile_parallel, 'r--', label='Parallel')
    plt.xlabel('x')
    plt.ylabel('Temperature T')
    plt.title(f'Mid-y Profile Comparison at time step {tid}')
    plt.legend()
    plt.xlim([-0.05, 1.05])
    plt.savefig(f'line_profile_comparison_{tid:06d}.png', dpi=300)
    plt.show()
