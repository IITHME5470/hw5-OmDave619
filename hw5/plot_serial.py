import numpy as np
import matplotlib.pyplot as plt

# ----- Parameters -----
tid = 2553
filename = f"T_x_y_{tid:06d}.dat"

# ----- Read Data -----
# Assumes the file has three columns: x, y, T.
a = np.loadtxt(filename)

# Determine n from the total number of rows
n = int(np.sqrt(a.shape[0]))

# Extract x and y coordinates:
# MATLAB code: x = a(1:n:n^2,1)
# In Python (0-indexed), take every n-th element starting at index 0.
x = a[0::n, 0]
# MATLAB: y = a(1:n,2) -> first n rows, column 2
y = a[:n, 1]

# Reshape the temperature column into an n-by-n matrix.
T = a[:, 2].reshape((n, n))

# ----- Plot Contours -----
plt.figure()
# Note: The MATLAB code transposes T (i.e. T') for plotting.
contour = plt.contourf(x, y, T.T, levels=50, cmap='jet')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f't = {tid:06d}')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.clim([-0.05, 1.05])
plt.colorbar(contour)
plt.savefig(f'cont_T_{tid:04d}.png', dpi=300)
plt.show()

# ----- Plot Mid-y Profile -----
plt.figure()
# Extract the column at mid-y; use integer division.
Tmid = T[:, n // 2]
plt.plot(x, Tmid, '-', linewidth=2)
plt.xlabel('x')
plt.ylabel('T')
plt.title(f'Profile along mid-y at t={tid:06d}')
plt.xlim([-0.05, 1.05])
plt.savefig(f'line_midy_T_{tid:04d}.png', dpi=300)
plt.show()
