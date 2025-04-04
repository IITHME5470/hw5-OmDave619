# Report - HW5
--------------
Author: Om Dave <br>
Roll: CO22BTECH11006
-------------------

---

## Part (a): Contour and Line Plots

### Contour Plots at Selected Time Steps

Below are the contour plots of the temperature field at three selected time steps (other than t = 0) for both the serial and the parallel runs.

#### Serial Code Contour Plots

- **Time Step 1:**  
  ![Serial Contour Plot - Time Step 1](hw5/cont_T_2553.png)

- **Time Step 2:**  
  ![Serial Contour Plot - Time Step 2](hw5/cont_T_5106.png)

- **Time Step 3:**  
  ![Serial Contour Plot - Time Step 3](hw5/cont_T_12769.png)

#### Parallel Code Contour Plots

*For each processor configuration (2×2, 4×1), include the corresponding contour plots.*

- NOTE: Due to less number of cores only 2x2 and 4x1 processor configurations are included.*

- **Parallel (2×2) - Time Step 1:**  
  ![Parallel 2x2 Contour Plot - Time Step 1](/hw5/cont_T_parallel_002552_2x2.png)

- **Parallel (4×1) - Time Step 1:**  
  ![Parallel 4x1 Contour Plot - Time Step 1](hw5/cont_T_parallel_002552_4x1.png)

- **Parallel (2×2) - Time Step 2:**
  ![Parallel 2x2 Contour Plot - Time Step 2](hw5/cont_T_parallel_005104_2x2.png)
- **Parallel (4×1) - Time Step 2:**
  ![Parallel 4x1 Contour Plot - Time Step 2](hw5/cont_T_parallel_005104_4x1.png)
- **Parallel (2×2) - Time Step 3:**
  ![Parallel 2x2 Contour Plot - Time Step 3](hw5/cont_T_parallel_012768_2x2.png)
- **Parallel (4×1) - Time Step 3:**
  ![Parallel 4x1 Contour Plot - Time Step 3](hw5/cont_T_parallel_012768_4x1.png)


### Line Plot Comparison

The line plot below compares the mid-y temperature profiles from the serial and parallel runs at a selected time step.

![Line Plot Comparison](/hw5/line_profile_comparison_012769.png)

*You can use the provided Python script (e.g., `plot_comparison.py`) to generate this plot.*

---

## Part (b): Tabulated Differences Between Serial and Parallel Runs

The following table shows the differences between the serial and the parallel runs at the final time step. The differences should be nearly at machine precision.

| Metric               | Value               |
| -------------------- | ------------------- |
| Maximum Absolute Difference | 2.5e-5 |
| L2 Norm Difference          | 3.5e-6 |

![alt text](image.png)

---

## Part (c): Timing per Time Step

The timing information (time per time step) for the serial and MPI-parallel runs are tabulated below. In the serial code, timing was measured using standard C clock functions; in the MPI code, `MPI_Wtime()` was used.

- Serial Timing:
![alt text](image-2.png)

- Parallel (2×2) Timing:
![alt text](image-1.png)

---

