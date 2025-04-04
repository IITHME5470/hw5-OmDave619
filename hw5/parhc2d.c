#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h> // For memcpy if needed, though loops are clearer here

/* ------------------------------------------------------------------
   1. Domain-Decomposition Helper
   ------------------------------------------------------------------ */
void get_processor_grid_ranks(int rank, int size, int px, int py,
                              int *rank_x, int *rank_y)
{
  // Calculate the 2D processor grid coordinates (rank_x, rank_y)
  // based on the linear rank. Assumes row-major mapping (or adjust if needed).
  // Standard mapping: ranks fill row by row.
  // Example px=4, py=2:
  // Rank 0: (0,0)  Rank 1: (1,0)  Rank 2: (2,0)  Rank 3: (3,0)
  // Rank 4: (0,1)  Rank 5: (1,1)  Rank 6: (2,1)  Rank 7: (3,1)
  // *rank_x = rank % px;
  // *rank_y = rank / px;

  // Let's try the mapping implied by the main logic (istglob/jstglob calculations):
  // Example px=2, py=4:
  // Rank 0: (0, 0) -> ist=0*nx, jst=0*ny
  // Rank 1: (1, 0) -> ist=1*nx, jst=0*ny
  // Rank 2: (0, 1) -> ist=0*nx, jst=1*ny
  // Rank 3: (1, 1) -> ist=1*nx, jst=1*ny
  // Rank 4: (0, 2) -> ist=0*nx, jst=2*ny
  // Rank 5: (1, 2) -> ist=1*nx, jst=2*ny
  // This seems to correspond to:
  *rank_x = rank % px; // Column index
  *rank_y = rank / px; // Row index
}

/* ------------------------------------------------------------------
   2. Grid Generation
   ------------------------------------------------------------------ */
void grid(int nx_local, int nx_global,
          int istglob, int ienglob, // Global start/end indices for this rank
          double xstglob, double xenglob,
          double *x, double *dx)
{
  int i;

  // Calculate global dx (same for all processes)
  *dx = (xenglob - xstglob) / (double)(nx_global - 1);

  // Calculate local coordinates based on global indices
  for (i = 0; i < nx_local; i++) {
    // The i-th local point corresponds to the (istglob + i)-th global point
    x[i] = xstglob + (double)(istglob + i) * (*dx);
  }
}

/* ------------------------------------------------------------------
   3. Boundary Conditions, Initial Condition
   ------------------------------------------------------------------ */
void enforce_bcs(int nx, int ny, // Local dimensions
                 int rank_x, int rank_y, // Processor coordinates
                 int px, int py,         // Processor grid size
                 double **T)
{
  int i, j;

  // Apply BCs ONLY if the process is on the GLOBAL boundary

  // Left global boundary (x=0)
  if (rank_x == 0) {
    for (j = 0; j < ny; j++) {
      T[0][j] = 0.0;
    }
  }

  // Right global boundary (x=1)
  if (rank_x == px - 1) {
    for (j = 0; j < ny; j++) {
      T[nx - 1][j] = 0.0;
    }
  }

  // Bottom global boundary (y=0)
  if (rank_y == 0) {
    for (i = 0; i < nx; i++) {
      T[i][0] = 0.0;
    }
  }

  // Top global boundary (y=1)
  if (rank_y == py - 1) {
    for (i = 0; i < nx; i++) {
      T[i][ny - 1] = 0.0;
    }
  }
}

// Helper for initial condition function
double initial_T(double x, double y, double dx, double dy) {
    double del = 1.0; // As in serial code
    // Use the physical coordinates x, y directly
    return 0.25 * (tanh((x - 0.4) / (del * dx)) - tanh((x - 0.6) / (del * dx))) *
                  (tanh((y - 0.4) / (del * dy)) - tanh((y - 0.6) / (del * dy)));
}


void set_initial_condition(int nx, int ny,     // Local dimensions
                           int rank_x, int rank_y, // Processor coordinates
                           int px, int py,         // Processor grid size
                           double *x, double *y, // Local coordinate arrays
                           double **T,
                           double dx, double dy) // Global grid spacings
{
  int i, j;

  for (i = 0; i < nx; i++) {
    for (j = 0; j < ny; j++) {
        // Use the local coordinate arrays x[i], y[j] which hold the correct physical values
        T[i][j] = initial_T(x[i], y[j], dx, dy);
    }
  }

  // Ensure BCs are satisfied at t = 0 on global boundaries
  enforce_bcs(nx, ny, rank_x, rank_y, px, py, T);
}

/* ------------------------------------------------------------------
   4. Halo Exchange Routines
   ------------------------------------------------------------------ */

// Exchanges data with Left/Right neighbors
void halo_exchange_2d_x(int rank, int rank_x, int rank_y,
                        int size, int px, int py,
                        int nx, int ny,
                        double **T,
                        double *xleftghost, double *xrightghost,
                        double *sendbuf_x, double *recvbuf_x)
{
    MPI_Status status;
    int tag_left = 0;  // Tag for sending left, receiving from right
    int tag_right = 1; // Tag for sending right, receiving from left

    // Determine neighbors based on rank_x and rank_y
    int left_nbr = (rank_x > 0) ? rank - 1 : MPI_PROC_NULL;
    int right_nbr = (rank_x < px - 1) ? rank + 1 : MPI_PROC_NULL;

    // --- Exchange data with left/right neighbors ---

    // 1. Send data to the right neighbor (my last column T[nx-1][j])
    //    and receive data from the left neighbor into xleftghost
    if (right_nbr != MPI_PROC_NULL) {
        for (int j = 0; j < ny; ++j) {
            sendbuf_x[j] = T[nx - 1][j]; // Pack last column
        }
    }
    MPI_Sendrecv(sendbuf_x, ny, MPI_DOUBLE, right_nbr, tag_right,
                 recvbuf_x, ny, MPI_DOUBLE, left_nbr, tag_right,
                 MPI_COMM_WORLD, &status);
    // If we received data (i.e., left_nbr exists), copy it to ghost buffer
    if (left_nbr != MPI_PROC_NULL) {
        for (int j = 0; j < ny; ++j) {
            xleftghost[j] = recvbuf_x[j];
        }
    } else {
        // If no left neighbor, boundary condition applies (T=0 for this problem)
        // This ghost data would be used if calculating RHS at i=0.
         for (int j = 0; j < ny; ++j) {
            xleftghost[j] = 0.0;
        }
    }


    // 2. Send data to the left neighbor (my first column T[0][j])
    //    and receive data from the right neighbor into xrightghost
    if (left_nbr != MPI_PROC_NULL) {
        for (int j = 0; j < ny; ++j) {
            sendbuf_x[j] = T[0][j]; // Pack first column
        }
    }
     MPI_Sendrecv(sendbuf_x, ny, MPI_DOUBLE, left_nbr, tag_left,
                 recvbuf_x, ny, MPI_DOUBLE, right_nbr, tag_left,
                 MPI_COMM_WORLD, &status);
    // If we received data (i.e., right_nbr exists), copy it to ghost buffer
    if (right_nbr != MPI_PROC_NULL) {
        for (int j = 0; j < ny; ++j) {
            xrightghost[j] = recvbuf_x[j];
        }
    } else {
        // If no right neighbor, boundary condition applies (T=0 for this problem)
        // This ghost data would be used if calculating RHS at i=nx-1.
        for (int j = 0; j < ny; ++j) {
            xrightghost[j] = 0.0;
        }
    }
}


// Exchanges data with Bottom/Top neighbors
void halo_exchange_2d_y(int rank, int rank_x, int rank_y,
                        int size, int px, int py,
                        int nx, int ny,
                        double **T,
                        double *ybotghost, double *ytopghost,
                        double *sendbuf_y, double *recvbuf_y)
{
    MPI_Status status;
    int tag_bot = 2; // Tag for sending bottom, receiving from top
    int tag_top = 3; // Tag for sending top, receiving from bottom

    // Determine neighbors based on rank_x and rank_y
    int bot_nbr = (rank_y > 0) ? rank - px : MPI_PROC_NULL;
    int top_nbr = (rank_y < py - 1) ? rank + px : MPI_PROC_NULL;

    // --- Exchange data with bottom/top neighbors ---

    // 1. Send data to the top neighbor (my last row T[i][ny-1])
    //    and receive data from the bottom neighbor into ybotghost
    if (top_nbr != MPI_PROC_NULL) {
        for (int i = 0; i < nx; ++i) {
            sendbuf_y[i] = T[i][ny - 1]; // Pack last row
        }
    }
    MPI_Sendrecv(sendbuf_y, nx, MPI_DOUBLE, top_nbr, tag_top,
                 recvbuf_y, nx, MPI_DOUBLE, bot_nbr, tag_top,
                 MPI_COMM_WORLD, &status);
    // If we received data (i.e., bot_nbr exists), copy it to ghost buffer
    if (bot_nbr != MPI_PROC_NULL) {
         for (int i = 0; i < nx; ++i) {
             ybotghost[i] = recvbuf_y[i];
         }
    } else {
        // If no bottom neighbor, boundary condition applies (T=0)
        for (int i = 0; i < nx; ++i) {
            ybotghost[i] = 0.0;
        }
    }


    // 2. Send data to the bottom neighbor (my first row T[i][0])
    //    and receive data from the top neighbor into ytopghost
    if (bot_nbr != MPI_PROC_NULL) {
        for (int i = 0; i < nx; ++i) {
            sendbuf_y[i] = T[i][0]; // Pack first row
        }
    }
    MPI_Sendrecv(sendbuf_y, nx, MPI_DOUBLE, bot_nbr, tag_bot,
                 recvbuf_y, nx, MPI_DOUBLE, top_nbr, tag_bot,
                 MPI_COMM_WORLD, &status);
    // If we received data (i.e., top_nbr exists), copy it to ghost buffer
    if (top_nbr != MPI_PROC_NULL) {
        for (int i = 0; i < nx; ++i) {
             ytopghost[i] = recvbuf_y[i];
         }
    } else {
        // If no top neighbor, boundary condition applies (T=0)
         for (int i = 0; i < nx; ++i) {
            ytopghost[i] = 0.0;
        }
    }
}


/* ------------------------------------------------------------------
   5. RHS Computation and Time-Stepping
   ------------------------------------------------------------------ */
void get_rhs(int nx, int ny,             // Local dimensions
             double dx, double dy,
             double *xleftghost, double *xrightghost,
             double *ybotghost, double *ytopghost,
             double kdiff,
             double **T, double **rhs)   // T is local temp, rhs is output
{
    int i, j;
    double T_im1, T_ip1, T_jm1, T_jp1;
    double dxsq = dx * dx;
    double dysq = dy * dy;

    // Calculate RHS for ALL interior points of the local domain,
    // using ghost cell data for points adjacent to processor boundaries.
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            // Get neighbors' T values, using ghost buffers for edge cases
            T_im1 = (i == 0)    ? xleftghost[j]  : T[i - 1][j];
            T_ip1 = (i == nx - 1) ? xrightghost[j] : T[i + 1][j];
            T_jm1 = (j == 0)    ? ybotghost[i]   : T[i][j - 1];
            T_jp1 = (j == ny - 1) ? ytopghost[i]   : T[i][j + 1];

            // Compute the Laplacian using central differences
            rhs[i][j] = kdiff * (T_ip1 + T_im1 - 2.0 * T[i][j]) / dxsq +
                        kdiff * (T_jp1 + T_jm1 - 2.0 * T[i][j]) / dysq;
        }
    }
    // Note: We don't need to worry about physical boundary conditions (T=0)
    // directly in this RHS calculation.
    // 1. The ghost buffers already contain 0.0 if the neighbor is MPI_PROC_NULL.
    // 2. Even if T[i][j] is on a physical boundary, calculating its RHS
    //    doesn't hurt, because enforce_bcs will overwrite T[i][j] after the
    //    Euler update step anyway.
}


void timestep_FwdEuler(int rank, int size,
                       int rank_x, int rank_y,
                       int px, int py,
                       int nx, int ny,            // Local dimensions
                       double dt, double dx, double dy, // Time/Grid steps
                       double *xleftghost, double *xrightghost,
                       double *ybotghost, double *ytopghost,
                       double kdiff,
                       double **T, double **rhs, // Input T, Output T, uses rhs internally
                       double *sendbuf_x, double *recvbuf_x,
                       double *sendbuf_y, double *recvbuf_y)
{
    int i, j;

    // 1. Exchange halo data (populate ghost buffers)
    halo_exchange_2d_x(rank, rank_x, rank_y, size, px, py,
                       nx, ny, T,
                       xleftghost, xrightghost,
                       sendbuf_x, recvbuf_x);

    halo_exchange_2d_y(rank, rank_x, rank_y, size, px, py,
                       nx, ny, T,
                       ybotghost, ytopghost,
                       sendbuf_y, recvbuf_y);

    // 2. Compute RHS using local T and received ghost data
    get_rhs(nx, ny, dx, dy,
            xleftghost, xrightghost,
            ybotghost, ytopghost,
            kdiff, T, rhs);

    // 3. Update T using Forward Euler (for all local points)
    for(i = 0; i < nx; i++) {
        for(j = 0; j < ny; j++) {
            T[i][j] += dt * rhs[i][j];
        }
    }

    // 4. Enforce physical BCs on ranks holding global boundaries
    enforce_bcs(nx, ny, rank_x, rank_y, px, py, T);
}

/* ------------------------------------------------------------------
   6. Output Routines
   ------------------------------------------------------------------ */
void output_soln(int rank, int nx, int ny, int it, double tcurr,
                 double *x, double *y, double **T)
{
    int i, j;
    FILE* fp;
    char fname[100];

    // Each rank writes its own data portion
    sprintf(fname, "T_x_y_%06d_%04d_par.dat", it, rank);
    fp = fopen(fname, "w");
    if (!fp) {
        fprintf(stderr, "Rank %d Error opening %s for writing!\n", rank, fname);
        MPI_Abort(MPI_COMM_WORLD, 1); // Abort if file cannot be opened
        return;
    }
    // Write header (optional, but can be useful)
    // fprintf(fp, "# Rank %d, Time Step %d, Time %e\n", rank, it, tcurr);
    // fprintf(fp, "# Local nx=%d, ny=%d\n", nx, ny);
    // fprintf(fp, "# x[0]=%f, x[nx-1]=%f\n", x[0], x[nx-1]);
    // fprintf(fp, "# y[0]=%f, y[ny-1]=%f\n", y[0], y[ny-1]);
    // fprintf(fp, "# X Y T\n");

    // Write data: x-coordinate, y-coordinate, Temperature
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            fprintf(fp, "%15.8e %15.8e %15.8e\n", x[i], y[j], T[i][j]);
        }
         // Add a blank line after each row in x (gnuplot likes this for pm3d map)
         // fprintf(fp, "\n");
    }
    fclose(fp);

    // Only rank 0 prints the confirmation message to avoid clutter
    // if (rank == 0) {
    //    printf("Done writing solution files for time step = %d, time = %e\n", it, tcurr);
    // }
    // Let's have each rank print its own confirmation for clarity during runs
    printf("Rank %d: Done writing %s (Time Step %d, Time %e)\n",
             rank, fname, it, tcurr);
}

// Function for the halo check requested in the prompt
void check_halo_exchange(int rank, int rank_x, int rank_y, int px, int py,
                         int nx, int ny, double **T,
                         double *xleftghost, double *xrightghost,
                         double *ybotghost, double *ytopghost,
                         double *sendbuf_x, double *recvbuf_x,
                         double *sendbuf_y, double *recvbuf_y)
{
    int i, j;
    FILE* fp_check;
    char fname_check[100];

    // 1. Assign unique values based on rank and local indices
    if (rank == 0) printf("\n--- Performing Halo Exchange Check ---\n");
    for(i = 0; i < nx; i++){
        for(j = 0; j < ny; j++){
            // Using a large base for rank to avoid overlap with i+j
            T[i][j] = (double)(rank * 10000 + i * 100 + j);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all ranks set T before exchanging

    // 2. Perform ONE halo exchange
     halo_exchange_2d_x(rank, rank_x, rank_y, px*py, px, py,
                       nx, ny, T,
                       xleftghost, xrightghost,
                       sendbuf_x, recvbuf_x);

    halo_exchange_2d_y(rank, rank_x, rank_y, px*py, px, py,
                       nx, ny, T,
                       ybotghost, ytopghost,
                       sendbuf_y, recvbuf_y);

    MPI_Barrier(MPI_COMM_WORLD); // Ensure all exchanges are complete

    // 3. Print ghost buffer contents to a file per rank
    sprintf(fname_check, "halo_check_%04d.dat", rank);
    fp_check = fopen(fname_check, "w");
    if(!fp_check) {
        fprintf(stderr, "Rank %d Error opening %s for writing!\n", rank, fname_check);
        return; // Don't abort for check file
    }

    fprintf(fp_check, "Rank %d (%d, %d) Halo Check Data\n", rank, rank_x, rank_y);

    fprintf(fp_check, "\nxleftghost (received from rank %d):\n", (rank_x > 0) ? rank - 1 : -1);
    if (rank_x > 0) { // Only print if expected to receive from left
        for (j = 0; j < ny; ++j) fprintf(fp_check, "j=%d : %f\n", j, xleftghost[j]);
    } else {
        fprintf(fp_check, "(Boundary - Should be 0.0)\n");
         for (j = 0; j < ny; ++j) fprintf(fp_check, "j=%d : %f\n", j, xleftghost[j]); // Print BCs too
    }

    fprintf(fp_check, "\nxrightghost (received from rank %d):\n", (rank_x < px - 1) ? rank + 1 : -1);
     if (rank_x < px - 1) { // Only print if expected to receive from right
        for (j = 0; j < ny; ++j) fprintf(fp_check, "j=%d : %f\n", j, xrightghost[j]);
    } else {
        fprintf(fp_check, "(Boundary - Should be 0.0)\n");
        for (j = 0; j < ny; ++j) fprintf(fp_check, "j=%d : %f\n", j, xrightghost[j]);
    }

    fprintf(fp_check, "\nybotghost (received from rank %d):\n", (rank_y > 0) ? rank - px : -1);
    if (rank_y > 0) { // Only print if expected to receive from bottom
        for (i = 0; i < nx; ++i) fprintf(fp_check, "i=%d : %f\n", i, ybotghost[i]);
    } else {
         fprintf(fp_check, "(Boundary - Should be 0.0)\n");
         for (i = 0; i < nx; ++i) fprintf(fp_check, "i=%d : %f\n", i, ybotghost[i]);
    }

    fprintf(fp_check, "\nytopghost (received from rank %d):\n", (rank_y < py - 1) ? rank + px : -1);
    if (rank_y < py - 1) { // Only print if expected to receive from top
        for (i = 0; i < nx; ++i) fprintf(fp_check, "i=%d : %f\n", i, ytopghost[i]);
    } else {
         fprintf(fp_check, "(Boundary - Should be 0.0)\n");
         for (i = 0; i < nx; ++i) fprintf(fp_check, "i=%d : %f\n", i, ytopghost[i]);
    }

    fclose(fp_check);
    printf("Rank %d: Finished writing %s\n", rank, fname_check);

    MPI_Barrier(MPI_COMM_WORLD); // Wait for all ranks to finish writing check files
    if (rank == 0) printf("--- Halo Exchange Check Complete ---\n\n");
}


/* ------------------------------------------------------------------
   7. Main
   ------------------------------------------------------------------ */
int main(int argc, char** argv)
{
    int nxglob, nyglob, rank, size, px, py, rank_x, rank_y;
    double *x = NULL, *y = NULL, **T = NULL, **rhs = NULL;
    double tst, ten, xstglob, xenglob, ystglob, yenglob;
    double dx = 0.0, dy = 0.0, dt = 0.0, tcurr = 0.0, kdiff = 0.0;
    int i, j, it, num_time_steps = 0, it_print = 1;
    int nx = 0, ny = 0; // Local dimensions
    int istglob, ienglob, jstglob, jenglob;
    double min_dx_dy;
    FILE* fid;
    char debugfname[100];

    // Timing variables
    double time_start = 0.0, time_end = 0.0, elapsed_time = 0.0, time_per_step = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*
       Read inputs from "input2d_par.in" on Rank 0
    */
    if (rank == 0) {
        fid = fopen("input2d_par.in", "r");
        if (fid == NULL) {
            fprintf(stderr, "Error opening input2d_par.in\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Use dummy variables for fscanf return check if needed
        int ret;
        ret = fscanf(fid, "%d %d\n", &nxglob, &nyglob);
        ret = fscanf(fid, "%lf %lf %lf %lf\n", &xstglob, &xenglob, &ystglob, &yenglob);
        ret = fscanf(fid, "%lf %lf\n", &tst, &ten);
        ret = fscanf(fid, "%lf\n", &kdiff);
        ret = fscanf(fid, "%d %d\n", &px, &py);
        fclose(fid);

        // --- Parameter Sanity Checks (Rank 0) ---
        if (px * py != size) {
            fprintf(stderr, "Error: px*py=%d does not match total MPI procs=%d. Stopping.\n",
                   px * py, size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (nxglob % px != 0 || nyglob % py != 0) {
             fprintf(stderr, "Error: nxglob (%d) must be divisible by px (%d), ", nxglob, px);
             fprintf(stderr, "and nyglob (%d) must be divisible by py (%d).\n", nyglob, py);
             MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (kdiff <= 0) {
             fprintf(stderr, "Error: kdiff (%.3f) must be positive.\n", kdiff);
             MPI_Abort(MPI_COMM_WORLD, 1);
        }
         if (ten <= tst) {
             fprintf(stderr, "Error: ten (%.3f) must be greater than tst (%.3f).\n", ten, tst);
             MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // --- End Sanity Checks ---


        // Calculate global dx, dy, dt (only rank 0 needs to do this initially)
        dx = (xenglob - xstglob) / (double)(nxglob - 1);
        dy = (yenglob - ystglob) / (double)(nyglob - 1);
        min_dx_dy = fmin(dx, dy);
        // Stability condition for explicit Euler: dt <= 0.5 / (kdiff * (1/dx^2 + 1/dy^2))
        // The factor 0.1 used is safer (more conservative) than 0.5 / (1/min_dx_dy^2 + 1/min_dx_dy^2) = 0.25 * min_dx_dy^2
        dt = 0.1 / kdiff * (min_dx_dy * min_dx_dy);
        if (dt <= 0) { // Prevent zero or negative dt
             fprintf(stderr, "Error: Calculated dt (%.8e) is non-positive. Check inputs (kdiff, dx, dy).\n", dt);
             MPI_Abort(MPI_COMM_WORLD, 1);
        }
        num_time_steps = (int)((ten - tst) / dt); // Number of steps to take
        if (num_time_steps <= 0) {
             fprintf(stderr, "Warning: num_time_steps (%d) is non-positive. ten might be too close to tst or dt too large. Setting num_time_steps=1.\n", num_time_steps);
             num_time_steps = 1;
        }
        it_print = num_time_steps / 10; // Print approx 10 intermediate results + final
        if (it_print <= 0) it_print = 1; // Ensure printing happens


        printf("\n--- Global Simulation Parameters ---\n");
        printf("  Grid: %d x %d points\n", nxglob, nyglob);
        printf("  Domain: x=[%.3f, %.3f], y=[%.3f, %.3f]\n",
               xstglob, xenglob, ystglob, yenglob);
        printf("  Time: tst=%.6f, ten=%.6f\n", tst, ten);
        printf("  Diffusion coeff kdiff=%.5f\n", kdiff);
        printf("  Processor Grid: %d x %d = %d processes\n", px, py, size);
        printf("  Calculated dx=%.6f, dy=%.6f\n", dx, dy);
        printf("  Calculated dt=%.8e\n", dt);
        printf("  Num time steps = %d\n", num_time_steps);
        printf("  Output every %d steps\n\n", it_print);
    }

    // --- Broadcast Parameters from Rank 0 to All Ranks ---
    // Pack integers into an array
    int sendarr_int[6];
    if (rank == 0) {
        sendarr_int[0] = nxglob;
        sendarr_int[1] = nyglob;
        sendarr_int[2] = num_time_steps;
        sendarr_int[3] = it_print;
        sendarr_int[4] = px;
        sendarr_int[5] = py;
    }
    MPI_Bcast(sendarr_int, 6, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) { // Unpack on other ranks
        nxglob      = sendarr_int[0];
        nyglob      = sendarr_int[1];
        num_time_steps = sendarr_int[2];
        it_print    = sendarr_int[3];
        px          = sendarr_int[4];
        py          = sendarr_int[5];
    }

    // Pack doubles into an array
    double sendarr_dbl[8]; // Add dx, dy, kdiff
    if (rank == 0) {
        sendarr_dbl[0] = tst;
        sendarr_dbl[1] = ten;
        sendarr_dbl[2] = dt;
        sendarr_dbl[3] = xstglob;
        sendarr_dbl[4] = xenglob;
        sendarr_dbl[5] = ystglob;
        sendarr_dbl[6] = yenglob;
        sendarr_dbl[7] = kdiff; // Broadcast kdiff too
        // dx and dy were calculated on rank 0, broadcast them too
        sendarr_dbl[8] = dx;
        sendarr_dbl[9] = dy;
    }
     MPI_Bcast(sendarr_dbl, 10, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Size is 10 now
     if (rank != 0) { // Unpack on other ranks
        tst      = sendarr_dbl[0];
        ten      = sendarr_dbl[1];
        dt       = sendarr_dbl[2];
        xstglob  = sendarr_dbl[3];
        xenglob  = sendarr_dbl[4];
        ystglob  = sendarr_dbl[5];
        yenglob  = sendarr_dbl[6];
        kdiff    = sendarr_dbl[7];
        dx       = sendarr_dbl[8]; // Receive dx
        dy       = sendarr_dbl[9]; // Receive dy
    }

    // --- Determine Local Subdomain Parameters ---
    get_processor_grid_ranks(rank, size, px, py, &rank_x, &rank_y);

    // Calculate local dimensions
    nx = nxglob / px; // Assumes perfect division
    ny = nyglob / py; // Assumes perfect division

    // Determine global indices for the start/end of this rank's local subdomain
    istglob = rank_x * nx;
    ienglob = istglob + nx - 1;
    jstglob = rank_y * ny;
    jenglob = jstglob + ny - 1;

    // Optional: Calculate physical subdomain boundaries for debugging
    double xst = xstglob + (double)istglob * dx;
    double xen = xstglob + (double)ienglob * dx;
    double yst = ystglob + (double)jstglob * dy;
    double yen = ystglob + (double)jenglob * dy;


    // --- Allocate Memory ---
    // Local grid coordinate arrays
    x = (double *)malloc(nx * sizeof(double));
    y = (double *)malloc(ny * sizeof(double));

    // Local Temperature (T) and RHS arrays (using 2D pointers)
    T = (double **)malloc(nx * sizeof(double *));
    rhs = (double **)malloc(nx * sizeof(double *));
    // Tnew is not needed for Forward Euler
    // double **Tnew = (double **)malloc(nx * sizeof(double *));
    if (T == NULL || rhs == NULL || x == NULL || y == NULL) {
         fprintf(stderr, "Rank %d: Error allocating primary arrays\n", rank);
         MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (i = 0; i < nx; i++) {
        T[i] = (double *)malloc(ny * sizeof(double));
        rhs[i] = (double *)malloc(ny * sizeof(double));
        // Tnew[i] = (double *)malloc(ny * sizeof(double));
        if(T[i] == NULL || rhs[i] == NULL){
             fprintf(stderr, "Rank %d: Error allocating row %d of T/rhs\n", rank, i);
             MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }


    // Ghost buffers (1D arrays) and Send/Receive buffers
    double *xleftghost  = (double *)malloc(ny * sizeof(double));
    double *xrightghost = (double *)malloc(ny * sizeof(double));
    double *ybotghost   = (double *)malloc(nx * sizeof(double));
    double *ytopghost   = (double *)malloc(nx * sizeof(double));
    double *sendbuf_x   = (double *)malloc(ny * sizeof(double));
    double *recvbuf_x   = (double *)malloc(ny * sizeof(double));
    double *sendbuf_y   = (double *)malloc(nx * sizeof(double));
    double *recvbuf_y   = (double *)malloc(nx * sizeof(double));
     if (!xleftghost || !xrightghost || !ybotghost || !ytopghost ||
         !sendbuf_x || !recvbuf_x || !sendbuf_y || !recvbuf_y) {
        fprintf(stderr, "Rank %d: Error allocating halo/comm buffers\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --- Initialize Grid and Data ---
    // Build local grid coordinate arrays (using global dx/dy)
    // Note: grid() calculates dx/dy again, but we already broadcasted them.
    // We can modify grid() or just pass the broadcasted values. Let's use the broadcasted ones.
    double temp_dx, temp_dy; // Dummy variables for grid function call
    grid(nx, nxglob, istglob, ienglob, xstglob, xenglob, x, &temp_dx);
    grid(ny, nyglob, jstglob, jenglob, ystglob, yenglob, y, &temp_dy);
    // Verify dx/dy consistency (optional debug)
    // if (fabs(temp_dx - dx) > 1e-12 || fabs(temp_dy - dy) > 1e-12) {
    //     fprintf(stderr, "Rank %d WARNING: dx/dy mismatch! %.8e vs %.8e, %.8e vs %.8e\n", rank, dx, temp_dx, dy, temp_dy);
    // }


    // Optional debug file per rank
    // sprintf(debugfname, "debug_%04d.dat", rank);
    // fid = fopen(debugfname, "w");
    // fprintf(fid, "Rank %d => Proc Coords: (%d, %d)\n", rank, rank_x, rank_y);
    // fprintf(fid, "Global Indices: i=[%d..%d], j=[%d..%d]\n", istglob, ienglob, jstglob, jenglob);
    // fprintf(fid, "Local Dimensions: nx=%d, ny=%d\n", nx, ny);
    // fprintf(fid, "Physical Subdomain: x=[%.4f..%.4f], y=[%.4f..%.4f]\n", xst, xen, yst, yen);
    // fprintf(fid, "dx=%.6f, dy=%.6f, dt=%.8e, kdiff=%.4f\n", dx, dy, dt, kdiff);
    // fclose(fid);

    // --- Halo Exchange Check ---
    check_halo_exchange(rank, rank_x, rank_y, px, py, nx, ny, T,
                        xleftghost, xrightghost, ybotghost, ytopghost,
                        sendbuf_x, recvbuf_x, sendbuf_y, recvbuf_y);

    // --- Set ACTUAL Initial Condition ---
    set_initial_condition(nx, ny, rank_x, rank_y, px, py, x, y, T, dx, dy);

    // Output initial condition (t=0, it=-1 or 0)
    tcurr = tst; // Set current time correctly for output
    // Use it=-1 conventionally for initial state before first step
    output_soln(rank, nx, ny, -1, tcurr, x, y, T);

    MPI_Barrier(MPI_COMM_WORLD); // Ensure IC is set and written before timing starts

    // --- Main Time-Stepping Loop ---
    if(rank == 0) printf("Starting time stepping...\n");
    time_start = MPI_Wtime(); // Start timer

    for (it = 0; it < num_time_steps; it++) {
        tcurr = tst + (double)(it + 1) * dt; // Time at the END of this step

        timestep_FwdEuler(rank, size, rank_x, rank_y, px, py,
                          nx, ny, dt, dx, dy,
                          xleftghost, xrightghost,
                          ybotghost, ytopghost,
                          kdiff, T, rhs, // Pass T and rhs
                          sendbuf_x, recvbuf_x,
                          sendbuf_y, recvbuf_y);

        // Output solution periodically (check step count, not time)
        // Output based on iteration count (it starts from 0)
        // Check if (it+1) is a multiple of it_print, or if it's the last step
         if ((it + 1) % it_print == 0 || it == num_time_steps - 1) {
            output_soln(rank, nx, ny, it + 1, tcurr, x, y, T);
            MPI_Barrier(MPI_COMM_WORLD); // Optional: sync after output
        }
    }

    time_end = MPI_Wtime(); // Stop timer
    elapsed_time = time_end - time_start;
    time_per_step = (num_time_steps > 0) ? elapsed_time / (double)num_time_steps : 0.0;

    MPI_Barrier(MPI_COMM_WORLD); // Ensure all ranks finish before Rank 0 prints timing

    // --- Final Output and Timing ---
    // Output solution at the very last time step if not already done
    // The loop condition `it == num_time_steps - 1` should handle this.

    if (rank == 0) {
        printf("\n--- Simulation Complete ---\n");
        printf("Total simulation wall time: %.6f seconds\n", elapsed_time);
        printf("Time per time step:       %.8f seconds\n", time_per_step);
        printf("Final time reached:       %.6f\n", tcurr);
        printf("Number of steps taken:    %d\n", num_time_steps);
    }


    // --- Free Memory ---
    free(x);
    free(y);
    for (i = 0; i < nx; i++) {
        free(T[i]);
        free(rhs[i]);
       // free(Tnew[i]); // Not allocated
    }
    free(T);
    free(rhs);
   // free(Tnew); // Not allocated

    free(xleftghost);
    free(xrightghost);
    free(ybotghost);
    free(ytopghost);
    free(sendbuf_x);
    free(recvbuf_x);
    free(sendbuf_y);
    free(recvbuf_y);


    MPI_Finalize();
    return 0;
}