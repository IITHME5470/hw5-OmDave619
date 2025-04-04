#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/* ------------------------------------------------------------------
   1. Domain-Decomposition Helper
   ------------------------------------------------------------------ */
void get_processor_grid_ranks(int rank, int size, int px, int py,
                              int *rank_x, int *rank_y)
{
\
}

/* ------------------------------------------------------------------
   2. Grid Generation
   ------------------------------------------------------------------ */
void grid(int nx_local, int nx_global,
          int istglob, int ienglob,
          double xstglob, double xenglob,
          double *x, double *dx)
{


}

/* ------------------------------------------------------------------
   3. Boundary Conditions, Initial Condition
   ------------------------------------------------------------------ */
void enforce_bcs(int nx, int ny, double *x, double *y, double **T)
{

}

void set_initial_condition(int nx, int ny, double *x, double *y,
                           double **T, double dx, double dy)
{

}

/* ------------------------------------------------------------------
   4. Halo Exchange Routines
   ------------------------------------------------------------------ */
void halo_exchange_2d_x(int rank, int rank_x, int rank_y,
                        int size, int px, int py,
                        int nx, int ny,
                        int nxglob, int nyglob,
                        double *x, double *y, double **T,
                        double *xleftghost, double *xrightghost,
                        double *sendbuf_x, double *recvbuf_x)
{

}

void halo_exchange_2d_y(int rank, int rank_x, int rank_y,
                        int size, int px, int py,
                        int nx, int ny,
                        int nxglob, int nyglob,
                        double *x, double *y, double **T,
                        double *ybotghost, double *ytopghost,
                        double *sendbuf_y, double *recvbuf_y)
{

}

/* ------------------------------------------------------------------
   5. RHS Computation and Time-Stepping
   ------------------------------------------------------------------ */
void get_rhs(int nx, int nxglob, int ny, int nyglob,
             int istglob, int ienglob, int jstglob, int jenglob,
             double dx, double dy,
             double *xleftghost, double *xrightghost,
             double *ybotghost, double *ytopghost,
             double kdiff, double *x, double *y,
             double **T, double **rhs)
{

}

void timestep_FwdEuler(int rank, int size,
                       int rank_x, int rank_y,
                       int px, int py,
                       int nx, int nxglob,
                       int ny, int nyglob,
                       int istglob, int ienglob,
                       int jstglob, int jenglob,
                       double dt, double dx, double dy,
                       double *xleftghost, double *xrightghost,
                       double *ybotghost, double *ytopghost,
                       double kdiff,
                       double *x, double *y,
                       double **T, double **rhs,
                       double *sendbuf_x, double *recvbuf_x,
                       double *sendbuf_y, double *recvbuf_y)
{
  // 1. Exchange halo data
  halo_exchange_2d_x(rank, rank_x, rank_y, size, px, py,
                     nx, ny, nxglob, nyglob,
                     x, y, T,
                     xleftghost, xrightghost,
                     sendbuf_x, recvbuf_x);

  halo_exchange_2d_y(rank, rank_x, rank_y, size, px, py,
                     nx, ny, nxglob, nyglob,
                     x, y, T,
                     ybotghost, ytopghost,
                     sendbuf_y, recvbuf_y);

  // 2. Compute RHS
  get_rhs(nx, nxglob, ny, nyglob,
          istglob, ienglob, jstglob, jenglob,
          dx, dy,
          xleftghost, xrightghost,
          ybotghost, ytopghost,
          kdiff, x, y, T, rhs);

  // 3. Update T using Forward Euler
  for(int i = 0; i < nx; i++){
    for(int j = 0; j < ny; j++){
      T[i][j] += dt * rhs[i][j];
    }
  }

  // 4. Enforce BCs
  enforce_bcs(nx, ny, x, y, T);
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
  sprintf(fname, "T_x_y_%06d_%04d_par.dat", it, rank);
  fp = fopen(fname, "w");
  if(!fp){
    printf("Error opening %s for writing!\n", fname);
    return;
  }
  for(i = 0; i < nx; i++){
    for(j = 0; j < ny; j++){
      fprintf(fp, "%lf %lf %lf\n", x[i], y[j], T[i][j]);
    }
  }
  fclose(fp);
  printf("Done writing solution for rank = %d, time step = %d, time = %e\n",
         rank, it, tcurr);
}

/* ------------------------------------------------------------------
   7. Main
   ------------------------------------------------------------------ */
int main(int argc, char** argv)
{
  int nxglob, nyglob, rank, size, px, py, rank_x, rank_y;
  double *x, *y, **T, **rhs, tst, ten, xstglob, xenglob, ystglob, yenglob;
  double dx, dy, dt, tcurr, kdiff;
  int i, it, num_time_steps, it_print;
  int istglob, ienglob, jstglob, jenglob;
  double min_dx_dy;
  FILE* fid;  
  char debugfname[100];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* 
     Read inputs from "input2d_par.in" with 5 lines:
       1. nxglob nyglob
       2. xstglob xenglob ystglob yenglob
       3. tst ten
       4. kdiff
       5. px py
  */
  if(rank == 0){
    fid = fopen("input2d_par.in", "r");
    if(fid == NULL) {
      printf("Error opening input2d_par.in\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fscanf(fid, "%d %d\n", &nxglob, &nyglob);
    fscanf(fid, "%lf %lf %lf %lf\n", &xstglob, &xenglob, &ystglob, &yenglob);
    fscanf(fid, "%lf %lf\n", &tst, &ten);
    fscanf(fid, "%lf\n", &kdiff);
    fscanf(fid, "%d %d\n", &px, &py);
    fclose(fid);

    // compute dx, dy, dt same as the serial code
    dx = (xenglob - xstglob) / (double)(nxglob - 1);
    dy = (yenglob - ystglob) / (double)(nyglob - 1);
    min_dx_dy = fmin(dx, dy);
    dt = 0.1 / kdiff * (min_dx_dy * min_dx_dy);
    num_time_steps = (int)((ten - tst)/dt) + 1;
    it_print = num_time_steps / 5;

    printf("\nGlobal Inputs (from input2d_par.in):\n");
    printf("  nxglob=%d, nyglob=%d\n", nxglob, nyglob);
    printf("  Domain: x=[%.3f, %.3f], y=[%.3f, %.3f]\n",
           xstglob, xenglob, ystglob, yenglob);
    printf("  Time: tst=%.6f, ten=%.6f, dt=%.8f\n", tst, ten, dt);
    printf("  kdiff=%.5f\n", kdiff);
    printf("  Processor grid: px=%d, py=%d\n", px, py);
    printf("  num_time_steps=%d, it_print=%d\n\n", num_time_steps, it_print);

    if(px * py != size){
      printf("Error: px*py=%d does not match total MPI procs=%d. Stopping.\n",
             px*py, size);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  // Broadcast integer parameters
  int sendarr_int[6];
  if(rank == 0){
    sendarr_int[0] = nxglob;
    sendarr_int[1] = nyglob;
    sendarr_int[2] = num_time_steps;
    sendarr_int[3] = it_print;
    sendarr_int[4] = px;
    sendarr_int[5] = py;
  }
  MPI_Bcast(sendarr_int, 6, MPI_INT, 0, MPI_COMM_WORLD);
  if(rank != 0){
    nxglob      = sendarr_int[0];
    nyglob      = sendarr_int[1];
    num_time_steps = sendarr_int[2];
    it_print    = sendarr_int[3];
    px          = sendarr_int[4];
    py          = sendarr_int[5];
  }

  // Each rank determines local nx, ny
  int nx = nxglob / px;
  int ny = nyglob / py;

  // Rank coordinates
  get_processor_grid_ranks(rank, size, px, py, &rank_x, &rank_y);

  // Broadcast double parameters
  double xlen, ylen;
  double sendarr_dbl[7];
  if(rank == 0){
    sendarr_dbl[0] = tst;
    sendarr_dbl[1] = ten;
    sendarr_dbl[2] = dt;
    sendarr_dbl[3] = xstglob;
    sendarr_dbl[4] = xenglob;
    sendarr_dbl[5] = ystglob;
    sendarr_dbl[6] = yenglob;
    xlen = (xenglob - xstglob)/(double)px;
    ylen = (yenglob - ystglob)/(double)py;
  }
  MPI_Bcast(sendarr_dbl, 7, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if(rank != 0){
    tst      = sendarr_dbl[0];
    ten      = sendarr_dbl[1];
    dt       = sendarr_dbl[2];
    xstglob  = sendarr_dbl[3];
    xenglob  = sendarr_dbl[4];
    ystglob  = sendarr_dbl[5];
    yenglob  = sendarr_dbl[6];
    xlen = (xenglob - xstglob)/(double)px;
    ylen = (yenglob - ystglob)/(double)py;
  }

  // Determine global indices for local subdomain
  istglob = rank_x * (nxglob / px);
  ienglob = (rank_x + 1) * (nxglob / px) - 1;
  jstglob = rank_y * (nyglob / py);
  jenglob = (rank_y + 1) * (nyglob / py) - 1;

  // Physical subdomain boundaries (optional for debugging)
  double xst = xstglob + rank_x * xlen;
  double xen = xst + xlen;
  double yst = ystglob + rank_y * ylen;
  double yen = yst + ylen;

  // Allocate arrays
  x = (double *)malloc(nx * sizeof(double));
  y = (double *)malloc(ny * sizeof(double));
  T = (double **)malloc(nx * sizeof(double *));
  rhs = (double **)malloc(nx * sizeof(double *));
  double **Tnew = (double **)malloc(nx * sizeof(double *));
  for(i = 0; i < nx; i++){
    T[i]    = (double *)malloc(ny * sizeof(double));
    rhs[i]  = (double *)malloc(ny * sizeof(double));
    Tnew[i] = (double *)malloc(ny * sizeof(double));
  }

  // Ghost buffers and send/recv buffers
  double *xleftghost  = (double *)malloc(ny * sizeof(double));
  double *xrightghost = (double *)malloc(ny * sizeof(double));
  double *ybotghost   = (double *)malloc(nx * sizeof(double));
  double *ytopghost   = (double *)malloc(nx * sizeof(double));
  double *sendbuf_x   = (double *)malloc(ny * sizeof(double));
  double *recvbuf_x   = (double *)malloc(ny * sizeof(double));
  double *sendbuf_y   = (double *)malloc(nx * sizeof(double));
  double *recvbuf_y   = (double *)malloc(nx * sizeof(double));

  // Build local grids by index-slicing
  grid(nx, nxglob, istglob, ienglob, xstglob, xenglob, x, &dx);
  grid(ny, nyglob, jstglob, jenglob, ystglob, yenglob, y, &dy);

  // Optional debug file
  sprintf(debugfname, "debug_%04d.dat", rank);
  fid = fopen(debugfname, "w");
  fprintf(fid, "Rank %d => rank_x=%d, rank_y=%d\n", rank, rank_x, rank_y);
  fprintf(fid, "Local domain i=[%d..%d], j=[%d..%d]\n", istglob, ienglob, jstglob, jenglob);
  fprintf(fid, "Local nx=%d, ny=%d\n", nx, ny);
  fprintf(fid, "Local x-range=%.4f..%.4f, y-range=%.4f..%.4f\n", xst, xen, yst, yen);
  fclose(fid);

  // Set initial condition
  set_initial_condition(nx, ny, x, y, T, dx, dy);
  output_soln(rank, nx, ny, 0, tst, x, y, T);

  // Main time-stepping loop
  for(it = 0; it < num_time_steps; it++){
    tcurr = tst + (double)(it + 1) * dt;

    timestep_FwdEuler(rank, size, rank_x, rank_y,
                      px, py, nx, nxglob, ny, nyglob,
                      istglob, ienglob, jstglob, jenglob,
                      dt, dx, dy,
                      xleftghost, xrightghost,
                      ybotghost, ytopghost,
                      kdiff, x, y, T, rhs,
                      sendbuf_x, recvbuf_x,
                      sendbuf_y, recvbuf_y);

    if(it % it_print == 0){
      output_soln(rank, nx, ny, it, tcurr, x, y, T);
    }
  }

  MPI_Finalize();
  return 0;
}
