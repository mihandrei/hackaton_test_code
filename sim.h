#ifndef SIM_
#define SIM_
//(4000/0.05)
#define NGPU_TIMESTEPS 30000 // no of GPU timesteps
#define NSWEEP 128 // no of param sweep

#define NSV 6 // no of state variables
#define NNODES 74 // no of nodes
#define DT 0.001 // time step
#define N_CV 2 // no of coupled variables

// coupling var indexes
static int epi_coupling_var_ids[] = {0, 3};
double incoming_activity[N_CV];

extern double conn_74_weights[NNODES * NNODES]; // connectivity matrix

#pragma acc routine seq
void model_dfun(const double *state, const double *incoming_activity, const double param, double *dstate) ;
#pragma acc routine seq
void euler_step( double param, const double *state, double *next);
#pragma acc routine seq
void heun_step(double param, const double *incoming_activity, const double *state, double *next);

double * sweep_model(double min_x0, double max_x0);
void prepare_initial_state(double *state);

#endif

