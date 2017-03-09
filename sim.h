#ifndef SIM_
#define SIM_
//(4000/0.05)
#define NGPU_TIMESTEPS 800000
#define NSWEEP 128

#define NSV 6
#define NNODES 2
#define DT 0.02

#define N_CV 2 // no of coupled variables
// coupling var indexes
extern int epi_coupling_var_ids[2];

extern double conn_74_weights[74 * 74]; // connectivity matrix
extern double conn2_zeros[2*2];
double conn2_antidiag[2*2];

#pragma acc routine seq
void model_dfun(const double *state, const double *incoming_activity, const double param, double *dstate) ;
#pragma acc routine seq
void euler_step( double param, const double *state, double *next);
#pragma acc routine seq
void heun_step(double param, const double *incoming_activity, const double *state, double *next);

double * sweep_model(double min_x0, double max_x0);
void prepare_initial_state(double *state);

#endif

