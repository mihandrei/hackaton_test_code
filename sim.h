#ifndef SIM_
#define SIM_
//(4000/0.05)
#define NGPU_TIMESTEPS 800000
#define NSWEEP 64

#define NSV 6
#define NNODES 2
#define DT 0.02

extern double *conn_74_weights;

#pragma acc routine seq
void model_dfun(const double *state, const double param, double *dstate) ;
#pragma acc routine seq
void euler_step( double param, const double *state, double *next);
#pragma acc routine seq
void heun_step(double param, const double *state, double *next);

double * sweep_model(double min_x0, double max_x0);
void prepare_initial_state(double *state);

#endif

