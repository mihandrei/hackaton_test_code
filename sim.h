#ifndef SIM_
#define SIM_
//(4000/0.05)
#define NGPU_TIMESTEPS 20000
#define NSWEEP 512

#define NSV 6
#define NNODES 4
#define DT 0.1

#pragma acc routine seq
void model_dfun(const double *state, const double param, double *dstate) ;
#pragma acc routine seq
void euler_step( double param, const double *state, double *next);
#pragma acc routine seq
void heun_step(double param, const double *state, double *next);

double * sweep_model(double min_x0, double max_x0);
void prepare_initial_state(double *state);

#endif

