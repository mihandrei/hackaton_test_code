#ifndef SIM_
#define SIM_
//(4000/0.05)
#define NGPU_TIMESTEPS 80000
#define NSWEEP 7

#define NSV 6
#define NNODES 2
#define DT 0.05


void model_dfun(const double *state, const double param, double *dstate) ;
void euler_step( double param, const double *state, double *next);
void heun_step(double param, const double *state, double *next);

double * sweep_model(double min_x0, double max_x0);
void prepare_initial_state(double *state);

#endif

