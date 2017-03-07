#ifndef SIM_
#define SIM_

#define NSWEEP 12
#define NSV 6
#define NNODES 2
#define DT 0.0001

struct ModelEpi {
        double *a;       // Coefficient of the cubic term in the first SV
        double *b;       // Coefficient of the squared term in the first SV
        double *c;       // Additive coefficient for the second SV
        double *d;       // Coefficient of the squared term in the second SV
        double *aa;      // Linear coefficient in fifth SV
        double *r;       // Temporal scaling in the third SV
        // double s; // Linear coefficient in the third SV todo: this parameter taken from TVB is not used in dfun
        double *Kvf;     // Coupling scaling on a very fast time scale.
        double *Kf;      // Correspond to the coupling scaling on a fast time scale.
        double *Ks;      // Permittivity coupling, that is from the fast time scale toward the slow time scale
        double *tau;     // Temporal scaling coefficient in fifth SV
        double *Iext;    // External input current to the first population
        double *Iext2;   // External input current to the second population
        double *slope;   // Linear coefficient in the first SV
        double *x0;      // Epileptogenicity parameter
        double *tt;      // Time scaling of the whole system

        int len;
};

struct ModelEpi * new_model(int len); 
void free_model(struct ModelEpi *self);

void model_dfun(int n_idx, int param_idx, const double *state, const double *param_space, double *dstate);


void sweep_model(struct ModelEpi *self, double min_x0, double max_x0);
void prepare_initial_state(double *state);

#endif

