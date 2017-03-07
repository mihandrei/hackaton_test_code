#include "sim.h"

static double epi_state_var_range[] = {
        -2.  , 1.,
       -20.  , 2.,
         2.0 , 5.0,
        -2.  , 0.,
         0.  , 2.,
        -1.  , 1
};

void sweep_model(struct ModelEpi *self, double min_x0, double max_x0){
        for (int i=0; i<self->len;i++){
                self->a[i] = 1.0;
                self->b[i] = 3.0;
                self->c[i] = 1.0;
                self->d[i] = 5.0;
                self->aa[i] = 6.0;
                self->r[i] = 0.00035;
                self->Kvf[i] = 0.0;
                self->Kf[i] = 0.0;
                self->Ks[i] = 0.0;
                self->tau[i] = 10.0;
                self->Iext[i] = 3.1;
                self->Iext2[i] = 0.45;
                self->slope[i] = 0.0;
                self->x0[i] = i * (max_x0 - min_x0) / (self->len -1)+ min_x0;
                self->tt[i] = 1.0;
        }
}

void prepare_initial_state(double *state){
        for(int i=0;i<NSWEEP*NSV;i+=NSV){
                for (int sv=0; sv<NSV; sv++){
                    double lo = epi_state_var_range[sv*2];
                    double hi = epi_state_var_range[sv*2 + 1];
                    state[i+sv] = (lo + hi)/2;
                }
        }
}

