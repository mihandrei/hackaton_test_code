#include "sim.h"
#include <stdlib.h>

//static double epi_state_var_range[] = {
//        -2.  , 1.,
//       -20.  , 2.,
//         2.0 , 5.0,
//        -2.  , 0.,
//         0.  , 2.,
//        -1.  , 1
//};

double initials[] = {
-0.533906,-8.445781,3.526376,-1.224648,1.023215,0.001695,

//-1.10188,-8.826803,3.412462,-1.261609,1.06071,0.047645

};
double * sweep_model(double min_x0, double max_x0){
        double *param_space = malloc(sizeof(double) * NNODES * NSWEEP);
        for(int pidx = 0; pidx < NSWEEP; pidx++){
                for (int nidx = 0; nidx < NNODES; nidx++){
                        param_space[nidx + pidx * NNODES] = pidx * (max_x0 - min_x0) / (NSWEEP -1)+ min_x0;;
                }
        }
        return param_space;
}

void prepare_initial_state(double *state){
        for(int p_idx=0;p_idx<NSWEEP;p_idx++){
                for (int n_idx = 0; n_idx < NNODES; ++n_idx) {
                        for (int sv = 0; sv < NSV; sv++) {
                               // double lo = epi_state_var_range[sv * 2];
                               // double hi = epi_state_var_range[sv * 2 + 1];
                                //state[sv + n_idx*NSV + p_idx * NSV * NNODES] = (lo + hi) / 2;
                                state[sv + n_idx*NSV + p_idx * NSV * NNODES] = initials[sv];
                        }
                }
        }
}
