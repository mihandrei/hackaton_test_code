#include <stdlib.h>
#include <math.h>
#include "sim.h"


int main(int a, char**argv){
        struct ModelEpi *sweep = new_model(NSWEEP);

        double *state = malloc(sizeof(double) * NSV * NSWEEP);
        double *dstate = malloc(sizeof(double) * NSV * NSWEEP);

        prepare_initial_state(state);

        sweep_model(sweep, -3.0, 1.0);

        // for (int t = 0; t < time_steps; t++)
        for(int pse1_idx=0; i<sweep->len; i++){
                model_dfun(i, sweep, state, dstate);
        }

        free(dstate);
        free(state);
        free_model(sweep);

        return 0;
}
