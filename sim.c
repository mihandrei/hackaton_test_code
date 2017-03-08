#include <stdlib.h>
#include <stdio.h>
#include "sim.h"
#include <omp.h>

#define TEMPORAL_SUBSAMPLE 32

void print_state(double *state){
        FILE *file = fopen("out.array", "w");

        fprintf(file, "%d %d %d %d\n", NGPU_TIMESTEPS/TEMPORAL_SUBSAMPLE, NSWEEP, NNODES, NSV);

        for (int t=0; t < NGPU_TIMESTEPS; t+=TEMPORAL_SUBSAMPLE) {
                for (int param_idx = 0; param_idx < NSWEEP; param_idx++) {
                        for (int n_idx = 0; n_idx < NNODES; n_idx++) {
                                for (int sv_idx = 0; sv_idx < NSV; sv_idx++) {
                                        long offset = t * NSWEEP * NNODES * NSV
                                                      + param_idx * NNODES * NSV
                                                      + n_idx * NSV;
                                        fprintf(file, "%f ", state[sv_idx + offset]);
                                }
                                fprintf(file, "\n");
                        }
                }
                fprintf(file, "\n\n");
        }
        fclose(file);
}


void print_params(double *params){
        FILE *file = fopen("out.array", "w");
        for (int i=0; i<NNODES*NSWEEP;i++)
                fprintf(file, "%f ", params[i]);
        fclose(file);
}


void kernel_step(double *ret, double *param_space, double *state, double *next){
//        #pragma acc kernels copy(state[0: NSV * NNODES * NSWEEP * NGPU_TIMESTEPS]) copyin(param_space[0: NNODES * NSWEEP])
        #pragma acc data copy(state[0: NSV * NNODES * NSWEEP * NGPU_TIMESTEPS]) copyin(param_space[0: NNODES * NSWEEP])
        for (int t = 0; t < NGPU_TIMESTEPS - 1; t++) {
                //#pragma omp parallel for
                #pragma acc parallel loop
                for (int param_idx = 0; param_idx < NSWEEP; param_idx++) {
                        #pragma acc loop
                        for (int n_idx = 0; n_idx < NNODES; n_idx++) {
                                long offset = param_idx * NNODES * NSV
                                              + n_idx * NSV;
                                heun_step(param_space[n_idx + param_idx * NNODES],
                                                state + offset,
                                                next + offset
                                        );
                        }
                }
                double  *tmp = state;
                state = next;
                next = tmp;
                for (int i = 0; i < NSWEEP * NNODES * NSV; ++i) {
                        ret[t*NSWEEP*NNODES*NSV + i] = next[i];
                }
        }
}

int main(int a, char**argv){
        double *timeseries = malloc(sizeof(double) * NSV * NNODES * NSWEEP * NGPU_TIMESTEPS);

        double *state = malloc(sizeof(double) * NSWEEP * NNODES * NSV);
        double *next = malloc(sizeof(double) * NSWEEP * NNODES * NSV);

        prepare_initial_state(state);

        double *param_space = sweep_model(-3.8, -1.0);

        double start_time = omp_get_wtime();
        kernel_step(timeseries, param_space, state, next);
        double time = omp_get_wtime() - start_time;
        printf("computation time %f sec\n", time);

        print_state(timeseries);
        //print_params(param_space);
        free(state);
        free(next);
        free(timeseries);
        free(param_space);

        return 0;
}

