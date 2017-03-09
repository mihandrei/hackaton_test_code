#include <stdlib.h>
#include <stdio.h>
#include "sim.h"
#include <omp.h>

#define TEMPORAL_SUBSAMPLE 32

void print_state(double *state){
        FILE *file = fopen("out.array", "w");
        // NOBSERVED vars is 1 . lpf is computed in kernel
        fprintf(file, "%d %d %d %d\n", NGPU_TIMESTEPS/TEMPORAL_SUBSAMPLE, NSWEEP, NNODES, 1);

        for (int t=0; t < NGPU_TIMESTEPS/TEMPORAL_SUBSAMPLE; t++) {
                for (int param_idx = 0; param_idx < NSWEEP; param_idx++) {
                        for (int n_idx = 0; n_idx < NNODES; n_idx++) {
                                for (int sv_idx = 0; sv_idx < 1; sv_idx++) {
                                        long offset = t * NSWEEP * NNODES * 1
                                                      + param_idx * NNODES * 1
                                                      + n_idx * 1;
                                        fprintf(file, "%f ", state[sv_idx + offset]);
                                }
                                fprintf(file, "\n");
                        }
                }
                fprintf(file, "\n\n");
        }
        fclose(file);
}

void print_variance(double *M2){
        FILE *file = fopen("variance.array", "w");
        fprintf(file, "%d %d %d \n", NSWEEP, NNODES, NSV);
        for (int param_idx = 0; param_idx < NSWEEP; param_idx++) {
                for (int i = 0; i < NSV * NNODES; i++) {
                        fprintf(file, "%f ", M2[param_idx * NNODES * NSV + i]);
                }
                fprintf(file, "\n");
        }
        fclose(file);
}

void print_params(double *params){
        FILE *file = fopen("out.array", "w");
        for (int i=0; i<NNODES*NSWEEP;i++)
                fprintf(file, "%f ", params[i]);
        fclose(file);
}

#pragma acc routine seq
void update_online_variance(int t, double *M2, double *mean, const double *state){
        for (int param_idx = 0; param_idx < NSWEEP; param_idx++) {
                for (int n_idx = 0; n_idx < NNODES; n_idx++) {
                        long offset = param_idx * NNODES * NSV
                                      + n_idx * NSV;

                        for (int sv = 0; sv < NSV; sv++) {
                                double delta, delta2;
                                double x = state[offset + sv];

                                delta = x - mean[offset + sv];
                                mean[offset + sv] += delta / (t + 1);
                                delta2 = x - mean[offset + sv];
                                M2[offset + sv] += delta * delta2;
                        }
                }
        }
}
#pragma acc routine seq
void data_reduce_kernel(double *ret, const double *state, int t) {
        if (t % TEMPORAL_SUBSAMPLE == 0){ // % here can be avoided by another loop under t in parent temporal iteration
                t = t / TEMPORAL_SUBSAMPLE;
                #pragma omp parallel for
                #pragma acc loop
                for (int param_idx = 0; param_idx < NSWEEP; param_idx++) {
                        for (int n_idx = 0; n_idx < NNODES; n_idx++) {
                                long offset = param_idx * NNODES * NSV
                                               + n_idx * NSV;
                                // NOBSERVED=1
                                // collapse all svars. compute lfp
                                long ret_idx = t*NSWEEP*NNODES*1 + param_idx * NNODES*1 + n_idx*1;
                                ret[ret_idx] = state[offset+0] - state[offset+3];
                                //to copy all svars :
//                                for (int sv_idx = 0; sv_idx < NSV; sv_idx++) {
//                                        ret[ret_idx + sv_idx] = state[offset + sv_idx];
//                                }
                        }
                }
        }
}

#pragma acc routine seq
void integration_kernel(const double *param_space, const double *state, double *next) {
        #pragma omp parallel for
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
}

void kernels_step(double *ret, double *param_space, double *state, double *next,
                  double *M2, double *mean){

        #pragma acc data copy(state[0: NSV * NNODES * NSWEEP])\
                              create(next[0: NSV * NNODES * NSWEEP])\
                              param_space[0: NNODES * NSWEEP]\
                              copyout(M2[0: NSV * NNODES * NSWEEP])\
                              create(mean[0: NSV * NNODES * NSWEEP]\
                              copyout(ret[0:1 * NNODES * NSWEEP * NGPU_TIMESTEPS/TEMPORAL_SUBSAMPLE]))
        {
                for (int t = 0; t < NGPU_TIMESTEPS - 1; t++) {
                        integration_kernel(param_space, state, next);
                        // swap current and next buffer
                        double *tmp = state;
                        state = next;
                        next = tmp;
                        // data reduction and copy to output buffer
                        data_reduce_kernel(ret, state, t);
                        update_online_variance(t, M2, mean, state);
                }

                for (int i = 0; i < NSWEEP * NNODES * NSV; ++i) {
                        M2[i] /= (NGPU_TIMESTEPS - 1);
                }
        }
}

int main(int a, char**argv){
        double *timeseries = malloc(sizeof(double) * 1 * NNODES * NSWEEP * NGPU_TIMESTEPS/TEMPORAL_SUBSAMPLE);

        double *state = malloc(sizeof(double) * NSWEEP * NNODES * NSV);
        double *next = malloc(sizeof(double) * NSWEEP * NNODES * NSV);

        double *M2 = calloc( NSWEEP * NNODES * NSV, sizeof(double));
        double *mean = calloc( NSWEEP * NNODES * NSV, sizeof(double));

        prepare_initial_state(state);

        double *param_space = sweep_model(-3.8, -1.0);

        double start_time = omp_get_wtime();
        kernels_step(timeseries, param_space, state, next, M2, mean);
        double time = omp_get_wtime() - start_time;
        printf("computation time %f sec\n", time);

        print_state(timeseries);
        print_variance(M2);
        //print_params(param_space);
        free(state);
        free(next);
        free(timeseries);
        free(param_space);

        return 0;
}

