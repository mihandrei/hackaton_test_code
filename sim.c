#include <stdlib.h>
#include <stdio.h>
#include "sim.h"
#include <omp.h>

void print_state(double *state){
        FILE *file = fopen("out.array", "w");
        for (int t=0; t < NGPU_TIMESTEPS; t+=50){
            for (int p_idx=0; p_idx<NSWEEP; p_idx++){
                int n_idx = 0;
                for (int sv_idx=0; sv_idx<NSV; sv_idx++){
                    fprintf(file, "%f ", state[sv_idx + n_idx*NSV + p_idx * NSV * NNODES + t * NSV*NNODES*NSWEEP]);
                }
            }
        }
        fclose(file);
}


void print_params(double *params){
        FILE *file = fopen("out.array", "w");
        for (int i=0; i<NNODES*NSWEEP;i++)
                fprintf(file, "%f ", params[i]);
        fclose(file);
}

void reset_incoming_activity(double *incoming_activity){
        for (int i=0; i<N_CV;i++) {
            incoming_activity[i]=0;
        }
}

void kernel_step(double *state, double *param_space){
        #pragma acc data copy(state[0: NSV * NNODES * NSWEEP * NGPU_TIMESTEPS]) copyin(param_space[0: NNODES * NSWEEP])
        for (int t = 0; t < NGPU_TIMESTEPS - 1; t++) {
                #pragma acc parallel loop
                for (int param_idx = 0; param_idx < NSWEEP; param_idx++) {
                        #pragma acc loop
                        for (int n_idx = 0; n_idx < NNODES; n_idx++) {
                                // Calc incoming activity from coupled
                                reset_incoming_activity(incoming_activity);
                                double *conn_node_weights = conn_74_weights + n_idx * NNODES;
                                for(int cn_idx = 0; cn_idx < NNODES; cn_idx++) {
                                    long coupling_node_offset = t * NSWEEP * NNODES * NSV
                                                                + param_idx * NNODES * NSV
                                                                + cn_idx * NSV;
                                    double *coupled_node_state = state + coupling_node_offset;
                                    for(int cv_idx = 0; cv_idx < N_CV; cv_idx++) {
                                        if(conn_node_weights[cn_idx] != 0) {
                                            // coupled function
                                            incoming_activity[cv_idx] += conn_node_weights[cn_idx] * coupled_node_state[epi_coupling_var_ids[cv_idx]];
                                        }
                                    }
                                }

                                long offset = t * NSWEEP * NNODES * NSV
                                                + param_idx * NNODES * NSV
                                                + n_idx * NSV;
                                long next_state_offset = (t + 1) * NSWEEP * NNODES * NSV
                                                + param_idx * NNODES * NSV
                                                + n_idx * NSV;
                                heun_step(param_space[n_idx + param_idx * NNODES],
                                                incoming_activity,
                                                state + offset,
                                                state + next_state_offset
                                        );
                        }
                }
        }
}

int main(int a, char**argv){
        double *state = malloc(sizeof(double) * NSV * NNODES * NSWEEP * NGPU_TIMESTEPS);

        prepare_initial_state(state);

        double *param_space = sweep_model(-3.0, 1.0);

        double start_time = omp_get_wtime();
        kernel_step(state, param_space);
        double time = omp_get_wtime() - start_time;
        printf("computation time %f sec\n", time);

        print_state(state);
        //print_params(param_space);
        free(state);
        free(param_space);

        return 0;
}

