#include <math.h>
#include "sim.h"

//params[node_idx*NVAR_PARAM + par_idx];
//state[sv + par1_idx * NSV + par2_idx * NSV * NPAR1_SWEEP + n_idx * NSV * NPAR1_SWEEP * NPAR2_SWEEP + time * NSV * NPAR1_SWEEP *NPAR2_SWEEP *NNODES];
void model_dfun(int n_idx, int param_idx, const double *state, const double *param_space, double *dstate) {

        state = state + param_idx*NNODES*NSV + n_idx*NSV;
        dstate = dstate + param_idx*NNODES*NSV + n_idx*NSV;

        double y0 = state[0];
        double y1 = state[1];
        double y2 = state[2];
        double y3 = state[3];
        double y4 = state[4];
        double y5 = state[5];
        double tmp;

        double a = 1.0;
        double b = 3.0;
        double c = 1.0;
        double d = 5.0;
        double aa = 6.0;
        double r = 0.00035;
        double Kvf = 0.0;
        double Kf = 0.0;
        double Ks = 0.0;
        double tau = 10.0;
        double Iext = 3.1;
        double Iext2 = 0.45;
        double slope = 0.0;
        double x0 = param_space[n_idx + param_idx*NNODES]; //i * (max_x0 - min_x0) / (double len -1)+ min_x0;
        double tt = 1.0;

//        double cpop_1 = coupling[0];
//        double cpop_2 = coupling[1];
        double cpop_1 = 0;
        double cpop_2 = 0;

        // population 1
        if (y0 < 0.0)
                tmp = -a * pow(y0, 2) + b * y0;
        else
                tmp = slope- y3 + 0.6 * pow(y2 - 4.0, 2);


        dstate[0] = y1 - y2 + Iext + Kvf * cpop_1 + tmp * y0;
        dstate[1] = c - d * pow(y0, 2) - y1;

        // energy
        if (y2 < 0.0)
                tmp = -0.1 * pow(y2, 7);
        else
                tmp = 0;

        dstate[2] = r * (4 * (y0 - x0) - y2 + tmp + Ks * cpop_1);

        // population 2
        dstate[3] = -y4 + y3 - pow(y3, 3) + Iext2 + 2 * y5 - 0.3 * (y2 - 3.5) + Kf * cpop_2;

        if (y3 < -0.25)
                tmp = 0;
        else
                tmp = aa * (y3 + 0.25);

        dstate[4] = (-y4 + tmp) / tau;

        // filter
        dstate[5] = -0.01 * (y5 - 0.1 * y0);

        // apply time scaling parameter
        for (int sv = 0; sv < 6; ++sv)
                dstate[sv] *= tt;
}
/*
void heun_step(int i, struct ModelEpi *mp, const double *state,
        double *next){

    int i;
    double *dleft, *dright; //todo

    model_dfun(i, mp, state, dleft);

    for (i = 0; i < NSV; ++i) {
        nexteuler[i] = state[i] + DT * dleft[i];// + gr_noise[i];
    }

    model_dfun(i, mp, state, dright);

    for (i = 0; i < NSV; ++i) {
        next[i] = state[i] + 0.5 * DT * (dleft[i] + dright[i]);// + gr_noise[i];
    }
}

*/