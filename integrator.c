#include <math.h>
#include "sim.h"


void model_dfun(const double *state, const double *incoming_activity, const double param, double *dstate) {
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
        double Ks = 1.5;//0.0;
        double tau = 10.0;
        double Iext = 3.1;
        double Iext2 = 0.45;
        double slope = 0.0;
        double x0 = -1.5; //param
        double tt = 1.0;


        double cpop_1 = incoming_activity[0];
        double cpop_2 = incoming_activity[1];

        // population 1
        if (y0 < 0.0)
                tmp = -a * pow(y0, 2) + b * y0;
        else
                tmp = slope - y3 + 0.6 * pow(y2 - 4.0, 2);


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

void heun_step(double param, const double *incoming_activity, const double *state, double *next){
        double dleft[NSV];
        double dright[NSV];
        double nexteuler[NSV];

        model_dfun(state, incoming_activity, param, dleft);

        for (int sv = 0; sv < NSV; ++sv) {
                nexteuler[sv] = state[sv] + DT * dleft[sv];
        }
        model_dfun(nexteuler, incoming_activity, param, dright);
        for (int sv = 0; sv < NSV; ++sv) {
                next[sv] = state[sv] + 0.5 * DT * (dleft[sv] + dright[sv]);
        }
}

