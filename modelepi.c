#include <stdlib.h>
#include "sim.h"

struct ModelEpi * new_model(int len) {
        struct ModelEpi *self = malloc(sizeof(struct ModelEpi));
        self->len = len;
        self->a = malloc(sizeof(double) * len);
        self->b = malloc(sizeof(double) * len);
        self->c = malloc(sizeof(double) * len);
        self->d = malloc(sizeof(double) * len);
        self->aa = malloc(sizeof(double) * len);
        self->r = malloc(sizeof(double) * len);
        
        self->Kvf = malloc(sizeof(double) * len);
        self->Kf = malloc(sizeof(double) * len);
        self->Ks = malloc(sizeof(double) * len);
        self->tau = malloc(sizeof(double) * len);
        self->Iext = malloc(sizeof(double) * len);
        self->Iext2 = malloc(sizeof(double) * len);
        self->slope = malloc(sizeof(double) * len);
        self->x0 = malloc(sizeof(double) * len);
        self->tt = malloc(sizeof(double) * len);
        
        return self;
}

void free_model(struct ModelEpi *self){
        free(self->a);
        free(self->b);
        free(self->c);
        free(self->d);
        free(self->aa);
        free(self->r);

        free(self->Kvf);
        free(self->Kf);
        free(self->Ks);
        free(self->tau);
        free(self->Iext);
        free(self->Iext2);
        free(self->slope);
        free(self->x0);
        free(self->tt);
        free(self);
}

