/** Implementation of generator.h */
#include "generator.h"
#include<math.h>
#include<stdio.h>
#include<stdlib.h>

int make_inliers(float* points_x, float* points_y, int n_inliers, 
                float* params, int n_params,
                float x_min, float x_max) {
    if(n_inliers < 2 || x_min == x_max || n_params < 2){
            return -1;
        }
    float step = (float) (x_max - x_min) / (n_inliers - 1);
    float x;
    float y;
    for(int i = 0; i < n_inliers; i++) {
        x = x_min + i * step;
        y = 0.0;
        for(int j = 0; j < n_params; j++){
            y += params[j] * pow(x, j);
        }
        points_x[i] = x;
        points_y[i] = y;
    }
    return 0;
}