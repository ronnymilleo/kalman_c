/**
 * @file test_ekf.c
 * @brief Unit tests for the Extended Kalman Filter (EKF) implementation
 * @author ronnymilleo
 * @date 11/07/25
 * @version 1.0
 *
 * This file contains unit tests for the EKF functionality, ensuring that the
 * non-linear prediction and update steps work correctly. It uses a simple
 * non-linear model to validate the filter's behavior.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../kalman.h"
#include "../matrix_math.h"

// Simple non-linear state transition function: f(x) = [x[0]^2, x[1]]
void state_transition_function(const double *state_in, const double *control_in, double *state_out) {
    (void)control_in; // Ignore control input for this test
    state_out[0] = state_in[0] * state_in[0];
    state_out[1] = state_in[1];
}

// Jacobian of the state transition function
void state_jacobian_function(const double *state_in, double **jacobian_out) {
    jacobian_out[0][0] = 2 * state_in[0];
    jacobian_out[0][1] = 0.0;
    jacobian_out[1][0] = 0.0;
    jacobian_out[1][1] = 1.0;
}

// Simple non-linear observation function: h(x) = [sqrt(x[0]^2 + x[1]^2)]
void observation_function(const double *state_in, double *measurement_out) {
    measurement_out[0] = sqrt(state_in[0] * state_in[0] + state_in[1] * state_in[1]);
}

// Jacobian of the observation function
void observation_jacobian_function(const double *state_in, double **jacobian_out) {
    const double denominator = sqrt(state_in[0] * state_in[0] + state_in[1] * state_in[1]);
    if (denominator > 1e-6) {
        jacobian_out[0][0] = state_in[0] / denominator;
        jacobian_out[0][1] = state_in[1] / denominator;
    } else {
        jacobian_out[0][0] = 0.0;
        jacobian_out[0][1] = 0.0;
    }
}

int main(void) {
    printf("Running EKF tests...\n");

    kalman_t *ekf = kalman_create(2, 1, 0);
    if (!ekf) {
        fprintf(stderr, "EKF creation failed.\n");
        return 1;
    }

    // Set EKF functions
    kalman_set_state_transition_function(ekf, state_transition_function);
    kalman_set_state_jacobian_function(ekf, state_jacobian_function);
    kalman_set_observation_function(ekf, observation_function);
    kalman_set_observation_jacobian_function(ekf, observation_jacobian_function);

    // Initialize state
    const double initialState[2] = {1.5, 1.0};
    kalman_initialize_state(ekf, initialState);

    // Set covariances
    double **P = matrix_allocate(2, 2);
    P[0][0] = 1.0; P[0][1] = 0.0;
    P[1][0] = 0.0; P[1][1] = 1.0;
    kalman_set_error_covariance(ekf, P);

    double **Q = matrix_allocate(2, 2);
    Q[0][0] = 0.1; Q[0][1] = 0.0;
    Q[1][0] = 0.0; Q[1][1] = 0.1;
    kalman_set_process_noise_covariance(ekf, Q);

    double **R = matrix_allocate(1, 1);
    R[0][0] = 0.2;
    kalman_set_measurement_noise_covariance(ekf, R);

    // Predict
    if (kalman_predict(ekf, NULL) != 0) {
        fprintf(stderr, "EKF prediction failed.\n");
        return 1;
    }

    printf("Predicted state: [%f, %f]\n", kalman_get_state_element(ekf, 0), kalman_get_state_element(ekf, 1));

    // Update
    const double measurement[1] = {2.5};
    if (kalman_update(ekf, measurement) != 0) {
        fprintf(stderr, "EKF update failed.\n");
        return 1;
    }

    printf("Updated state: [%f, %f]\n", kalman_get_state_element(ekf, 0), kalman_get_state_element(ekf, 1));

    // Cleanup
    matrix_free(P, 2);
    matrix_free(Q, 2);
    matrix_free(R, 1);
    kalman_destroy(ekf);

    printf("EKF tests completed successfully.\n");
    return 0;
}
