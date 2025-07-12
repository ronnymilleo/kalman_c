/**
 * @file kalman.c
 * @brief Implementation of Kalman and Extended Kalman filters
 * @author ronnymilleo
 * @date 08/07/25
 * @version 1.1
 *
 * This file implements a complete discrete-time Kalman filter and EKF with
 * prediction and update steps. The implementation provides robust
 * error handling, memory management, and numerical stability.
 *
 * @details Implementation features:
 * - Complete filter cycle (predict/update) for both KF and EKF
 * - Support for control inputs (optional)
 * - Robust memory management with cleanup
 * - Error checking at each step
 * - Numerical stability considerations
 *
 * @note The implementation follows standard filter equations
 * @note All intermediate calculations use temporary matrices for safety
 * @note Memory allocation failures are handled gracefully
 *
 * @see kalman.h for API documentation
 * @see matrix_math.h for underlying matrix operations
 */

#include "kalman.h"
#include "matrix_math.h"
#include <stdlib.h>
#include <string.h>

kalman_t* kalman_create(const int stateDimension, const int measurementDimension, const int controlDimension) {
    if (stateDimension <= 0 || measurementDimension <= 0 || controlDimension < 0) {
        return NULL;
    }

    kalman_t *kf = (kalman_t*)malloc(sizeof(kalman_t));
    if (!kf) return NULL;

    // Initialize dimensions
    kf->stateDim = stateDimension;
    kf->measurementDim = measurementDimension;
    kf->controlDim = controlDimension;
    kf->initialized = false;

    // Initialize EKF function pointers to NULL
    kf->state_transition_func = NULL;
    kf->observation_func = NULL;
    kf->state_jacobian_func = NULL;
    kf->observation_jacobian_func = NULL;

    // Allocate memory for state vector
    kf->state = (double*)calloc(stateDimension, sizeof(double));
    if (!kf->state) {
        free(kf);
        return NULL;
    }

    // Allocate memory for matrices
    kf->errorCovariance = matrix_allocate(stateDimension, stateDimension);
    kf->stateTransition = matrix_allocate(stateDimension, stateDimension);
    kf->processNoise = matrix_allocate(stateDimension, stateDimension);
    kf->observationMatrix = matrix_allocate(measurementDimension, stateDimension);
    kf->measurementNoise = matrix_allocate(measurementDimension, measurementDimension);
    kf->identity = matrix_allocate(stateDimension, stateDimension);

    if (!kf->errorCovariance || !kf->stateTransition || !kf->processNoise ||
        !kf->observationMatrix || !kf->measurementNoise || !kf->identity) {
        kalman_destroy(kf);
        return NULL;
    }

    // Allocate control matrix if needed
    if (controlDimension > 0) {
        kf->controlMatrix = matrix_allocate(stateDimension, controlDimension);
        if (!kf->controlMatrix) {
            kalman_destroy(kf);
            return NULL;
        }
    } else {
        kf->controlMatrix = NULL;
    }

    // Initialize identity matrix
    matrix_identity(kf->identity, stateDimension);

    return kf;
}

void kalman_destroy(kalman_t *kf) {
    if (!kf) return;

    free(kf->state);
    matrix_free(kf->errorCovariance, kf->stateDim);
    matrix_free(kf->stateTransition, kf->stateDim);
    matrix_free(kf->processNoise, kf->stateDim);
    matrix_free(kf->observationMatrix, kf->measurementDim);
    matrix_free(kf->measurementNoise, kf->measurementDim);
    matrix_free(kf->identity, kf->stateDim);

    if (kf->controlMatrix) {
        matrix_free(kf->controlMatrix, kf->stateDim);
    }

    free(kf);
}

int kalman_initialize_state(kalman_t *kf, const double *initialState) {
    if (!kf || !initialState) return -1;

    for (size_t i = 0; i < kf->stateDim; i++) {
        kf->state[i] = initialState[i];
    }
    return 0;
}

int kalman_set_state_transition_matrix(kalman_t *kf, double **F) {
    if (!kf || !F) return -1;

    matrix_copy(F, kf->stateTransition, kf->stateDim, kf->stateDim);
    return 0;
}

int kalman_set_control_matrix(kalman_t *kf, double **B) {
    if (!kf || !B || kf->controlDim == 0) return -1;

    matrix_copy(B, kf->controlMatrix, kf->stateDim, kf->controlDim);
    return 0;
}

int kalman_set_observation_matrix(kalman_t *kf, double **H) {
    if (!kf || !H) return -1;

    matrix_copy(H, kf->observationMatrix, kf->measurementDim, kf->stateDim);
    return 0;
}

int kalman_set_process_noise_covariance(kalman_t *kf, double **Q) {
    if (!kf || !Q) return -1;

    matrix_copy(Q, kf->processNoise, kf->stateDim, kf->stateDim);
    return 0;
}

int kalman_set_measurement_noise_covariance(kalman_t *kf, double **R) {
    if (!kf || !R) return -1;

    matrix_copy(R, kf->measurementNoise, kf->measurementDim, kf->measurementDim);
    return 0;
}

int kalman_set_error_covariance(kalman_t *kf, double **P) {
    if (!kf || !P) return -1;

    matrix_copy(P, kf->errorCovariance, kf->stateDim, kf->stateDim);
    kf->initialized = true;
    return 0;
}

// EKF configuration functions
int kalman_set_state_transition_function(kalman_t *kf, const ekf_state_transition_fn f) {
    if (!kf) return -1;
    kf->state_transition_func = f;
    return 0;
}

int kalman_set_observation_function(kalman_t *kf, const ekf_observation_fn h) {
    if (!kf) return -1;
    kf->observation_func = h;
    return 0;
}

int kalman_set_state_jacobian_function(kalman_t *kf, const ekf_state_jacobian_fn Jf) {
    if (!kf) return -1;
    kf->state_jacobian_func = Jf;
    return 0;
}

int kalman_set_observation_jacobian_function(kalman_t *kf, const ekf_observation_jacobian_fn Jh) {
    if (!kf) return -1;
    kf->observation_jacobian_func = Jh;
    return 0;
}


int kalman_predict(kalman_t *kf, const double *control) {
    if (!kf || !kf->initialized) return -1;

    // EKF prediction
    if (kf->state_transition_func && kf->state_jacobian_func) {
        // Predict state using non-linear function: x = f(x, u)
        kf->state_transition_func(kf->state, control, kf->state);

        // Update Jacobian matrix Fj
        kf->state_jacobian_func(kf->state, kf->stateTransition);
    }
    // Standard KF prediction
    else {
        double *result = (double*)malloc(kf->stateDim * sizeof(double));
        if (!result) return -1;

        matrix_vector_multiply(kf->stateTransition, kf->state, result, kf->stateDim, kf->stateDim);

        for (size_t i = 0; i < kf->stateDim; i++) {
            kf->state[i] = result[i];
        }
        free(result);

        if (control && kf->controlMatrix && kf->controlDim > 0) {
            double *control_effect = (double*)malloc(kf->stateDim * sizeof(double));
            if (!control_effect) return -1;

            matrix_vector_multiply(kf->controlMatrix, control, control_effect, kf->stateDim, kf->controlDim);
            for (size_t i = 0; i < kf->stateDim; i++) {
                kf->state[i] += control_effect[i];
            }
            free(control_effect);
        }
    }

    // Predict error covariance: P = F * P * F^T + Q (where F is Jacobian for EKF)
    double **FP = matrix_allocate(kf->stateDim, kf->stateDim);
    double **FT = matrix_allocate(kf->stateDim, kf->stateDim);
    double **FPFT = matrix_allocate(kf->stateDim, kf->stateDim);

    if (!FP || !FT || !FPFT) {
        matrix_free(FP, kf->stateDim);
        matrix_free(FT, kf->stateDim);
        matrix_free(FPFT, kf->stateDim);
        return -1;
    }

    matrix_multiply(kf->stateTransition, kf->errorCovariance, FP, kf->stateDim, kf->stateDim, kf->stateDim);
    matrix_transpose(kf->stateTransition, FT, kf->stateDim, kf->stateDim);
    matrix_multiply(FP, FT, FPFT, kf->stateDim, kf->stateDim, kf->stateDim);
    matrix_add(FPFT, kf->processNoise, kf->errorCovariance, kf->stateDim, kf->stateDim);

    matrix_free(FP, kf->stateDim);
    matrix_free(FT, kf->stateDim);
    matrix_free(FPFT, kf->stateDim);

    return 0;
}

int kalman_update(kalman_t *kf, const double *measurement) {
    if (!kf || !kf->initialized || !measurement) return -1;

    double *predicted_measurement = (double*)malloc(kf->measurementDim * sizeof(double));
    if (!predicted_measurement) return -1;

    // EKF update
    if (kf->observation_func && kf->observation_jacobian_func) {
        // Predict measurement using non-linear function: z_pred = h(x)
        kf->observation_func(kf->state, predicted_measurement);

        // Update Jacobian matrix Hj
        kf->observation_jacobian_func(kf->state, kf->observationMatrix);
    }
    // Standard KF update
    else {
        matrix_vector_multiply(kf->observationMatrix, kf->state, predicted_measurement, kf->measurementDim, kf->stateDim);
    }

    // Innovation: y = z - z_pred
    double *innovation = (double*)malloc(kf->measurementDim * sizeof(double));
    if (!innovation) {
        free(predicted_measurement);
        return -1;
    }
    for (size_t i = 0; i < kf->measurementDim; i++) {
        innovation[i] = measurement[i] - predicted_measurement[i];
    }
    free(predicted_measurement);

    // Innovation covariance: S = H * P * H^T + R
    double **HP = matrix_allocate(kf->measurementDim, kf->stateDim);
    double **HT = matrix_allocate(kf->stateDim, kf->measurementDim);
    double **S = matrix_allocate(kf->measurementDim, kf->measurementDim);
    if (!HP || !HT || !S) {
        free(innovation);
        matrix_free(HP, kf->measurementDim);
        matrix_free(HT, kf->stateDim);
        matrix_free(S, kf->measurementDim);
        return -1;
    }

    matrix_multiply(kf->observationMatrix, kf->errorCovariance, HP, kf->measurementDim, kf->stateDim, kf->stateDim);
    matrix_transpose(kf->observationMatrix, HT, kf->measurementDim, kf->stateDim);
    matrix_multiply(HP, HT, S, kf->measurementDim, kf->stateDim, kf->measurementDim);
    matrix_add(S, kf->measurementNoise, S, kf->measurementDim, kf->measurementDim);

    // Kalman gain: K = P * H^T * S^(-1)
    double **PHT = matrix_allocate(kf->stateDim, kf->measurementDim);
    double **S_inv = matrix_allocate(kf->measurementDim, kf->measurementDim);
    double **K = matrix_allocate(kf->stateDim, kf->measurementDim);
    if (!PHT || !S_inv || !K) {
        free(innovation);
        matrix_free(HP, kf->measurementDim);
        matrix_free(HT, kf->stateDim);
        matrix_free(S, kf->measurementDim);
        matrix_free(PHT, kf->stateDim);
        matrix_free(S_inv, kf->measurementDim);
        matrix_free(K, kf->stateDim);
        return -1;
    }

    matrix_multiply(kf->errorCovariance, HT, PHT, kf->stateDim, kf->stateDim, kf->measurementDim);
    if (matrix_inverse(S, S_inv, kf->measurementDim) != 0) {
        // Handle inversion failure
        free(innovation);
        matrix_free(HP, kf->measurementDim);
        matrix_free(HT, kf->stateDim);
        matrix_free(S, kf->measurementDim);
        matrix_free(PHT, kf->stateDim);
        matrix_free(S_inv, kf->measurementDim);
        matrix_free(K, kf->stateDim);
        return -1;
    }
    matrix_multiply(PHT, S_inv, K, kf->stateDim, kf->measurementDim, kf->measurementDim);

    // Update state: x = x + K * y
    double *correction = (double*)malloc(kf->stateDim * sizeof(double));
    if (!correction) {
        free(innovation);
        matrix_free(HP, kf->measurementDim);
        matrix_free(HT, kf->stateDim);
        matrix_free(S, kf->measurementDim);
        matrix_free(PHT, kf->stateDim);
        matrix_free(S_inv, kf->measurementDim);
        matrix_free(K, kf->stateDim);
        return -1;
    }
    matrix_vector_multiply(K, innovation, correction, kf->stateDim, kf->measurementDim);
    for (size_t i = 0; i < kf->stateDim; i++) {
        kf->state[i] += correction[i];
    }
    free(correction);
    free(innovation);

    // Update error covariance: P = (I - K * H) * P
    double **KH = matrix_allocate(kf->stateDim, kf->stateDim);
    double **IKH = matrix_allocate(kf->stateDim, kf->stateDim);
    double **newP = matrix_allocate(kf->stateDim, kf->stateDim);
    if (!KH || !IKH || !newP) {
        matrix_free(HP, kf->measurementDim);
        matrix_free(HT, kf->stateDim);
        matrix_free(S, kf->measurementDim);
        matrix_free(PHT, kf->stateDim);
        matrix_free(S_inv, kf->measurementDim);
        matrix_free(K, kf->stateDim);
        matrix_free(KH, kf->stateDim);
        matrix_free(IKH, kf->stateDim);
        matrix_free(newP, kf->stateDim);
        return -1;
    }

    matrix_multiply(K, kf->observationMatrix, KH, kf->stateDim, kf->measurementDim, kf->stateDim);
    matrix_subtract(kf->identity, KH, IKH, kf->stateDim, kf->stateDim);
    matrix_multiply(IKH, kf->errorCovariance, newP, kf->stateDim, kf->stateDim, kf->stateDim);
    matrix_copy(newP, kf->errorCovariance, kf->stateDim, kf->stateDim);

    // Cleanup
    matrix_free(HP, kf->measurementDim);
    matrix_free(HT, kf->stateDim);
    matrix_free(S, kf->measurementDim);
    matrix_free(PHT, kf->stateDim);
    matrix_free(S_inv, kf->measurementDim);
    matrix_free(K, kf->stateDim);
    matrix_free(KH, kf->stateDim);
    matrix_free(IKH, kf->stateDim);
    matrix_free(newP, kf->stateDim);

    return 0;
}

double* kalman_get_state(kalman_t *kf) {
    if (!kf) return NULL;
    return kf->state;
}

double** kalman_get_error_covariance(kalman_t *kf) {
    if (!kf) return NULL;
    return kf->errorCovariance;
}

double kalman_get_state_element(const kalman_t *kf, const int index) {
    if (!kf || index < 0 || (size_t)index >= kf->stateDim) return 0.0;
    return kf->state[index];
}

void kalman_reset(kalman_t *kf) {
    if (!kf) return;

    for (size_t i = 0; i < kf->stateDim; i++) {
        kf->state[i] = 0.0;
        for (size_t j = 0; j < kf->stateDim; j++) {
            kf->errorCovariance[i][j] = 0.0;
        }
    }
    kf->initialized = false;
}

bool kalman_is_initialized(const kalman_t *kf) {
    if (!kf) return false;
    return kf->initialized;
}
