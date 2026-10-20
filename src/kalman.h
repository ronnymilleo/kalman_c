/**
 * @file    kalman.h
 * @brief   Discrete-time Kalman filter and Extended Kalman filter for state estimation.
 * @details The filter owns no heap memory: kalman_t holds every matrix inline, with room for up to KALMAN_MAX_DIM
 *          states, measurements and controls. The caller provides the kalman_t, calls kalman_init() on it and
 *          kalman_deinit() when done.
 *
 *          A filter predicts with the linear model `x = F x + B u` unless both the state transition function and its
 *          Jacobian are set, which switches the prediction to the Extended Kalman filter, `x = f(x, u)`. The update
 *          works the same way with the observation matrix `H` or the observation function and its Jacobian.
 */

#ifndef KALMAN_C_KALMAN_H
#define KALMAN_C_KALMAN_H

#include "matrix.h"

#include <stdbool.h>
#include <stddef.h>

// Largest number of states, measurements or controls a filter can have
#define KALMAN_MAX_DIM MATRIX_MAX_DIM

/**
 * @enum    kalman_status_t
 * @brief   Outcome of a filter operation.
 */
typedef enum kalman_status_t {
    kalman_status_ok = 0,
    kalman_status_null_argument,
    kalman_status_invalid_dimension,
    kalman_status_dimension_mismatch,
    kalman_status_index_out_of_range,
    kalman_status_not_initialized,
    kalman_status_incomplete_model,
    kalman_status_singular_innovation,
} kalman_status_t;

/**
 * @brief   Non-linear state transition `f(x, u)`.
 * @param[in]  state       Current state, `state_dim` values.
 * @param[in]  control     Control input, `control_dim` values, or null when there is none.
 * @param[out] next_state  Predicted state, `state_dim` values; it never shares storage with `state`.
 */
typedef void (*kalman_state_transition_fn)(const double *state, const double *control, double *next_state);

/**
 * @brief   Non-linear observation function `h(x)`.
 * @param[in]  state        Current state, `state_dim` values.
 * @param[out] measurement  Expected measurement, `measurement_dim` values.
 */
typedef void (*kalman_observation_fn)(const double *state, double *measurement);

/**
 * @brief   Jacobian of the state transition function with respect to the state.
 * @param[in]  state     State the Jacobian is evaluated at, `state_dim` values.
 * @param[out] jacobian  Matrix to fill, already sized `state_dim x state_dim`.
 */
typedef void (*kalman_state_jacobian_fn)(const double *state, matrix_t *jacobian);

/**
 * @brief   Jacobian of the observation function with respect to the state.
 * @param[in]  state     State the Jacobian is evaluated at, `state_dim` values.
 * @param[out] jacobian  Matrix to fill, already sized `measurement_dim x state_dim`.
 */
typedef void (*kalman_observation_jacobian_fn)(const double *state, matrix_t *jacobian);

/**
 * @struct  kalman_t
 * @brief   Everything a Kalman or Extended Kalman filter needs, stored inline.
 * @details The struct is public so the caller can allocate it, but its invariants are kept by convention: set it up
 *          with kalman_init() and the setters rather than writing the fields, which would skip the size checks. The
 *          dimensions never exceed KALMAN_MAX_DIM, each matrix has the size its dimensions imply, and the filter
 *          only predicts and updates once the error covariance has been set.
 */
typedef struct kalman_t {
    size_t state_dim;
    size_t measurement_dim;
    size_t control_dim;

    double state[KALMAN_MAX_DIM];

    // P, the uncertainty of the state estimate
    matrix_t error_covariance;
    // F, how the state evolves from one step to the next in the linear model
    matrix_t state_transition;
    // B, how the control input moves the state in the linear model
    matrix_t control_matrix;
    // Q, the uncertainty the model adds at every step
    matrix_t process_noise;
    // H, which part of the state the sensors observe in the linear model
    matrix_t observation_matrix;
    // R, the uncertainty of the measurements
    matrix_t measurement_noise;

    kalman_state_transition_fn state_transition_function;
    kalman_observation_fn observation_function;
    kalman_state_jacobian_fn state_jacobian_function;
    kalman_observation_jacobian_fn observation_jacobian_function;

    bool is_initialized;
} kalman_t;

// Lifecycle
kalman_status_t kalman_init(kalman_t *kf, size_t state_dim, size_t measurement_dim, size_t control_dim);
void kalman_deinit(kalman_t *kf);
kalman_status_t kalman_reset(kalman_t *kf);

// Linear model configuration
kalman_status_t kalman_initialize_state(kalman_t *kf, const double *initial_state);
kalman_status_t kalman_set_state_transition_matrix(kalman_t *kf, const matrix_t *f);
kalman_status_t kalman_set_control_matrix(kalman_t *kf, const matrix_t *b);
kalman_status_t kalman_set_observation_matrix(kalman_t *kf, const matrix_t *h);
kalman_status_t kalman_set_process_noise_covariance(kalman_t *kf, const matrix_t *q);
kalman_status_t kalman_set_measurement_noise_covariance(kalman_t *kf, const matrix_t *r);
kalman_status_t kalman_set_error_covariance(kalman_t *kf, const matrix_t *p);

// Extended Kalman filter configuration
kalman_status_t kalman_set_state_transition_function(kalman_t *kf, kalman_state_transition_fn f);
kalman_status_t kalman_set_observation_function(kalman_t *kf, kalman_observation_fn h);
kalman_status_t kalman_set_state_jacobian_function(kalman_t *kf, kalman_state_jacobian_fn jf);
kalman_status_t kalman_set_observation_jacobian_function(kalman_t *kf, kalman_observation_jacobian_fn jh);

// Filtering
kalman_status_t kalman_predict(kalman_t *kf, const double *control);
kalman_status_t kalman_update(kalman_t *kf, const double *measurement);

// Estimates
kalman_status_t kalman_get_state(const kalman_t *kf, double *state);
kalman_status_t kalman_get_state_element(const kalman_t *kf, size_t index, double *value);
kalman_status_t kalman_get_error_covariance(const kalman_t *kf, matrix_t *covariance);
bool kalman_is_initialized(const kalman_t *kf);

#endif // KALMAN_C_KALMAN_H
