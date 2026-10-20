/**
 * @file    kalman.c
 * @brief   Discrete-time Kalman filter and Extended Kalman filter for state estimation.
 */

#include "kalman.h"

static kalman_status_t from_matrix_status(const matrix_status_t status) {
    switch (status) {
    case matrix_status_ok:
        return kalman_status_ok;
    case matrix_status_null_argument:
        return kalman_status_null_argument;
    case matrix_status_singular:
        return kalman_status_singular_innovation;
    case matrix_status_invalid_size:
    case matrix_status_size_mismatch:
    default:
        return kalman_status_dimension_mismatch;
    }
}

static bool has_size(const matrix_t *matrix, const size_t rows, const size_t cols) {
    return matrix->rows == rows && matrix->cols == cols;
}

static kalman_status_t copy_matrix(const matrix_t *source, matrix_t *destination, const size_t rows,
                                   const size_t cols) {
    if (source == NULL) {
        return kalman_status_null_argument;
    }
    if (!has_size(source, rows, cols)) {
        return kalman_status_dimension_mismatch;
    }
    *destination = *source;
    return kalman_status_ok;
}

// The two functions of a non-linear model only work as a pair: a Jacobian without its function, or the other way
// round, would silently fall back to the linear model and its default matrices
static bool is_state_model_incomplete(const kalman_t *kf) {
    return (kf->state_transition_function == NULL) != (kf->state_jacobian_function == NULL);
}

static bool is_observation_model_incomplete(const kalman_t *kf) {
    return (kf->observation_function == NULL) != (kf->observation_jacobian_function == NULL);
}

/**
 * @brief   Prepares a filter with the given dimensions.
 * @details The state transition matrix starts as the identity and every other matrix as zeros. The filter is not
 *          ready to predict or update until kalman_set_error_covariance() has been called.
 * @param[out] kf               Filter to set up.
 * @param[in]  state_dim        Number of states, from 1 to KALMAN_MAX_DIM.
 * @param[in]  measurement_dim  Number of measurements, from 1 to KALMAN_MAX_DIM.
 * @param[in]  control_dim      Number of control inputs, from 0 to KALMAN_MAX_DIM.
 * @return  kalman_status_ok, or why the filter could not be set up.
 */
kalman_status_t kalman_init(kalman_t *kf, const size_t state_dim, const size_t measurement_dim,
                            const size_t control_dim) {
    if (kf == NULL) {
        return kalman_status_null_argument;
    }
    if (state_dim < 1 || state_dim > KALMAN_MAX_DIM || measurement_dim < 1 || measurement_dim > KALMAN_MAX_DIM ||
        control_dim > KALMAN_MAX_DIM) {
        return kalman_status_invalid_dimension;
    }

    *kf = (kalman_t){.state_dim = state_dim, .measurement_dim = measurement_dim, .control_dim = control_dim};

    matrix_status_t status = matrix_identity(&kf->state_transition, state_dim);
    if (status == matrix_status_ok) {
        status = matrix_init(&kf->error_covariance, state_dim, state_dim);
    }
    if (status == matrix_status_ok) {
        status = matrix_init(&kf->process_noise, state_dim, state_dim);
    }
    if (status == matrix_status_ok) {
        status = matrix_init(&kf->observation_matrix, measurement_dim, state_dim);
    }
    if (status == matrix_status_ok) {
        status = matrix_init(&kf->measurement_noise, measurement_dim, measurement_dim);
    }
    if (status == matrix_status_ok && control_dim > 0) {
        status = matrix_init(&kf->control_matrix, state_dim, control_dim);
    }
    return from_matrix_status(status);
}

/**
 * @brief   Clears a filter so no stale model or estimate can be used by mistake.
 * @param[in,out] kf  Filter to clear, or null to do nothing; it must be initialized again before use.
 */
void kalman_deinit(kalman_t *kf) {
    if (kf != NULL) {
        *kf = (kalman_t){0};
    }
}

/**
 * @brief   Zeroes the state and the error covariance and marks the filter as not ready.
 * @details The dimensions, the model matrices and the non-linear functions are kept, so the filter can be started
 *          again with new initial conditions.
 * @param[in,out] kf  Filter to reset.
 * @return  kalman_status_ok, or why the filter could not be reset.
 */
kalman_status_t kalman_reset(kalman_t *kf) {
    if (kf == NULL) {
        return kalman_status_null_argument;
    }
    for (size_t index = 0; index < kf->state_dim; index++) {
        kf->state[index] = 0.0;
    }
    const matrix_status_t status = matrix_init(&kf->error_covariance, kf->state_dim, kf->state_dim);
    kf->is_initialized = false;
    return from_matrix_status(status);
}

/**
 * @brief   Sets the state estimate.
 * @param[in,out] kf             Filter to update.
 * @param[in]     initial_state  `state_dim` values.
 * @return  kalman_status_ok, or why the state could not be set.
 */
kalman_status_t kalman_initialize_state(kalman_t *kf, const double *initial_state) {
    if (kf == NULL || initial_state == NULL) {
        return kalman_status_null_argument;
    }
    for (size_t index = 0; index < kf->state_dim; index++) {
        kf->state[index] = initial_state[index];
    }
    return kalman_status_ok;
}

/**
 * @brief   Sets the state transition matrix `F` of the linear model.
 * @param[in,out] kf  Filter to update.
 * @param[in]     f   Matrix of size `state_dim x state_dim`.
 * @return  kalman_status_ok, or why the matrix was rejected.
 */
kalman_status_t kalman_set_state_transition_matrix(kalman_t *kf, const matrix_t *f) {
    if (kf == NULL) {
        return kalman_status_null_argument;
    }
    return copy_matrix(f, &kf->state_transition, kf->state_dim, kf->state_dim);
}

/**
 * @brief   Sets the control matrix `B` of the linear model.
 * @param[in,out] kf  Filter to update.
 * @param[in]     b   Matrix of size `state_dim x control_dim`; the filter must have been set up with controls.
 * @return  kalman_status_ok, or why the matrix was rejected.
 */
kalman_status_t kalman_set_control_matrix(kalman_t *kf, const matrix_t *b) {
    if (kf == NULL) {
        return kalman_status_null_argument;
    }
    return copy_matrix(b, &kf->control_matrix, kf->state_dim, kf->control_dim);
}

/**
 * @brief   Sets the observation matrix `H` of the linear model.
 * @param[in,out] kf  Filter to update.
 * @param[in]     h   Matrix of size `measurement_dim x state_dim`.
 * @return  kalman_status_ok, or why the matrix was rejected.
 */
kalman_status_t kalman_set_observation_matrix(kalman_t *kf, const matrix_t *h) {
    if (kf == NULL) {
        return kalman_status_null_argument;
    }
    return copy_matrix(h, &kf->observation_matrix, kf->measurement_dim, kf->state_dim);
}

/**
 * @brief   Sets the process noise covariance `Q`.
 * @param[in,out] kf  Filter to update.
 * @param[in]     q   Matrix of size `state_dim x state_dim`.
 * @return  kalman_status_ok, or why the matrix was rejected.
 */
kalman_status_t kalman_set_process_noise_covariance(kalman_t *kf, const matrix_t *q) {
    if (kf == NULL) {
        return kalman_status_null_argument;
    }
    return copy_matrix(q, &kf->process_noise, kf->state_dim, kf->state_dim);
}

/**
 * @brief   Sets the measurement noise covariance `R`.
 * @param[in,out] kf  Filter to update.
 * @param[in]     r   Matrix of size `measurement_dim x measurement_dim`.
 * @return  kalman_status_ok, or why the matrix was rejected.
 */
kalman_status_t kalman_set_measurement_noise_covariance(kalman_t *kf, const matrix_t *r) {
    if (kf == NULL) {
        return kalman_status_null_argument;
    }
    return copy_matrix(r, &kf->measurement_noise, kf->measurement_dim, kf->measurement_dim);
}

/**
 * @brief   Sets the error covariance `P`, which also makes the filter ready to predict and update.
 * @param[in,out] kf  Filter to update.
 * @param[in]     p   Matrix of size `state_dim x state_dim`.
 * @return  kalman_status_ok, or why the matrix was rejected.
 */
kalman_status_t kalman_set_error_covariance(kalman_t *kf, const matrix_t *p) {
    if (kf == NULL) {
        return kalman_status_null_argument;
    }
    const kalman_status_t status = copy_matrix(p, &kf->error_covariance, kf->state_dim, kf->state_dim);
    if (status == kalman_status_ok) {
        kf->is_initialized = true;
    }
    return status;
}

/**
 * @brief   Sets the non-linear state transition function `f(x, u)` of the Extended Kalman filter.
 * @param[in,out] kf  Filter to update.
 * @param[in]     f   Function to use, or null to go back to the linear model; it must be paired with a Jacobian.
 * @return  kalman_status_ok, or why the function was rejected.
 */
kalman_status_t kalman_set_state_transition_function(kalman_t *kf, const kalman_state_transition_fn f) {
    if (kf == NULL) {
        return kalman_status_null_argument;
    }
    kf->state_transition_function = f;
    return kalman_status_ok;
}

/**
 * @brief   Sets the non-linear observation function `h(x)` of the Extended Kalman filter.
 * @param[in,out] kf  Filter to update.
 * @param[in]     h   Function to use, or null to go back to the linear model; it must be paired with a Jacobian.
 * @return  kalman_status_ok, or why the function was rejected.
 */
kalman_status_t kalman_set_observation_function(kalman_t *kf, const kalman_observation_fn h) {
    if (kf == NULL) {
        return kalman_status_null_argument;
    }
    kf->observation_function = h;
    return kalman_status_ok;
}

/**
 * @brief   Sets the Jacobian of the state transition function.
 * @param[in,out] kf  Filter to update.
 * @param[in]     jf  Function to use, or null to go back to the linear model; it must be paired with `f(x, u)`.
 * @return  kalman_status_ok, or why the function was rejected.
 */
kalman_status_t kalman_set_state_jacobian_function(kalman_t *kf, const kalman_state_jacobian_fn jf) {
    if (kf == NULL) {
        return kalman_status_null_argument;
    }
    kf->state_jacobian_function = jf;
    return kalman_status_ok;
}

/**
 * @brief   Sets the Jacobian of the observation function.
 * @param[in,out] kf  Filter to update.
 * @param[in]     jh  Function to use, or null to go back to the linear model; it must be paired with `h(x)`.
 * @return  kalman_status_ok, or why the function was rejected.
 */
kalman_status_t kalman_set_observation_jacobian_function(kalman_t *kf, const kalman_observation_jacobian_fn jh) {
    if (kf == NULL) {
        return kalman_status_null_argument;
    }
    kf->observation_jacobian_function = jh;
    return kalman_status_ok;
}

/**
 * @brief   Moves the estimate one step forward in time.
 * @details Linear model: `x = F x + B u` and `P = F P F^T + Q`. When the state transition function and its Jacobian
 *          are set, `x = f(x, u)` and `F` is the Jacobian evaluated at the estimate the step starts from. The filter is
 *          left unchanged when the call fails.
 * @param[in,out] kf       Filter to advance.
 * @param[in]     control  `control_dim` values, or null when there is no control input.
 * @return  kalman_status_ok, or why the step could not be taken.
 */
kalman_status_t kalman_predict(kalman_t *kf, const double *control) {
    if (kf == NULL) {
        return kalman_status_null_argument;
    }
    if (!kf->is_initialized) {
        return kalman_status_not_initialized;
    }
    if (is_state_model_incomplete(kf)) {
        return kalman_status_incomplete_model;
    }

    matrix_t f = kf->state_transition;
    double next_state[KALMAN_MAX_DIM] = {0.0};
    matrix_status_t status = matrix_status_ok;

    if (kf->state_transition_function != NULL) {
        kf->state_jacobian_function(kf->state, &f);
        kf->state_transition_function(kf->state, control, next_state);
    } else {
        status = matrix_vector_multiply(&kf->state_transition, kf->state, next_state);
        if (status == matrix_status_ok && control != NULL && kf->control_dim > 0) {
            double control_effect[KALMAN_MAX_DIM] = {0.0};
            status = matrix_vector_multiply(&kf->control_matrix, control, control_effect);
            for (size_t index = 0; index < kf->state_dim; index++) {
                next_state[index] += control_effect[index];
            }
        }
    }

    matrix_t f_transposed;
    matrix_t covariance;
    if (status == matrix_status_ok) {
        status = matrix_transpose(&f, &f_transposed);
    }
    if (status == matrix_status_ok) {
        status = matrix_multiply(&f, &kf->error_covariance, &covariance);
    }
    if (status == matrix_status_ok) {
        status = matrix_multiply(&covariance, &f_transposed, &covariance);
    }
    if (status == matrix_status_ok) {
        status = matrix_add(&covariance, &kf->process_noise, &covariance);
    }
    if (status != matrix_status_ok) {
        return from_matrix_status(status);
    }

    for (size_t index = 0; index < kf->state_dim; index++) {
        kf->state[index] = next_state[index];
    }
    kf->error_covariance = covariance;
    return kalman_status_ok;
}

/**
 * @brief   Corrects the estimate with a measurement.
 * @details Computes the innovation `y = z - H x`, its covariance `S = H P H^T + R`, the gain `K = P H^T S^-1`, and
 *          then `x = x + K y` and `P = (I - K H) P`. When the observation function and its Jacobian are set, `H x`
 *          becomes `h(x)` and `H` the Jacobian evaluated at the current estimate. The filter is left unchanged when
 *          the call fails.
 * @param[in,out] kf           Filter to correct.
 * @param[in]     measurement  `measurement_dim` values.
 * @return  kalman_status_ok, kalman_status_singular_innovation when `S` cannot be inverted, or why the correction
 *          could not be made.
 */
kalman_status_t kalman_update(kalman_t *kf, const double *measurement) {
    if (kf == NULL || measurement == NULL) {
        return kalman_status_null_argument;
    }
    if (!kf->is_initialized) {
        return kalman_status_not_initialized;
    }
    if (is_observation_model_incomplete(kf)) {
        return kalman_status_incomplete_model;
    }

    matrix_t h = kf->observation_matrix;
    double expected_measurement[KALMAN_MAX_DIM] = {0.0};
    matrix_status_t status = matrix_status_ok;

    if (kf->observation_function != NULL) {
        kf->observation_jacobian_function(kf->state, &h);
        kf->observation_function(kf->state, expected_measurement);
    } else {
        status = matrix_vector_multiply(&h, kf->state, expected_measurement);
    }

    double innovation[KALMAN_MAX_DIM] = {0.0};
    for (size_t index = 0; index < kf->measurement_dim; index++) {
        innovation[index] = measurement[index] - expected_measurement[index];
    }

    // S = H P H^T + R
    matrix_t h_transposed;
    matrix_t innovation_covariance;
    if (status == matrix_status_ok) {
        status = matrix_transpose(&h, &h_transposed);
    }
    if (status == matrix_status_ok) {
        status = matrix_multiply(&h, &kf->error_covariance, &innovation_covariance);
    }
    if (status == matrix_status_ok) {
        status = matrix_multiply(&innovation_covariance, &h_transposed, &innovation_covariance);
    }
    if (status == matrix_status_ok) {
        status = matrix_add(&innovation_covariance, &kf->measurement_noise, &innovation_covariance);
    }

    // K = P H^T S^-1
    matrix_t gain;
    matrix_t innovation_covariance_inverse;
    if (status == matrix_status_ok) {
        status = matrix_inverse(&innovation_covariance, &innovation_covariance_inverse);
    }
    if (status == matrix_status_ok) {
        status = matrix_multiply(&kf->error_covariance, &h_transposed, &gain);
    }
    if (status == matrix_status_ok) {
        status = matrix_multiply(&gain, &innovation_covariance_inverse, &gain);
    }

    // x = x + K y
    double correction[KALMAN_MAX_DIM] = {0.0};
    if (status == matrix_status_ok) {
        status = matrix_vector_multiply(&gain, innovation, correction);
    }

    // P = (I - K H) P
    matrix_t covariance;
    matrix_t identity;
    if (status == matrix_status_ok) {
        status = matrix_multiply(&gain, &h, &covariance);
    }
    if (status == matrix_status_ok) {
        status = matrix_identity(&identity, kf->state_dim);
    }
    if (status == matrix_status_ok) {
        status = matrix_subtract(&identity, &covariance, &covariance);
    }
    if (status == matrix_status_ok) {
        status = matrix_multiply(&covariance, &kf->error_covariance, &covariance);
    }
    if (status != matrix_status_ok) {
        return from_matrix_status(status);
    }

    for (size_t index = 0; index < kf->state_dim; index++) {
        kf->state[index] += correction[index];
    }
    kf->error_covariance = covariance;
    return kalman_status_ok;
}

/**
 * @brief   Copies the state estimate out of the filter.
 * @param[in]  kf     Filter to read.
 * @param[out] state  Array that receives `state_dim` values.
 * @return  kalman_status_ok, or why the state could not be read.
 */
kalman_status_t kalman_get_state(const kalman_t *kf, double *state) {
    if (kf == NULL || state == NULL) {
        return kalman_status_null_argument;
    }
    for (size_t index = 0; index < kf->state_dim; index++) {
        state[index] = kf->state[index];
    }
    return kalman_status_ok;
}

/**
 * @brief   Reads one element of the state estimate.
 * @param[in]  kf     Filter to read.
 * @param[in]  index  Position in the state vector, from 0 to `state_dim - 1`.
 * @param[out] value  Receives the element.
 * @return  kalman_status_ok, or why the element could not be read.
 */
kalman_status_t kalman_get_state_element(const kalman_t *kf, const size_t index, double *value) {
    if (kf == NULL || value == NULL) {
        return kalman_status_null_argument;
    }
    if (index >= kf->state_dim) {
        return kalman_status_index_out_of_range;
    }
    *value = kf->state[index];
    return kalman_status_ok;
}

/**
 * @brief   Copies the error covariance out of the filter.
 * @param[in]  kf          Filter to read.
 * @param[out] covariance  Receives the `state_dim x state_dim` matrix `P`.
 * @return  kalman_status_ok, or why the covariance could not be read.
 */
kalman_status_t kalman_get_error_covariance(const kalman_t *kf, matrix_t *covariance) {
    if (kf == NULL || covariance == NULL) {
        return kalman_status_null_argument;
    }
    *covariance = kf->error_covariance;
    return kalman_status_ok;
}

/**
 * @brief   Tells whether the filter is ready to predict and update.
 * @param[in] kf  Filter to check, or null.
 * @return  True once the error covariance has been set; false for a null filter or after a reset.
 */
bool kalman_is_initialized(const kalman_t *kf) {
    return kf != NULL && kf->is_initialized;
}
