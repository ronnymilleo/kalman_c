/**
 * @file kalman.h
 * @brief Kalman and Extended Kalman filter implementation for state estimation
 * @author ronnymilleo
 * @date 08/07/25
 * @version 1.1
 *
 * This file provides a complete implementation of a discrete-time Kalman filter
 * and an Extended Kalman Filter (EKF) for real-time state estimation.
 * The filter supports prediction and update (correction) steps with configurable
 * state dimensions, measurement dimensions, and optional control inputs.
 *
 * The implementation follows the standard filter equations:
 * - KF Prediction: x(k|k-1) = F * x(k-1|k-1) + B * u(k-1)
 * - EKF Prediction: x(k|k-1) = f(x(k-1|k-1), u(k-1))
 * - KF Update: x(k|k) = x(k|k-1) + K(k) * (z(k) - H * x(k|k-1))
 * - EKF Update: x(k|k) = x(k|k-1) + K(k) * (z(k) - h(x(k|k-1)))
 *
 * @note All matrices use double precision floating point arithmetic
 * @note Memory management is handled internally with proper cleanup functions
 * @note Error handling uses return codes: 0 for success, -1 for failure
 *
 * @see matrix_math.h for underlying matrix operations
 */

#ifndef KALMAN_H
#define KALMAN_H

#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup ekf_types EKF Function Types
 *  @brief Function pointer types for Extended Kalman Filter (EKF)
 *  @{
 */

/**
 * @brief Non-linear state transition function f(x, u)
 * @param state_in Input state vector (x) of size stateDim
 * @param control_in Control vector (u) of size controlDim can be NULL
 * @param state_out Output state vector of size stateDim
 */
typedef void (*ekf_state_transition_fn)(const double *state_in, const double *control_in, double *state_out);

/**
 * @brief Non-linear observation function h(x)
 * @param state_in Input state vector (x) of size stateDim
 * @param measurement_out Output measurement vector of size measurementDim
 */
typedef void (*ekf_observation_fn)(const double *state_in, double *measurement_out);

/**
 * @brief Jacobian of the state transition function with respect to the state
 * @param state_in Input state vector (x) of size stateDim
 * @param jacobian_out Output Jacobian matrix (Fj) of size stateDim x stateDim
 */
typedef void (*ekf_state_jacobian_fn)(const double *state_in, double **jacobian_out);

/**
 * @brief Jacobian of the observation function with respect to the state
 * @param state_in Input state vector (x) of size stateDim
 * @param jacobian_out Output Jacobian matrix (Hj) of size measurementDim x stateDim
 */
typedef void (*ekf_observation_jacobian_fn)(const double *state_in, double **jacobian_out);

/** @} */ // end of ekf_types group


/** @defgroup kalman_lifecycle Kalman Filter Lifecycle
 *  @brief Functions for creating, configuring, and destroying Kalman filters
 *  @{
 */

/**
 * @struct kalman_t
 * @brief Kalman/EKF data structure
 *
 * This structure contains all the matrices and parameters required for
 * filter operation. The structure is opaque to the user and should
 * only be accessed through the provided API functions.
 *
 * @details The structure contains:
 * - State vector (x): Current estimated state
 * - Error covariance matrix (P): Uncertainty in state estimate
 * - State transition matrix (F): System dynamics model (for linear KF)
 * - Control matrix (B): Control input model (optional)
 * - Process noise covariance (Q): Model uncertainty
 * - Observation matrix (H): Measurement model (for linear KF)
 * - Measurement noise covariance (R): Sensor uncertainty
 * - Identity matrix: Pre-computed for efficiency
 * - EKF function pointers for non-linear models
 *
 * @warning Do not access structure members directly. Use provided API functions.
 */
typedef struct {
    // State vector (x)
    double *state;

    // Error covariance matrix (P)
    double **errorCovariance;

    // State transition matrix (F)
    double **stateTransition;

    // Control matrix (B)
    double **controlMatrix;

    // Process noise covariance matrix (Q)
    double **processNoise;

    // Observation matrix (H)
    double **observationMatrix;

    // Measurement noise covariance matrix (R)
    double **measurementNoise;

    // Identity matrix for calculations
    double **identity;

    // Dimensions
    size_t stateDim;
    size_t measurementDim;
    size_t controlDim;

    // EKF function pointers
    ekf_state_transition_fn state_transition_func;
    ekf_observation_fn observation_func;
    ekf_state_jacobian_fn state_jacobian_func;
    ekf_observation_jacobian_fn observation_jacobian_func;

    // Initialization flag
    bool initialized;
} kalman_t;

/**
 * @brief Create a new Kalman filter instance
 *
 * Allocates memory and initializes a new Kalman filter with the specified
 * dimensions. All matrices are allocated and initialized to zero, except
 * the identity matrix which is properly initialized.
 *
 * @param stateDimension State vector dimension (must be > 0)
 * @param measurementDimension Measurement vector dimension (must be > 0)
 * @param controlDimension Control vector dimension (0 if no control)
 *
 * @return Pointer to kalman_t structure on success, NULL on failure
 *
 * @note The returned filter must be destroyed with kalman_destroy()
 * @note All matrices are initialized to zero and must be configured
 * @note Control matrix is only allocated if controlDimension > 0
 *
 * @warning Returns NULL if memory allocation fails
 * @warning Input dimensions must be positive (except control can be 0)
 *
 * @see kalman_destroy()
 */
kalman_t* kalman_create(int stateDimension, int measurementDimension, int controlDimension);

/**
 * @brief Destroy a Kalman filter and free all memory
 *
 * Frees all allocated memory for the Kalman filter including all matrices
 * and the structure itself. The pointer becomes invalid after this call.
 *
 * @param kf Pointer to kalman_t structure to destroy
 *
 * @note This function is safe to call with NULL pointer
 * @note All allocated matrices are properly freed
 * @note The structure pointer becomes invalid after this call
 *
 * @warning Do not use the pointer after calling this function
 *
 * @see kalman_create()
 */
void kalman_destroy(kalman_t *kf);

/** @} */ // end of kalman_lifecycle group

/** @defgroup kalman_configuration Kalman Filter Configuration
 *  @brief Functions for setting up filter matrices and parameters
 *  @{
 */

/**
 * Initialize the state vector
 * @param kf Pointer to kalman_t structure
 * @param initialState Initial state values
 * @return 0 on success, -1 on failure
 */
int kalman_initialize_state(kalman_t *kf, const double *initialState);

/**
 * Set the state transition matrix F
 * @param kf Pointer to kalman_t structure
 * @param F State transition matrix (stateDim x stateDim)
 * @return 0 on success, -1 on failure
 */
int kalman_set_state_transition_matrix(kalman_t *kf, double **F);

/**
 * Set the control matrix B
 * @param kf Pointer to kalman_t structure
 * @param B Control matrix (stateDim x controlDim)
 * @return 0 on success, -1 on failure
 */
int kalman_set_control_matrix(kalman_t *kf, double **B);

/**
 * Set the observation matrix H
 * @param kf Pointer to kalman_t structure
 * @param H Observation matrix (measurementDim x stateDim)
 * @return 0 on success, -1 on failure
 */
int kalman_set_observation_matrix(kalman_t *kf, double **H);

/**
 * Set the process noise covariance matrix Q
 * @param kf Pointer to kalman_t structure
 * @param Q Process noise covariance matrix (stateDim x stateDim)
 * @return 0 on success, -1 on failure
 */
int kalman_set_process_noise_covariance(kalman_t *kf, double **Q);

/**
 * Set the measurement noise covariance matrix R
 * @param kf Pointer to kalman_t structure
 * @param R Measurement noise covariance matrix (measurementDim x measurementDim)
 * @return 0 on success, -1 on failure
 */
int kalman_set_measurement_noise_covariance(kalman_t *kf, double **R);

/**
 * Set the error covariance matrix P
 * @param kf Pointer to kalman_t structure
 * @param P Error covariance matrix (stateDim x stateDim)
 * @return 0 on success, -1 on failure
 */
int kalman_set_error_covariance(kalman_t *kf, double **P);

/** @} */ // end of kalman_configuration group


/** @defgroup ekf_configuration EKF Configuration
 *  @brief Functions for setting up EKF non-linear functions and Jacobians
 *  @{
 */

/**
 * @brief Set the non-linear state transition function f(x, u) for EKF
 * @param kf Pointer to kalman_t structure
 * @param f Function pointer to the state transition function
 * @return 0 on success, -1 on failure
 */
int kalman_set_state_transition_function(kalman_t *kf, ekf_state_transition_fn f);

/**
 * @brief Set the non-linear observation function h(x) for EKF
 * @param kf Pointer to kalman_t structure
 * @param h Function pointer to the observation function
 * @return 0 on success, -1 on failure
 */
int kalman_set_observation_function(kalman_t *kf, ekf_observation_fn h);

/**
 * @brief Set the Jacobian of the state transition function for EKF
 * @param kf Pointer to kalman_t structure
 * @param Jf Function pointer to the state Jacobian function
 * @return 0 on success, -1 on failure
 */
int kalman_set_state_jacobian_function(kalman_t *kf, ekf_state_jacobian_fn Jf);

/**
 * @brief Set the Jacobian of the observation function for EKF
 * @param kf Pointer to kalman_t structure
 * @param Jh Function pointer to the observation Jacobian function
 * @return 0 on success, -1 on failure
 */
int kalman_set_observation_jacobian_function(kalman_t *kf, ekf_observation_jacobian_fn Jh);

/** @} */ // end of ekf_configuration group


/** @defgroup kalman_operations Kalman Filter Operations
 *  @brief Core filter operations: prediction and update
 *  @{
 */

/**
 * @brief Perform filter prediction step
 *
 * Executes the prediction phase of the filter algorithm.
 * If EKF functions are set, it performs an EKF prediction. Otherwise, it
 * performs a standard linear KF prediction.
 *
 * - KF:   x(k|k-1) = F * x(k-1|k-1) + B * u(k-1)
 * - KF:   P(k|k-1) = F * P(k-1|k-1) * F^T + Q
 * - EKF:  x(k|k-1) = f(x(k-1|k-1), u(k-1))
 * - EKF:  P(k|k-1) = Jf * P(k-1|k-1) * Jf^T + Q
 *
 * @param kf Pointer to initialized kalman_t structure
 * @param control Control vector (can be NULL if no control input)
 *
 * @return 0 on success, -1 on failure
 *
 * @pre Filter must be initialized with kalman_set_error_covariance()
 * @pre For KF, required matrices (F, Q) must be set.
 * @pre For EKF, required functions (f, Jf) and matrix (Q) must be set.
 * @pre If control is provided, control matrix B (for KF) must be set.
 *
 * @note Control parameter can be NULL if no control input is used
 *
 * @warning Returns -1 if filter is not initialized or required components are not set.
 *
 * @see kalman_update()
 * @see kalman_set_state_transition_matrix()
 * @see kalman_set_state_transition_function()
 */
int kalman_predict(kalman_t *kf, const double *control);

/**
 * @brief Perform filter update step (correction)
 *
 * Executes the update (correction) phase of the filter algorithm.
 * If EKF functions are set, it performs an EKF update. Otherwise, it
 * performs a standard linear KF update.
 *
 * - Innovation (KF):   y = z - H * x(k|k-1)
 * - Innovation (EKF):  y = z - h(x(k|k-1))
 * - Kalman gain:       K = P * H^T * (H * P * H^T + R)^(-1) (H is Hj for EKF)
 * - State update:      x(k|k) = x(k|k-1) + K * y
 * - Covariance update: P(k|k) = (I - K * H) * P(k|k-1) (H is Hj for EKF)
 *
 * @param kf Pointer to initialized kalman_t structure
 * @param measurement Measurement vector (must not be NULL)
 *
 * @return 0 on success, -1 on failure
 *
 * @pre Filter must be initialized with kalman_set_error_covariance()
 * @pre For KF, required matrices (H, R) must be set.
 * @pre For EKF, required functions (h, Jh) and matrix (R) must be set.
 * @pre Measurement vector must have correct dimension.
 *
 * @note Matrix inversion uses Gaussian elimination with partial pivoting.
 *
 * @warning Returns -1 if filter is not initialized, measurement is NULL,
 *          required components are not set, or matrix inversion fails.
 *
 * @see kalman_predict()
 * @see kalman_set_observation_matrix()
 * @see kalman_set_observation_function()
 */
int kalman_update(kalman_t *kf, const double *measurement);

/** @} */ // end of kalman_operations group

/** @defgroup kalman_accessors Kalman Filter Accessors
 *  @brief Functions for accessing filter state and properties
 *  @{
 */

/**
 * Get the current state vector
 * @param kf Pointer to kalman_t structure
 * @return Pointer to state vector or NULL on failure
 */
double* kalman_get_state(kalman_t *kf);

/**
 * Get the current error covariance matrix
 * @param kf Pointer to kalman_t structure
 * @return Pointer to error covariance matrix or NULL on failure
 */
double** kalman_get_error_covariance(kalman_t *kf);

/**
 * Get a specific state element
 * @param kf Pointer to kalman_t structure
 * @param index State element index
 * @return State element value or 0.0 on failure
 */
double kalman_get_state_element(const kalman_t *kf, int index);

/**
 * Reset the Kalman filter
 * @param kf Pointer to kalman_t structure
 */
void kalman_reset(kalman_t *kf);

/** @} */ // end of kalman_utility group

/**
 * Check if the Kalman filter is initialized
 * @param kf Pointer to kalman_t structure
 * @return true if initialized, false otherwise
 */
bool kalman_is_initialized(const kalman_t *kf);

/** @} */ // end of kalman_accessors group

/** @defgroup kalman_utility Kalman Filter Utilities
 *  @brief Utility functions for filter management
 *  @{
 */

#ifdef __cplusplus
}
#endif

#endif //KALMAN_H
