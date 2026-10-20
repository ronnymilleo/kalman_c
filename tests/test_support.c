/**
 * @file    test_support.c
 * @brief   Helpers shared by the unit tests: a repeatable noise source and ready-made filter models.
 */

#include "test_support.h"

#include "unity.h"

// Any non-zero value works; xorshift generators get stuck at zero
static const uint64_t fallback_seed = 0x9E3779B97F4A7C15ULL;
static const uint64_t xorshift_multiplier = 2685821657736338717ULL;
static const double unit_interval_steps = 9007199254740992.0; // 2^53, the doubles that fit a 53-bit mantissa

// Unity calls these around every test; the tests here build their own fixtures
void setUp(void) {
}

void tearDown(void) {
}

/**
 * @brief   Seeds the generator.
 * @param[out] random  Generator to seed.
 * @param[in]  seed    Starting point; the same seed always gives the same sequence.
 */
void test_support_random_seed(test_support_random_t *random, const uint64_t seed) {
    random->state = seed != 0 ? seed : fallback_seed;
}

/**
 * @brief   Draws a number uniformly distributed around zero.
 * @param[in,out] random     Generator to draw from.
 * @param[in]     amplitude  Half the width of the interval.
 * @return  A value in `[-amplitude, amplitude)`.
 */
double test_support_random_symmetric(test_support_random_t *random, const double amplitude) {
    uint64_t state = random->state;
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    random->state = state;

    const uint64_t mixed = (state * xorshift_multiplier) >> 11;
    const double unit = (double)mixed / unit_interval_steps;
    return (2.0 * unit - 1.0) * amplitude;
}

/**
 * @brief   Builds a matrix from row-major values, failing the running test when the size is invalid.
 * @param[in] rows    Number of rows.
 * @param[in] cols    Number of columns.
 * @param[in] values  `rows * cols` values in row-major order.
 * @return  The matrix.
 */
matrix_t test_support_matrix(const size_t rows, const size_t cols, const double *values) {
    matrix_t matrix;
    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_from_values(&matrix, rows, cols, values));
    return matrix;
}

/**
 * @brief   Builds a square matrix with the same value along the diagonal and zeros elsewhere.
 * @param[in] size   Number of rows and columns.
 * @param[in] value  Value of the diagonal.
 * @return  The matrix.
 */
matrix_t test_support_diagonal(const size_t size, const double value) {
    matrix_t matrix;
    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_identity(&matrix, size));
    for (size_t index = 0; index < size; index++) {
        matrix.data[index][index] = value;
    }
    return matrix;
}

/**
 * @brief   Describes a position and velocity tracker that measures the position only.
 * @details The process noise starts as a small diagonal; tests that care about it replace `q`.
 * @param[in] dt                    Time between steps.
 * @param[in] measurement_variance  Variance of the position measurement.
 * @param[in] initial_variance      Starting variance of both states; the starting state is zero.
 * @return  The model.
 */
test_support_model_t test_support_model_1d(const double dt, const double measurement_variance,
                                           const double initial_variance) {
    return (test_support_model_t){
        .state_dim = 2,
        .measurement_dim = 1,
        .f = test_support_matrix(2, 2, (const double[]){1.0, dt, 0.0, 1.0}),
        .h = test_support_matrix(1, 2, (const double[]){1.0, 0.0}),
        .q = test_support_diagonal(2, 0.01),
        .r = test_support_diagonal(1, measurement_variance),
        .p = test_support_diagonal(2, initial_variance),
    };
}

/**
 * @brief   Describes a tracker of position and velocity on two axes that measures both positions.
 * @details The states are `x`, `y`, `vx` and `vy`, and the starting state is zero.
 * @param[in] dt                    Time between steps.
 * @param[in] process_variance      Variance of the velocity noise on each axis.
 * @param[in] measurement_variance  Variance of each position measurement.
 * @param[in] initial_variance      Starting variance of all four states.
 * @return  The model.
 */
test_support_model_t test_support_model_2d(const double dt, const double process_variance,
                                           const double measurement_variance, const double initial_variance) {
    // clang-format off
    const double f[] = {1.0, 0.0, dt,  0.0,
                        0.0, 1.0, 0.0, dt,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0};
    const double q[] = {process_variance * dt * dt, 0.0, process_variance * dt, 0.0,
                        0.0, process_variance * dt * dt, 0.0, process_variance * dt,
                        process_variance * dt, 0.0, process_variance, 0.0,
                        0.0, process_variance * dt, 0.0, process_variance};
    // clang-format on
    return (test_support_model_t){
        .state_dim = 4,
        .measurement_dim = 2,
        .f = test_support_matrix(4, 4, f),
        .h = test_support_matrix(2, 4, (const double[]){1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}),
        .q = test_support_matrix(4, 4, q),
        .r = test_support_diagonal(2, measurement_variance),
        .p = test_support_diagonal(4, initial_variance),
    };
}

/**
 * @brief   Builds the process noise of a position and velocity pair driven by random velocity changes.
 * @param[in] dt                   Time between steps.
 * @param[in] process_noise_sigma  Standard deviation of the velocity changes.
 * @return  The 2x2 covariance.
 */
matrix_t test_support_position_velocity_noise(const double dt, const double process_noise_sigma) {
    const double variance = process_noise_sigma * process_noise_sigma;
    return test_support_matrix(2, 2, (const double[]){variance * dt * dt, variance * dt, variance * dt, variance});
}

/**
 * @brief   Sets a filter up with a model and leaves it ready to predict, failing the running test on any error.
 * @param[out] kf     Filter to set up.
 * @param[in]  model  Model to apply.
 */
void test_support_start_filter(kalman_t *kf, const test_support_model_t *model) {
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_init(kf, model->state_dim, model->measurement_dim, 0));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_initialize_state(kf, model->initial_state));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_transition_matrix(kf, &model->f));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_observation_matrix(kf, &model->h));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_process_noise_covariance(kf, &model->q));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_measurement_noise_covariance(kf, &model->r));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_error_covariance(kf, &model->p));
}
