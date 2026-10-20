/**
 * @file    kalman_ekf_tests.c
 * @brief   Unit tests for the Extended Kalman filter mode of the kalman module.
 */

#include "kalman.h"

#include "test_support.h"
#include "unity.h"

#include <math.h>

static const double tolerance = 1e-10;

// f(x) = (x0^2, x1)
static void square_first_state(const double *state, const double *control, double *next_state) {
    (void)control;
    next_state[0] = state[0] * state[0];
    next_state[1] = state[1];
}

static void square_first_state_jacobian(const double *state, matrix_t *jacobian) {
    jacobian->data[0][0] = 2.0 * state[0];
    jacobian->data[0][1] = 0.0;
    jacobian->data[1][0] = 0.0;
    jacobian->data[1][1] = 1.0;
}

// f(x) = (x1, x0); written so that it only works when `next_state` is not the same array as `state`
static void swap_states(const double *state, const double *control, double *next_state) {
    (void)control;
    next_state[0] = state[1];
    next_state[1] = state[0];
}

static void swap_states_jacobian(const double *state, matrix_t *jacobian) {
    (void)state;
    jacobian->data[0][0] = 0.0;
    jacobian->data[0][1] = 1.0;
    jacobian->data[1][0] = 1.0;
    jacobian->data[1][1] = 0.0;
}

// h(x) = distance from the origin
static void distance_from_origin(const double *state, double *measurement) {
    measurement[0] = sqrt(state[0] * state[0] + state[1] * state[1]);
}

static void distance_from_origin_jacobian(const double *state, matrix_t *jacobian) {
    const double distance = sqrt(state[0] * state[0] + state[1] * state[1]);
    jacobian->data[0][0] = state[0] / distance;
    jacobian->data[0][1] = state[1] / distance;
}

// Two states and one measurement, identity covariance, and process and measurement noise as given
static void start_filter(kalman_t *kf, const double process_noise, const double measurement_noise,
                         const double *initial_state) {
    const matrix_t covariance = test_support_diagonal(2, 1.0);
    const matrix_t q = test_support_diagonal(2, process_noise);
    const matrix_t r = test_support_diagonal(1, measurement_noise);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_init(kf, 2, 1, 0));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_initialize_state(kf, initial_state));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_process_noise_covariance(kf, &q));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_measurement_noise_covariance(kf, &r));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_error_covariance(kf, &covariance));
}

static void test_predicting_applies_the_non_linear_state_transition(void) {
    kalman_t kf;
    const double initial_state[2] = {1.5, 1.0};
    start_filter(&kf, 0.1, 0.2, initial_state);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_transition_function(&kf, square_first_state));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_jacobian_function(&kf, square_first_state_jacobian));

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_predict(&kf, NULL));

    double state[2];
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_get_state(&kf, state));
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, 2.25, state[0]);
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, 1.0, state[1]);
}

static void test_predicting_linearizes_around_the_state_the_step_starts_from(void) {
    kalman_t kf;
    const double initial_state[2] = {2.0, 1.0};
    start_filter(&kf, 0.1, 0.2, initial_state);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_transition_function(&kf, square_first_state));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_jacobian_function(&kf, square_first_state_jacobian));

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_predict(&kf, NULL));

    // The Jacobian at x0 = 2 is 4, so P00 = 4 * 1 * 4 + 0.1; evaluating it at the predicted x0 = 4 would give 64.1
    matrix_t covariance;
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_get_error_covariance(&kf, &covariance));
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, 16.1, covariance.data[0][0]);
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, 1.1, covariance.data[1][1]);
}

static void test_the_state_transition_may_read_the_state_while_writing_the_next_one(void) {
    kalman_t kf;
    const double initial_state[2] = {1.0, 2.0};
    start_filter(&kf, 0.0, 1.0, initial_state);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_transition_function(&kf, swap_states));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_jacobian_function(&kf, swap_states_jacobian));

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_predict(&kf, NULL));

    TEST_ASSERT_EQUAL_DOUBLE(2.0, kf.state[0]);
    TEST_ASSERT_EQUAL_DOUBLE(1.0, kf.state[1]);
}

static void test_updating_moves_the_state_along_the_gradient_of_the_observation(void) {
    kalman_t kf;
    const double initial_state[2] = {3.0, 4.0};
    start_filter(&kf, 0.0, 1.0, initial_state);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_observation_function(&kf, distance_from_origin));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok,
                          kalman_set_observation_jacobian_function(&kf, distance_from_origin_jacobian));
    const double measurement[1] = {5.5};

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_update(&kf, measurement));

    // The true distance is 5, so the innovation is 0.5. With H = (0.6, 0.8), P = I and R = 1, S = 2 and the gain is
    // (0.3, 0.4); the state moves by gain * innovation
    double state[2];
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_get_state(&kf, state));
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, 3.15, state[0]);
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, 4.2, state[1]);
}

static void test_the_linear_matrices_are_not_overwritten_by_the_jacobians(void) {
    kalman_t kf;
    const double initial_state[2] = {3.0, 4.0};
    start_filter(&kf, 0.0, 1.0, initial_state);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_transition_function(&kf, square_first_state));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_jacobian_function(&kf, square_first_state_jacobian));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_observation_function(&kf, distance_from_origin));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok,
                          kalman_set_observation_jacobian_function(&kf, distance_from_origin_jacobian));
    const double measurement[1] = {10.0};

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_predict(&kf, NULL));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_update(&kf, measurement));

    TEST_ASSERT_EQUAL_DOUBLE(1.0, kf.state_transition.data[0][0]);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, kf.observation_matrix.data[0][0]);
}

static void test_a_function_without_its_jacobian_is_an_incomplete_model(void) {
    kalman_t kf;
    const double initial_state[2] = {1.0, 1.0};
    const double measurement[1] = {1.0};
    start_filter(&kf, 0.1, 0.2, initial_state);

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_transition_function(&kf, square_first_state));
    TEST_ASSERT_EQUAL_INT(kalman_status_incomplete_model, kalman_predict(&kf, NULL));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok,
                          kalman_set_observation_jacobian_function(&kf, distance_from_origin_jacobian));
    TEST_ASSERT_EQUAL_INT(kalman_status_incomplete_model, kalman_update(&kf, measurement));
}

static void test_a_failed_update_leaves_the_extended_filter_unchanged(void) {
    kalman_t kf;
    const double initial_state[2] = {3.0, 4.0};
    start_filter(&kf, 0.0, 0.0, initial_state);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_observation_function(&kf, distance_from_origin));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok,
                          kalman_set_observation_jacobian_function(&kf, distance_from_origin_jacobian));
    // With no uncertainty at all the innovation covariance is zero
    const matrix_t no_uncertainty = test_support_diagonal(2, 0.0);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_error_covariance(&kf, &no_uncertainty));
    const double measurement[1] = {6.0};

    TEST_ASSERT_EQUAL_INT(kalman_status_singular_innovation, kalman_update(&kf, measurement));

    TEST_ASSERT_EQUAL_DOUBLE(3.0, kf.state[0]);
    TEST_ASSERT_EQUAL_DOUBLE(4.0, kf.state[1]);
}

static void test_clearing_the_functions_goes_back_to_the_linear_model(void) {
    kalman_t kf;
    const double initial_state[2] = {2.0, 1.0};
    start_filter(&kf, 0.0, 1.0, initial_state);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_transition_function(&kf, square_first_state));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_jacobian_function(&kf, square_first_state_jacobian));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_transition_function(&kf, NULL));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_jacobian_function(&kf, NULL));

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_predict(&kf, NULL));

    // The default state transition is the identity
    TEST_ASSERT_EQUAL_DOUBLE(2.0, kf.state[0]);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_predicting_applies_the_non_linear_state_transition);
    RUN_TEST(test_predicting_linearizes_around_the_state_the_step_starts_from);
    RUN_TEST(test_the_state_transition_may_read_the_state_while_writing_the_next_one);
    RUN_TEST(test_updating_moves_the_state_along_the_gradient_of_the_observation);
    RUN_TEST(test_the_linear_matrices_are_not_overwritten_by_the_jacobians);
    RUN_TEST(test_a_function_without_its_jacobian_is_an_incomplete_model);
    RUN_TEST(test_a_failed_update_leaves_the_extended_filter_unchanged);
    RUN_TEST(test_clearing_the_functions_goes_back_to_the_linear_model);
    return UNITY_END();
}
