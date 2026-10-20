/**
 * @file    kalman_tests.c
 * @brief   Unit tests for the linear Kalman filter in the kalman module.
 */

#include "kalman.h"

#include "test_support.h"
#include "unity.h"

static const double tolerance = 1e-10;
static const double dt = 0.1;

// Position and velocity tracker with the state (1.0, 0.5), the measurement noise 0.1 and an identity covariance
static void start_position_velocity_filter(kalman_t *kf) {
    test_support_model_t model = test_support_model_1d(dt, 0.1, 1.0);
    model.initial_state[0] = 1.0;
    model.initial_state[1] = 0.5;
    test_support_start_filter(kf, &model);
}

static void test_a_filter_needs_at_least_one_state_and_one_measurement(void) {
    kalman_t kf;

    TEST_ASSERT_EQUAL_INT(kalman_status_invalid_dimension, kalman_init(&kf, 0, 1, 0));
    TEST_ASSERT_EQUAL_INT(kalman_status_invalid_dimension, kalman_init(&kf, 1, 0, 0));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_init(&kf, 1, 1, 0));
}

static void test_a_filter_cannot_exceed_the_capacity(void) {
    kalman_t kf;

    TEST_ASSERT_EQUAL_INT(kalman_status_invalid_dimension, kalman_init(&kf, KALMAN_MAX_DIM + 1, 1, 0));
    TEST_ASSERT_EQUAL_INT(kalman_status_invalid_dimension, kalman_init(&kf, 1, KALMAN_MAX_DIM + 1, 0));
    TEST_ASSERT_EQUAL_INT(kalman_status_invalid_dimension, kalman_init(&kf, 1, 1, KALMAN_MAX_DIM + 1));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_init(&kf, KALMAN_MAX_DIM, KALMAN_MAX_DIM, KALMAN_MAX_DIM));
}

static void test_a_new_filter_has_a_zero_state_and_an_identity_state_transition(void) {
    kalman_t kf;

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_init(&kf, 3, 2, 1));

    double state[3];
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_get_state(&kf, state));
    TEST_ASSERT_EQUAL_DOUBLE(0.0, state[0]);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, state[2]);
    TEST_ASSERT_EQUAL_DOUBLE(1.0, kf.state_transition.data[1][1]);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, kf.state_transition.data[1][2]);
}

static void test_functions_reject_a_null_filter(void) {
    const matrix_t matrix = test_support_diagonal(2, 1.0);
    const double values[2] = {0.0, 0.0};

    TEST_ASSERT_EQUAL_INT(kalman_status_null_argument, kalman_init(NULL, 2, 1, 0));
    TEST_ASSERT_EQUAL_INT(kalman_status_null_argument, kalman_reset(NULL));
    TEST_ASSERT_EQUAL_INT(kalman_status_null_argument, kalman_initialize_state(NULL, values));
    TEST_ASSERT_EQUAL_INT(kalman_status_null_argument, kalman_set_state_transition_matrix(NULL, &matrix));
    TEST_ASSERT_EQUAL_INT(kalman_status_null_argument, kalman_set_error_covariance(NULL, &matrix));
    TEST_ASSERT_EQUAL_INT(kalman_status_null_argument, kalman_predict(NULL, NULL));
    TEST_ASSERT_EQUAL_INT(kalman_status_null_argument, kalman_update(NULL, values));
    TEST_ASSERT_FALSE(kalman_is_initialized(NULL));
    kalman_deinit(NULL);
}

static void test_functions_reject_null_data(void) {
    kalman_t kf;
    start_position_velocity_filter(&kf);

    TEST_ASSERT_EQUAL_INT(kalman_status_null_argument, kalman_initialize_state(&kf, NULL));
    TEST_ASSERT_EQUAL_INT(kalman_status_null_argument, kalman_set_state_transition_matrix(&kf, NULL));
    TEST_ASSERT_EQUAL_INT(kalman_status_null_argument, kalman_update(&kf, NULL));
    TEST_ASSERT_EQUAL_INT(kalman_status_null_argument, kalman_get_state(&kf, NULL));
}

static void test_the_filter_is_ready_only_once_the_error_covariance_is_set(void) {
    kalman_t kf;
    const double measurement[1] = {1.0};
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_init(&kf, 2, 1, 0));

    TEST_ASSERT_FALSE(kalman_is_initialized(&kf));
    TEST_ASSERT_EQUAL_INT(kalman_status_not_initialized, kalman_predict(&kf, NULL));
    TEST_ASSERT_EQUAL_INT(kalman_status_not_initialized, kalman_update(&kf, measurement));

    const matrix_t covariance = test_support_diagonal(2, 1.0);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_error_covariance(&kf, &covariance));
    TEST_ASSERT_TRUE(kalman_is_initialized(&kf));
}

static void test_matrices_of_the_wrong_size_are_rejected(void) {
    kalman_t kf;
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_init(&kf, 2, 1, 0));
    const matrix_t wrong = test_support_diagonal(3, 1.0);

    TEST_ASSERT_EQUAL_INT(kalman_status_dimension_mismatch, kalman_set_state_transition_matrix(&kf, &wrong));
    TEST_ASSERT_EQUAL_INT(kalman_status_dimension_mismatch, kalman_set_observation_matrix(&kf, &wrong));
    TEST_ASSERT_EQUAL_INT(kalman_status_dimension_mismatch, kalman_set_process_noise_covariance(&kf, &wrong));
    TEST_ASSERT_EQUAL_INT(kalman_status_dimension_mismatch, kalman_set_measurement_noise_covariance(&kf, &wrong));
    TEST_ASSERT_EQUAL_INT(kalman_status_dimension_mismatch, kalman_set_error_covariance(&kf, &wrong));
    TEST_ASSERT_FALSE(kalman_is_initialized(&kf));
}

static void test_a_control_matrix_needs_a_filter_with_control_inputs(void) {
    kalman_t without_control;
    kalman_t with_control;
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_init(&without_control, 2, 1, 0));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_init(&with_control, 2, 1, 1));
    const matrix_t control_matrix = test_support_matrix(2, 1, (const double[]){0.5, 1.0});

    TEST_ASSERT_EQUAL_INT(kalman_status_dimension_mismatch,
                          kalman_set_control_matrix(&without_control, &control_matrix));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_control_matrix(&with_control, &control_matrix));
}

static void test_the_state_can_be_set_and_read_back(void) {
    kalman_t kf;
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_init(&kf, 3, 1, 0));
    const double initial_state[3] = {1.0, 2.0, 3.0};

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_initialize_state(&kf, initial_state));

    double state[3];
    double element = 0.0;
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_get_state(&kf, state));
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(initial_state, state, 3);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_get_state_element(&kf, 2, &element));
    TEST_ASSERT_EQUAL_DOUBLE(3.0, element);
}

static void test_a_state_element_past_the_end_cannot_be_read(void) {
    kalman_t kf;
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_init(&kf, 2, 1, 0));
    double element = 123.0;

    TEST_ASSERT_EQUAL_INT(kalman_status_index_out_of_range, kalman_get_state_element(&kf, 2, &element));
    TEST_ASSERT_EQUAL_DOUBLE(123.0, element);
}

static void test_predicting_moves_the_position_by_the_velocity_times_the_time_step(void) {
    kalman_t kf;
    start_position_velocity_filter(&kf);

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_predict(&kf, NULL));

    double position = 0.0;
    double velocity = 0.0;
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_get_state_element(&kf, 0, &position));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_get_state_element(&kf, 1, &velocity));
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, 1.0 + 0.5 * dt, position);
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, 0.5, velocity);
}

static void test_predicting_grows_the_uncertainty_by_f_p_f_transposed_plus_q(void) {
    kalman_t kf;
    start_position_velocity_filter(&kf);

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_predict(&kf, NULL));

    // With P = I, Q = 0.01 I and F = [[1, dt], [0, 1]], F P F^T + Q = [[1 + dt^2 + 0.01, dt], [dt, 1.01]]
    matrix_t covariance;
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_get_error_covariance(&kf, &covariance));
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, 1.0 + dt * dt + 0.01, covariance.data[0][0]);
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, dt, covariance.data[0][1]);
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, dt, covariance.data[1][0]);
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, 1.01, covariance.data[1][1]);
}

static void test_predicting_adds_the_effect_of_the_control_input(void) {
    kalman_t kf;
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_init(&kf, 2, 1, 1));
    const double initial_state[2] = {0.0, 1.0};
    const matrix_t control_matrix = test_support_matrix(2, 1, (const double[]){0.5 * dt * dt, dt});
    const matrix_t covariance = test_support_diagonal(2, 1.0);
    const matrix_t f = test_support_matrix(2, 2, (const double[]){1.0, dt, 0.0, 1.0});
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_initialize_state(&kf, initial_state));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_transition_matrix(&kf, &f));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_control_matrix(&kf, &control_matrix));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_error_covariance(&kf, &covariance));
    const double acceleration[1] = {2.0};

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_predict(&kf, acceleration));

    // Constant acceleration of 2 for dt: the velocity grows by 2 dt and the position by v dt + dt^2
    double state[2];
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_get_state(&kf, state));
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, dt + dt * dt, state[0]);
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, 1.0 + 2.0 * dt, state[1]);
}

static void test_updating_with_equal_trust_lands_halfway_to_the_measurement(void) {
    kalman_t kf;
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_init(&kf, 1, 1, 0));
    const matrix_t one = test_support_diagonal(1, 1.0);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_observation_matrix(&kf, &one));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_measurement_noise_covariance(&kf, &one));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_error_covariance(&kf, &one));
    const double measurement[1] = {2.0};

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_update(&kf, measurement));

    // With P = R = 1 the gain is P / (P + R) = 0.5, so the estimate moves from 0 to 1 and the variance halves
    double estimate = 0.0;
    matrix_t covariance;
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_get_state_element(&kf, 0, &estimate));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_get_error_covariance(&kf, &covariance));
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, 1.0, estimate);
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, 0.5, covariance.data[0][0]);
}

static void test_an_update_lands_between_the_prediction_and_the_measurement(void) {
    kalman_t kf;
    start_position_velocity_filter(&kf);
    const double measurement[1] = {1.06};
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_predict(&kf, NULL));

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_update(&kf, measurement));

    double position = 0.0;
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_get_state_element(&kf, 0, &position));
    TEST_ASSERT_TRUE(position > 1.0 + 0.5 * dt);
    TEST_ASSERT_TRUE(position < 1.06);
}

static void test_an_update_with_a_singular_innovation_fails_and_changes_nothing(void) {
    kalman_t kf;
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_init(&kf, 1, 1, 0));
    const matrix_t zero_observation = test_support_matrix(1, 1, (const double[]){0.0});
    const matrix_t covariance = test_support_diagonal(1, 1.0);
    const double initial_state[1] = {5.0};
    // H = 0 and R = 0 leave S = 0, which cannot be inverted
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_observation_matrix(&kf, &zero_observation));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_error_covariance(&kf, &covariance));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_initialize_state(&kf, initial_state));
    const double measurement[1] = {9.0};

    TEST_ASSERT_EQUAL_INT(kalman_status_singular_innovation, kalman_update(&kf, measurement));

    TEST_ASSERT_EQUAL_DOUBLE(5.0, kf.state[0]);
    TEST_ASSERT_EQUAL_DOUBLE(1.0, kf.error_covariance.data[0][0]);
}

static void test_uncertainty_shrinks_with_every_update_and_grows_with_every_prediction(void) {
    kalman_t kf;
    start_position_velocity_filter(&kf);
    const double measurement[1] = {1.0};
    matrix_t covariance;

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_update(&kf, measurement));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_get_error_covariance(&kf, &covariance));
    const double after_update = covariance.data[0][0];
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_predict(&kf, NULL));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_get_error_covariance(&kf, &covariance));

    TEST_ASSERT_TRUE(after_update < 1.0);
    TEST_ASSERT_TRUE(covariance.data[0][0] > after_update);
}

static void test_resetting_clears_the_estimate_but_keeps_the_model(void) {
    kalman_t kf;
    start_position_velocity_filter(&kf);

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_reset(&kf));

    TEST_ASSERT_FALSE(kalman_is_initialized(&kf));
    TEST_ASSERT_EQUAL_DOUBLE(0.0, kf.state[0]);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, kf.error_covariance.data[0][0]);
    TEST_ASSERT_EQUAL_DOUBLE(dt, kf.state_transition.data[0][1]);
    TEST_ASSERT_EQUAL_UINT(2, kf.error_covariance.rows);

    // The same filter starts again once its covariance is set
    const matrix_t covariance = test_support_diagonal(2, 1.0);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_error_covariance(&kf, &covariance));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_predict(&kf, NULL));
}

static void test_deinitializing_clears_the_filter(void) {
    kalman_t kf;
    start_position_velocity_filter(&kf);

    kalman_deinit(&kf);

    TEST_ASSERT_FALSE(kalman_is_initialized(&kf));
    TEST_ASSERT_EQUAL_UINT(0, kf.state_dim);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_a_filter_needs_at_least_one_state_and_one_measurement);
    RUN_TEST(test_a_filter_cannot_exceed_the_capacity);
    RUN_TEST(test_a_new_filter_has_a_zero_state_and_an_identity_state_transition);
    RUN_TEST(test_functions_reject_a_null_filter);
    RUN_TEST(test_functions_reject_null_data);
    RUN_TEST(test_the_filter_is_ready_only_once_the_error_covariance_is_set);
    RUN_TEST(test_matrices_of_the_wrong_size_are_rejected);
    RUN_TEST(test_a_control_matrix_needs_a_filter_with_control_inputs);
    RUN_TEST(test_the_state_can_be_set_and_read_back);
    RUN_TEST(test_a_state_element_past_the_end_cannot_be_read);
    RUN_TEST(test_predicting_moves_the_position_by_the_velocity_times_the_time_step);
    RUN_TEST(test_predicting_grows_the_uncertainty_by_f_p_f_transposed_plus_q);
    RUN_TEST(test_predicting_adds_the_effect_of_the_control_input);
    RUN_TEST(test_updating_with_equal_trust_lands_halfway_to_the_measurement);
    RUN_TEST(test_an_update_lands_between_the_prediction_and_the_measurement);
    RUN_TEST(test_an_update_with_a_singular_innovation_fails_and_changes_nothing);
    RUN_TEST(test_uncertainty_shrinks_with_every_update_and_grows_with_every_prediction);
    RUN_TEST(test_resetting_clears_the_estimate_but_keeps_the_model);
    RUN_TEST(test_deinitializing_clears_the_filter);
    return UNITY_END();
}
