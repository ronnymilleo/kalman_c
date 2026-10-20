/**
 * @file    kalman_wrap_tests.c
 * @brief   Unit tests for filters whose measurements or states are angles: the residual and state normalizer hooks.
 */

#include "kalman.h"

#include "angle.h"
#include "test_support.h"
#include "unity.h"

#include <math.h>

static const double tolerance = 1e-9;

static void wrapped_residual(const double *measurement, const double *expected_measurement, double *residual) {
    residual[0] = angle_difference_radians(measurement[0], expected_measurement[0]);
}

static void wrap_heading(double *state) {
    state[0] = angle_wrap_radians(state[0]);
}

// A heading and its turn rate, with the heading measured: the constant turn rate model
static test_support_model_t heading_model(const double dt, const double measurement_variance) {
    test_support_model_t model = test_support_model_1d(dt, measurement_variance, 1.0);
    model.q = test_support_diagonal(2, 1e-4);
    return model;
}

// Tracks a heading that turns at a constant rate through the boundary at pi; returns the final heading error
static double track_heading(kalman_t *kf, const double dt, const double turn_rate, const double start_heading,
                            const int steps, const double noise_amplitude) {
    test_support_random_t random;
    test_support_random_seed(&random, 11);

    double true_heading = start_heading;
    for (int step = 0; step < steps; step++) {
        true_heading += turn_rate * dt;
        const double measurement[1] = {angle_wrap_radians(true_heading) +
                                       test_support_random_symmetric(&random, noise_amplitude)};
        TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_predict(kf, NULL));
        TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_update(kf, measurement));
    }
    return fabs(angle_difference_radians(kf->state[0], true_heading));
}

static void test_the_residual_function_replaces_the_subtraction_of_the_innovation(void) {
    kalman_t kf;
    test_support_model_t model = heading_model(0.1, 0.01);
    model.initial_state[0] = ANGLE_PI - 0.01;
    test_support_start_filter(&kf, &model);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_residual_function(&kf, wrapped_residual));
    const double measurement[1] = {-ANGLE_PI + 0.01};

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_update(&kf, measurement));

    // The two headings are 0.02 apart across the boundary, so the estimate moves a little, not by a whole turn
    TEST_ASSERT_TRUE(fabs(angle_difference_radians(kf.state[0], ANGLE_PI)) < 0.05);
}

static void test_without_the_residual_function_a_heading_across_the_boundary_throws_the_estimate_off(void) {
    kalman_t kf;
    test_support_model_t model = heading_model(0.1, 0.01);
    model.initial_state[0] = ANGLE_PI - 0.01;
    test_support_start_filter(&kf, &model);
    const double measurement[1] = {-ANGLE_PI + 0.01};

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_update(&kf, measurement));

    // The innovation is almost a whole turn, and the estimate follows most of it instead of moving 0.02
    TEST_ASSERT_TRUE(fabs(kf.state[0] - (ANGLE_PI - 0.01)) > 3.0);
}

static void test_the_state_normalizer_adjusts_the_state_after_a_prediction(void) {
    kalman_t kf;
    test_support_model_t model = heading_model(1.0, 0.01);
    model.initial_state[0] = ANGLE_PI - 0.1;
    model.initial_state[1] = 0.3;
    test_support_start_filter(&kf, &model);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_normalizer(&kf, wrap_heading));

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_predict(&kf, NULL));

    // The heading would be pi + 0.2, which is the same direction as -pi + 0.2
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, -ANGLE_PI + 0.2, kf.state[0]);
}

static void test_the_state_normalizer_adjusts_the_state_after_an_update(void) {
    kalman_t kf;
    test_support_model_t model = heading_model(0.1, 1e-6);
    model.initial_state[0] = ANGLE_PI - 0.001;
    test_support_start_filter(&kf, &model);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_normalizer(&kf, wrap_heading));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_residual_function(&kf, wrapped_residual));
    const double measurement[1] = {-ANGLE_PI + 0.001};

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_update(&kf, measurement));

    TEST_ASSERT_TRUE(kf.state[0] < 0.0);
    TEST_ASSERT_TRUE(kf.state[0] >= -ANGLE_PI);
}

static void test_a_filter_with_both_hooks_follows_a_heading_through_the_boundary(void) {
    const double dt = 0.1;
    kalman_t kf;
    test_support_model_t model = heading_model(dt, 0.01);
    model.initial_state[0] = 2.5;
    test_support_start_filter(&kf, &model);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_residual_function(&kf, wrapped_residual));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_normalizer(&kf, wrap_heading));

    const double error = track_heading(&kf, dt, 1.0, 2.5, 100, 0.1);

    TEST_ASSERT_TRUE(error < 0.1);
    TEST_ASSERT_TRUE(fabs(kf.state[1] - 1.0) < 0.1);
}

static void test_a_filter_without_the_hooks_loses_a_heading_that_crosses_the_boundary(void) {
    const double dt = 0.1;
    kalman_t kf;
    test_support_model_t model = heading_model(dt, 0.01);
    model.initial_state[0] = 2.5;
    test_support_start_filter(&kf, &model);

    const double error = track_heading(&kf, dt, 1.0, 2.5, 100, 0.1);

    TEST_ASSERT_TRUE(error > 0.5);
}

static void test_the_hooks_can_be_cleared(void) {
    kalman_t kf;
    test_support_model_t model = heading_model(0.1, 0.01);
    test_support_start_filter(&kf, &model);
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_residual_function(&kf, wrapped_residual));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_normalizer(&kf, wrap_heading));

    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_residual_function(&kf, NULL));
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_set_state_normalizer(&kf, NULL));

    TEST_ASSERT_NULL(kf.residual_function);
    TEST_ASSERT_NULL(kf.state_normalizer);
}

static void test_the_hook_setters_reject_a_null_filter(void) {
    TEST_ASSERT_EQUAL_INT(kalman_status_null_argument, kalman_set_residual_function(NULL, wrapped_residual));
    TEST_ASSERT_EQUAL_INT(kalman_status_null_argument, kalman_set_state_normalizer(NULL, wrap_heading));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_the_residual_function_replaces_the_subtraction_of_the_innovation);
    RUN_TEST(test_without_the_residual_function_a_heading_across_the_boundary_throws_the_estimate_off);
    RUN_TEST(test_the_state_normalizer_adjusts_the_state_after_a_prediction);
    RUN_TEST(test_the_state_normalizer_adjusts_the_state_after_an_update);
    RUN_TEST(test_a_filter_with_both_hooks_follows_a_heading_through_the_boundary);
    RUN_TEST(test_a_filter_without_the_hooks_loses_a_heading_that_crosses_the_boundary);
    RUN_TEST(test_the_hooks_can_be_cleared);
    RUN_TEST(test_the_hook_setters_reject_a_null_filter);
    return UNITY_END();
}
