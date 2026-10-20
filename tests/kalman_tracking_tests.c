/**
 * @file    kalman_tracking_tests.c
 * @brief   Tracking scenarios for the kalman module: a filter follows a moving object through noisy measurements.
 */

#include "kalman.h"

#include "test_support.h"
#include "unity.h"

#include <math.h>

/**
 * @struct  tracking_error_t
 * @brief   Distance between the final estimate and the truth.
 */
typedef struct tracking_error_t {
    double position;
    double velocity;
} tracking_error_t;

// Follows an object that moves at a constant velocity along one axis, measured with uniform noise
static tracking_error_t track_1d(kalman_t *kf, const double dt, const double true_velocity, const int steps,
                                 const double noise_amplitude, const uint64_t seed) {
    test_support_random_t random;
    test_support_random_seed(&random, seed);

    double true_position = 0.0;
    for (int step = 0; step < steps; step++) {
        true_position += true_velocity * dt;
        const double measurement[1] = {true_position + test_support_random_symmetric(&random, noise_amplitude)};
        TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_predict(kf, NULL));
        TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_update(kf, measurement));
    }

    double state[2];
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_get_state(kf, state));
    return (tracking_error_t){.position = fabs(state[0] - true_position), .velocity = fabs(state[1] - true_velocity)};
}

// Follows an object that moves at a constant velocity on a plane, measured with uniform noise on each axis
static tracking_error_t track_2d(kalman_t *kf, const double dt, const double true_vx, const double true_vy,
                                 const int steps, const double noise_amplitude, const uint64_t seed) {
    test_support_random_t random;
    test_support_random_seed(&random, seed);

    double true_x = 0.0;
    double true_y = 0.0;
    for (int step = 0; step < steps; step++) {
        true_x += true_vx * dt;
        true_y += true_vy * dt;
        const double measurement[2] = {true_x + test_support_random_symmetric(&random, noise_amplitude),
                                       true_y + test_support_random_symmetric(&random, noise_amplitude)};
        TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_predict(kf, NULL));
        TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_update(kf, measurement));
    }

    double state[4];
    TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_get_state(kf, state));
    return (tracking_error_t){.position = hypot(state[0] - true_x, state[1] - true_y),
                              .velocity = hypot(state[2] - true_vx, state[3] - true_vy)};
}

static void test_a_filter_with_a_poor_initial_guess_finds_the_velocity_of_a_1d_object(void) {
    const double dt = 0.1;
    const double measurement_noise = 0.5;
    test_support_model_t model = test_support_model_1d(dt, measurement_noise * measurement_noise, 10.0);
    model.q = test_support_position_velocity_noise(dt, 0.1);
    kalman_t kf;
    test_support_start_filter(&kf, &model);

    const tracking_error_t error = track_1d(&kf, dt, 2.0, 50, measurement_noise, 42);

    TEST_ASSERT_TRUE(error.velocity < 0.5);
    TEST_ASSERT_TRUE(error.position < 2.0);
}

static void test_the_filter_finds_the_velocity_from_every_initial_velocity_guess(void) {
    const double dt = 0.1;
    const double initial_velocities[] = {0.0, 0.5, 1.0, 2.0, 3.0};

    for (size_t index = 0; index < sizeof(initial_velocities) / sizeof(initial_velocities[0]); index++) {
        test_support_model_t model = test_support_model_1d(dt, 0.25, 1.0);
        model.initial_state[1] = initial_velocities[index];
        kalman_t kf;
        test_support_start_filter(&kf, &model);

        const tracking_error_t error = track_1d(&kf, dt, 1.5, 30, 0.5, 100 + index);

        TEST_ASSERT_TRUE(error.velocity < 0.5);
    }
}

static void test_the_filter_follows_an_object_on_a_plane(void) {
    const double dt = 0.2;
    const double measurement_noise = 0.3;
    test_support_model_t model = test_support_model_2d(dt, 0.05 * 0.05, measurement_noise * measurement_noise, 5.0);
    kalman_t kf;
    test_support_start_filter(&kf, &model);

    const tracking_error_t error = track_2d(&kf, dt, 1.5, -1.0, 30, measurement_noise, 123);

    TEST_ASSERT_TRUE(error.position < 1.0);
}

static void test_the_filter_follows_every_direction_of_motion(void) {
    const double dt = 0.1;
    const struct {
        double vx;
        double vy;
    } motions[] = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {-1.0, 0.5}};

    for (size_t index = 0; index < sizeof(motions) / sizeof(motions[0]); index++) {
        test_support_model_t model = test_support_model_2d(dt, 0.01, 0.1, 1.0);
        kalman_t kf;
        test_support_start_filter(&kf, &model);

        const tracking_error_t error = track_2d(&kf, dt, motions[index].vx, motions[index].vy, 25, 0.3, 200 + index);

        TEST_ASSERT_TRUE(error.position < 0.5);
        TEST_ASSERT_TRUE(error.velocity < 0.3);
    }
}

static void test_the_filter_stays_close_to_the_truth_at_every_measurement_noise_level(void) {
    const double dt = 0.1;
    const double noise_levels[] = {0.1, 0.5, 1.0, 2.0};

    for (size_t index = 0; index < sizeof(noise_levels) / sizeof(noise_levels[0]); index++) {
        test_support_model_t model = test_support_model_1d(dt, noise_levels[index] * noise_levels[index], 1.0);
        kalman_t kf;
        test_support_start_filter(&kf, &model);

        const tracking_error_t error = track_1d(&kf, dt, 1.0, 20, noise_levels[index], 200 + index);

        TEST_ASSERT_TRUE(error.velocity < 1.0);
    }
}

static void test_the_filter_stays_close_to_the_truth_at_every_process_noise_level(void) {
    const double dt = 0.1;
    const double process_noise_levels[] = {0.01, 0.05, 0.1, 0.2, 0.5};

    for (size_t index = 0; index < sizeof(process_noise_levels) / sizeof(process_noise_levels[0]); index++) {
        test_support_model_t model = test_support_model_1d(dt, 0.25, 1.0);
        model.q = test_support_position_velocity_noise(dt, process_noise_levels[index]);
        kalman_t kf;
        test_support_start_filter(&kf, &model);

        const tracking_error_t error = track_1d(&kf, dt, 1.0, 25, 0.5, 300 + index);

        TEST_ASSERT_TRUE(error.velocity < 0.5);
    }
}

static void test_the_filter_balances_trust_in_the_model_and_in_the_sensor(void) {
    const double dt = 0.1;
    const struct {
        double measurement_noise;
        double process_noise;
    } combinations[] = {{0.1, 0.01}, {0.1, 0.1}, {1.0, 0.01}, {1.0, 0.1}};

    for (size_t index = 0; index < sizeof(combinations) / sizeof(combinations[0]); index++) {
        const double measurement_noise = combinations[index].measurement_noise;
        test_support_model_t model = test_support_model_1d(dt, measurement_noise * measurement_noise, 1.0);
        model.q = test_support_position_velocity_noise(dt, combinations[index].process_noise);
        kalman_t kf;
        test_support_start_filter(&kf, &model);

        const tracking_error_t error = track_1d(&kf, dt, 1.0, 30, measurement_noise, 400 + index);

        TEST_ASSERT_TRUE(error.velocity < 0.5);
        TEST_ASSERT_TRUE(error.position < 1.0);
    }
}

static void test_filtering_beats_trusting_the_raw_measurements(void) {
    const double dt = 0.1;
    const double measurement_noise = 1.0;
    test_support_model_t model = test_support_model_1d(dt, measurement_noise * measurement_noise / 3.0, 1.0);
    model.q = test_support_position_velocity_noise(dt, 0.01);
    kalman_t kf;
    test_support_start_filter(&kf, &model);

    // Uniform noise of amplitude 1 has a variance of 1/3. Compare the squared error of the filter with that of the
    // raw measurements over the last steps, once the filter has settled
    test_support_random_t random;
    test_support_random_seed(&random, 7);
    double true_position = 0.0;
    double filtered_squared_error = 0.0;
    double raw_squared_error = 0.0;
    const int steps = 200;
    const int settling_steps = 50;
    for (int step = 0; step < steps; step++) {
        true_position += 1.0 * dt;
        const double noise = test_support_random_symmetric(&random, measurement_noise);
        const double measurement[1] = {true_position + noise};
        TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_predict(&kf, NULL));
        TEST_ASSERT_EQUAL_INT(kalman_status_ok, kalman_update(&kf, measurement));
        if (step >= settling_steps) {
            filtered_squared_error += (kf.state[0] - true_position) * (kf.state[0] - true_position);
            raw_squared_error += noise * noise;
        }
    }

    TEST_ASSERT_TRUE(filtered_squared_error < 0.5 * raw_squared_error);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_a_filter_with_a_poor_initial_guess_finds_the_velocity_of_a_1d_object);
    RUN_TEST(test_the_filter_finds_the_velocity_from_every_initial_velocity_guess);
    RUN_TEST(test_the_filter_follows_an_object_on_a_plane);
    RUN_TEST(test_the_filter_follows_every_direction_of_motion);
    RUN_TEST(test_the_filter_stays_close_to_the_truth_at_every_measurement_noise_level);
    RUN_TEST(test_the_filter_stays_close_to_the_truth_at_every_process_noise_level);
    RUN_TEST(test_the_filter_balances_trust_in_the_model_and_in_the_sensor);
    RUN_TEST(test_filtering_beats_trusting_the_raw_measurements);
    return UNITY_END();
}
