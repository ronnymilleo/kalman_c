/**
 * @file    angle_tests.c
 * @brief   Unit tests for the angle module.
 */

#include "angle.h"

#include "unity.h"

static const double tolerance = 1e-12;

static void test_an_angle_inside_one_turn_is_left_alone(void) {
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, 0.5, angle_wrap_radians(0.5));
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, -2.5, angle_wrap_radians(-2.5));
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, 0.0, angle_wrap_radians(0.0));
}

static void test_an_angle_past_pi_comes_back_on_the_negative_side(void) {
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, -ANGLE_PI + 0.2, angle_wrap_radians(ANGLE_PI + 0.2));
}

static void test_an_angle_below_minus_pi_comes_back_on_the_positive_side(void) {
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, ANGLE_PI - 0.2, angle_wrap_radians(-ANGLE_PI - 0.2));
}

static void test_pi_and_minus_pi_are_the_same_direction_and_wrap_to_pi(void) {
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, ANGLE_PI, angle_wrap_radians(ANGLE_PI));
    TEST_ASSERT_DOUBLE_WITHIN(tolerance, ANGLE_PI, angle_wrap_radians(-ANGLE_PI));
}

static void test_whole_turns_wrap_to_the_same_angle(void) {
    TEST_ASSERT_DOUBLE_WITHIN(1e-9, 1.0, angle_wrap_radians(1.0 + 10.0 * 2.0 * ANGLE_PI));
    TEST_ASSERT_DOUBLE_WITHIN(1e-9, 1.0, angle_wrap_radians(1.0 - 7.0 * 2.0 * ANGLE_PI));
}

static void test_the_difference_of_angles_across_the_boundary_is_the_short_way_around(void) {
    const double degrees = ANGLE_PI / 180.0;

    // 179 degrees minus -179 degrees is -2 degrees, not 358
    TEST_ASSERT_DOUBLE_WITHIN(1e-9, -2.0 * degrees, angle_difference_radians(179.0 * degrees, -179.0 * degrees));
    TEST_ASSERT_DOUBLE_WITHIN(1e-9, 2.0 * degrees, angle_difference_radians(-179.0 * degrees, 179.0 * degrees));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_an_angle_inside_one_turn_is_left_alone);
    RUN_TEST(test_an_angle_past_pi_comes_back_on_the_negative_side);
    RUN_TEST(test_an_angle_below_minus_pi_comes_back_on_the_positive_side);
    RUN_TEST(test_pi_and_minus_pi_are_the_same_direction_and_wrap_to_pi);
    RUN_TEST(test_whole_turns_wrap_to_the_same_angle);
    RUN_TEST(test_the_difference_of_angles_across_the_boundary_is_the_short_way_around);
    return UNITY_END();
}
