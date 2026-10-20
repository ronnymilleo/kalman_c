/**
 * @file    test_support.h
 * @brief   Helpers shared by the unit tests: a repeatable noise source and ready-made filter models.
 */

#ifndef KALMAN_C_TEST_SUPPORT_H
#define KALMAN_C_TEST_SUPPORT_H

#include "kalman.h"
#include "matrix.h"

#include <stddef.h>
#include <stdint.h>

/**
 * @struct  test_support_random_t
 * @brief   Small pseudo-random generator.
 * @details The standard rand() gives different numbers on every C library, so tracking tests that depend on it would
 *          pass on one platform and fail on another. This generator gives the same sequence everywhere.
 */
typedef struct test_support_random_t {
    uint64_t state;
} test_support_random_t;

/**
 * @struct  test_support_model_t
 * @brief   Matrices and initial state needed to configure a linear filter.
 */
typedef struct test_support_model_t {
    size_t state_dim;
    size_t measurement_dim;
    matrix_t f;
    matrix_t h;
    matrix_t q;
    matrix_t r;
    matrix_t p;
    double initial_state[KALMAN_MAX_DIM];
} test_support_model_t;

// Noise
void test_support_random_seed(test_support_random_t *random, uint64_t seed);
double test_support_random_symmetric(test_support_random_t *random, double amplitude);

// Matrices
matrix_t test_support_matrix(size_t rows, size_t cols, const double *values);
matrix_t test_support_diagonal(size_t size, double value);

// Models
test_support_model_t test_support_model_1d(double dt, double measurement_variance, double initial_variance);
test_support_model_t test_support_model_2d(double dt, double process_variance, double measurement_variance,
                                           double initial_variance);
matrix_t test_support_position_velocity_noise(double dt, double process_noise_sigma);
void test_support_start_filter(kalman_t *kf, const test_support_model_t *model);

#endif // KALMAN_C_TEST_SUPPORT_H
