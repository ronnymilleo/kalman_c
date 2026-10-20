/**
 * @file    matrix.h
 * @brief   Fixed-capacity matrices and the linear algebra the Kalman filters need.
 */

#ifndef KALMAN_C_MATRIX_H
#define KALMAN_C_MATRIX_H

#include <stddef.h>

// Largest number of rows or columns a matrix can have; override with -DMATRIX_MAX_DIM=n to fit the target's memory
#ifndef MATRIX_MAX_DIM
#define MATRIX_MAX_DIM 8
#endif

/**
 * @enum    matrix_status_t
 * @brief   Outcome of a matrix operation.
 */
typedef enum matrix_status_t {
    matrix_status_ok = 0,
    matrix_status_null_argument,
    matrix_status_invalid_size,
    matrix_status_size_mismatch,
    matrix_status_singular,
} matrix_status_t;

/**
 * @struct  matrix_t
 * @brief   Row-major matrix stored inline, so it needs no dynamic allocation.
 * @details Only the first `rows` x `cols` entries of `data` are in use, and both must be between 1 and
 *          MATRIX_MAX_DIM. Fill a matrix with matrix_init() or matrix_from_values() before using it.
 */
typedef struct matrix_t {
    size_t rows;
    size_t cols;
    double data[MATRIX_MAX_DIM][MATRIX_MAX_DIM];
} matrix_t;

// Construction
matrix_status_t matrix_init(matrix_t *matrix, size_t rows, size_t cols);
matrix_status_t matrix_from_values(matrix_t *matrix, size_t rows, size_t cols, const double *values);
matrix_status_t matrix_identity(matrix_t *matrix, size_t size);

// Element-wise arithmetic
matrix_status_t matrix_add(const matrix_t *a, const matrix_t *b, matrix_t *result);
matrix_status_t matrix_subtract(const matrix_t *a, const matrix_t *b, matrix_t *result);

// Products and transforms
matrix_status_t matrix_transpose(const matrix_t *a, matrix_t *result);
matrix_status_t matrix_multiply(const matrix_t *a, const matrix_t *b, matrix_t *result);
matrix_status_t matrix_vector_multiply(const matrix_t *a, const double *vector, double *result);
matrix_status_t matrix_inverse(const matrix_t *a, matrix_t *result);

#endif // KALMAN_C_MATRIX_H
