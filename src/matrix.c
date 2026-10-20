/**
 * @file    matrix.c
 * @brief   Fixed-capacity matrices and the linear algebra the Kalman filters need.
 */

#include "matrix.h"

#include <math.h>
#include <stdbool.h>

// A pivot smaller than this share of the largest entry is treated as zero; a fixed absolute limit would reject
// well-conditioned matrices whose entries are all tiny, such as a converged covariance
static const double singular_relative_tolerance = 1e-12;

static bool is_valid_size(const size_t rows, const size_t cols) {
    return rows >= 1 && rows <= MATRIX_MAX_DIM && cols >= 1 && cols <= MATRIX_MAX_DIM;
}

static double largest_magnitude(const matrix_t *matrix) {
    double largest = 0.0;
    for (size_t row = 0; row < matrix->rows; row++) {
        for (size_t col = 0; col < matrix->cols; col++) {
            largest = fmax(largest, fabs(matrix->data[row][col]));
        }
    }
    return largest;
}

/**
 * @brief   Sets the size of a matrix and fills it with zeros.
 * @param[out] matrix  Matrix to initialize.
 * @param[in]  rows    Number of rows, from 1 to MATRIX_MAX_DIM.
 * @param[in]  cols    Number of columns, from 1 to MATRIX_MAX_DIM.
 * @return  matrix_status_ok, or why the matrix could not be initialized.
 */
matrix_status_t matrix_init(matrix_t *matrix, const size_t rows, const size_t cols) {
    if (matrix == NULL) {
        return matrix_status_null_argument;
    }
    if (!is_valid_size(rows, cols)) {
        return matrix_status_invalid_size;
    }
    *matrix = (matrix_t){.rows = rows, .cols = cols};
    return matrix_status_ok;
}

/**
 * @brief   Sets the size of a matrix and fills it from a flat list of values.
 * @param[out] matrix  Matrix to fill.
 * @param[in]  rows    Number of rows, from 1 to MATRIX_MAX_DIM.
 * @param[in]  cols    Number of columns, from 1 to MATRIX_MAX_DIM.
 * @param[in]  values  `rows * cols` values in row-major order.
 * @return  matrix_status_ok, or why the matrix could not be filled.
 */
matrix_status_t matrix_from_values(matrix_t *matrix, const size_t rows, const size_t cols, const double *values) {
    if (values == NULL) {
        return matrix_status_null_argument;
    }
    const matrix_status_t status = matrix_init(matrix, rows, cols);
    if (status != matrix_status_ok) {
        return status;
    }
    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            matrix->data[row][col] = values[row * cols + col];
        }
    }
    return matrix_status_ok;
}

/**
 * @brief   Turns a matrix into the identity matrix of a given size.
 * @param[out] matrix  Matrix to overwrite.
 * @param[in]  size    Number of rows and columns, from 1 to MATRIX_MAX_DIM.
 * @return  matrix_status_ok, or why the matrix could not be initialized.
 */
matrix_status_t matrix_identity(matrix_t *matrix, const size_t size) {
    const matrix_status_t status = matrix_init(matrix, size, size);
    if (status != matrix_status_ok) {
        return status;
    }
    for (size_t index = 0; index < size; index++) {
        matrix->data[index][index] = 1.0;
    }
    return matrix_status_ok;
}

/**
 * @brief   Adds two matrices of the same size.
 * @param[in]  a       First matrix.
 * @param[in]  b       Second matrix.
 * @param[out] result  Sum `a + b`; it may be the same matrix as `a` or `b`.
 * @return  matrix_status_ok, or why the sum could not be computed.
 */
matrix_status_t matrix_add(const matrix_t *a, const matrix_t *b, matrix_t *result) {
    if (a == NULL || b == NULL || result == NULL) {
        return matrix_status_null_argument;
    }
    if (a->rows != b->rows || a->cols != b->cols) {
        return matrix_status_size_mismatch;
    }
    result->rows = a->rows;
    result->cols = a->cols;
    for (size_t row = 0; row < a->rows; row++) {
        for (size_t col = 0; col < a->cols; col++) {
            result->data[row][col] = a->data[row][col] + b->data[row][col];
        }
    }
    return matrix_status_ok;
}

/**
 * @brief   Subtracts a matrix from another of the same size.
 * @param[in]  a       Matrix to subtract from.
 * @param[in]  b       Matrix to subtract.
 * @param[out] result  Difference `a - b`; it may be the same matrix as `a` or `b`.
 * @return  matrix_status_ok, or why the difference could not be computed.
 */
matrix_status_t matrix_subtract(const matrix_t *a, const matrix_t *b, matrix_t *result) {
    if (a == NULL || b == NULL || result == NULL) {
        return matrix_status_null_argument;
    }
    if (a->rows != b->rows || a->cols != b->cols) {
        return matrix_status_size_mismatch;
    }
    result->rows = a->rows;
    result->cols = a->cols;
    for (size_t row = 0; row < a->rows; row++) {
        for (size_t col = 0; col < a->cols; col++) {
            result->data[row][col] = a->data[row][col] - b->data[row][col];
        }
    }
    return matrix_status_ok;
}

/**
 * @brief   Swaps the rows and columns of a matrix.
 * @param[in]  a       Matrix to transpose.
 * @param[out] result  Transpose of `a`; it may be the same matrix as `a`.
 * @return  matrix_status_ok, or why the transpose could not be computed.
 */
matrix_status_t matrix_transpose(const matrix_t *a, matrix_t *result) {
    if (a == NULL || result == NULL) {
        return matrix_status_null_argument;
    }
    // Built apart from the result, so that transposing a matrix onto itself works
    matrix_t transposed = {.rows = a->cols, .cols = a->rows};
    for (size_t row = 0; row < a->rows; row++) {
        for (size_t col = 0; col < a->cols; col++) {
            transposed.data[col][row] = a->data[row][col];
        }
    }
    *result = transposed;
    return matrix_status_ok;
}

/**
 * @brief   Multiplies two matrices.
 * @param[in]  a       Left matrix, `m x k`.
 * @param[in]  b       Right matrix, `k x n`.
 * @param[out] result  Product `a * b`, `m x n`; it may be the same matrix as `a` or `b`.
 * @return  matrix_status_ok, or why the product could not be computed.
 */
matrix_status_t matrix_multiply(const matrix_t *a, const matrix_t *b, matrix_t *result) {
    if (a == NULL || b == NULL || result == NULL) {
        return matrix_status_null_argument;
    }
    if (a->cols != b->rows) {
        return matrix_status_size_mismatch;
    }
    // Built apart from the result, so that it may share storage with either operand
    matrix_t product = {.rows = a->rows, .cols = b->cols};
    for (size_t row = 0; row < a->rows; row++) {
        for (size_t col = 0; col < b->cols; col++) {
            for (size_t inner = 0; inner < a->cols; inner++) {
                product.data[row][col] += a->data[row][inner] * b->data[inner][col];
            }
        }
    }
    *result = product;
    return matrix_status_ok;
}

/**
 * @brief   Multiplies a matrix by a column vector.
 * @param[in]  a       Matrix, `m x n`.
 * @param[in]  vector  `n` values.
 * @param[out] result  `m` values holding `a * vector`; it may be the same array as `vector` when `m == n`.
 * @return  matrix_status_ok, or why the product could not be computed.
 */
matrix_status_t matrix_vector_multiply(const matrix_t *a, const double *vector, double *result) {
    if (a == NULL || vector == NULL || result == NULL) {
        return matrix_status_null_argument;
    }
    double product[MATRIX_MAX_DIM] = {0.0};
    for (size_t row = 0; row < a->rows; row++) {
        for (size_t col = 0; col < a->cols; col++) {
            product[row] += a->data[row][col] * vector[col];
        }
    }
    for (size_t row = 0; row < a->rows; row++) {
        result[row] = product[row];
    }
    return matrix_status_ok;
}

/**
 * @brief   Inverts a square matrix by Gauss-Jordan elimination with partial pivoting.
 * @param[in]  a       Square matrix to invert.
 * @param[out] result  Inverse of `a`; it may be the same matrix as `a`, and is left untouched on failure.
 * @return  matrix_status_ok, matrix_status_singular when `a` has no inverse, or why the input is unusable.
 */
matrix_status_t matrix_inverse(const matrix_t *a, matrix_t *result) {
    if (a == NULL || result == NULL) {
        return matrix_status_null_argument;
    }
    if (a->rows != a->cols) {
        return matrix_status_size_mismatch;
    }

    const size_t size = a->rows;
    const double tolerance = singular_relative_tolerance * largest_magnitude(a);

    // Work on [A | I]; once A has been reduced to the identity, the right half holds its inverse
    double augmented[MATRIX_MAX_DIM][2 * MATRIX_MAX_DIM] = {{0.0}};
    for (size_t row = 0; row < size; row++) {
        for (size_t col = 0; col < size; col++) {
            augmented[row][col] = a->data[row][col];
        }
        augmented[row][size + row] = 1.0;
    }

    for (size_t pivot_index = 0; pivot_index < size; pivot_index++) {
        size_t pivot_row = pivot_index;
        for (size_t row = pivot_index + 1; row < size; row++) {
            if (fabs(augmented[row][pivot_index]) > fabs(augmented[pivot_row][pivot_index])) {
                pivot_row = row;
            }
        }
        if (fabs(augmented[pivot_row][pivot_index]) <= tolerance) {
            return matrix_status_singular;
        }
        if (pivot_row != pivot_index) {
            for (size_t col = 0; col < 2 * size; col++) {
                const double swapped = augmented[pivot_index][col];
                augmented[pivot_index][col] = augmented[pivot_row][col];
                augmented[pivot_row][col] = swapped;
            }
        }

        const double pivot = augmented[pivot_index][pivot_index];
        for (size_t col = 0; col < 2 * size; col++) {
            augmented[pivot_index][col] /= pivot;
        }
        for (size_t row = 0; row < size; row++) {
            if (row == pivot_index) {
                continue;
            }
            const double factor = augmented[row][pivot_index];
            for (size_t col = 0; col < 2 * size; col++) {
                augmented[row][col] -= factor * augmented[pivot_index][col];
            }
        }
    }

    result->rows = size;
    result->cols = size;
    for (size_t row = 0; row < size; row++) {
        for (size_t col = 0; col < size; col++) {
            result->data[row][col] = augmented[row][size + col];
        }
    }
    return matrix_status_ok;
}
