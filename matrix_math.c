/**
 * @file matrix_math.c
 * @brief Implementation of matrix operations for numerical computations
 * @author ronnymilleo
 * @date 08/07/25
 * @version 1.0
 * 
 * This file implements matrix operations including memory management,
 * basic arithmetic operations, and advanced operations like matrix
 * inversion using Gaussian elimination with partial pivoting.
 * 
 * @details Implementation features:
 * - Dynamic memory allocation with error checking
 * - Robust error handling with proper cleanup
 * - Optimized algorithms for common operations
 * - Numerical stability considerations
 * 
 * @note All functions use double precision arithmetic
 * @note Memory allocation uses calloc() for zero-initialization
 * @note Matrix inversion uses Gaussian elimination with partial pivoting
 */

#include "matrix_math.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

double** matrix_allocate(size_t rows, size_t cols) {
    double **matrix = (double**)malloc(rows * sizeof(double*));
    if (!matrix) return NULL;
    
    for (size_t i = 0; i < rows; i++) {
        matrix[i] = (double*)calloc(cols, sizeof(double));
        if (!matrix[i]) {
            // Free previously allocated rows on failure
            for (size_t j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

void matrix_free(double **matrix, size_t rows) {
    if (!matrix) return;
    
    for (size_t i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void matrix_copy(double **A, double **B, size_t rows, size_t cols) {
    if (!A || !B) return;
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            B[i][j] = A[i][j];
        }
    }
}

int matrix_add(double **A, double **B, double **result, size_t rows, size_t cols) {
    if (!A || !B || !result) return -1;

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    return 0;
}

int matrix_subtract(double **A, double **B, double **result, size_t rows, size_t cols) {
    if (!A || !B || !result) return -1;

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    return 0;
}

int matrix_transpose(double **A, double **result, size_t rows, size_t cols) {
    if (!A || !result) return -1;

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[j][i] = A[i][j];
        }
    }
    return 0;
}

int matrix_multiply(double **A, double **B, double **result, size_t rowsA, size_t colsA, size_t colsB) {
    if (!A || !B || !result) return -1;

    for (size_t i = 0; i < rowsA; i++) {
        for (size_t j = 0; j < colsB; j++) {
            result[i][j] = 0.0;
            for (size_t k = 0; k < colsA; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return 0;
}

int matrix_vector_multiply(double **A, double *v, double *result, size_t rows, size_t cols) {
    if (!A || !v || !result) return -1;

    for (size_t i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (size_t j = 0; j < cols; j++) {
            result[i] += A[i][j] * v[j];
        }
    }
    return 0;
}

int matrix_identity(double **result, size_t n) {
    if (!result) return -1;
    
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            result[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    return 0;
}

int matrix_inverse(double **A, double **result, size_t n) {
    if (!A || !result) return -1;

    // Create augmented matrix [A|I]
    double **augmented = matrix_allocate(n, 2 * n);
    if (!augmented) return -1;

    // Initialize augmented matrix
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            augmented[i][j] = A[i][j];
        }
        augmented[i][i + n] = 1.0;
    }

    // Gaussian elimination
    for (size_t i = 0; i < n; i++) {
        // Find pivot
        size_t maxRow = i;
        for (size_t k = i + 1; k < n; k++) {
            if (fabs(augmented[k][i]) > fabs(augmented[maxRow][i])) {
                maxRow = k;
            }
        }

        // Swap rows if needed
        if (maxRow != i) {
            double *temp = augmented[i];
            augmented[i] = augmented[maxRow];
            augmented[maxRow] = temp;
        }

        // Check for singular matrix
        if (fabs(augmented[i][i]) < 1e-12) {
            matrix_free(augmented, n);
            return -1;
        }

        // Normalize pivot row
        double pivot = augmented[i][i];
        for (size_t j = 0; j < 2 * n; j++) {
            augmented[i][j] /= pivot;
        }

        // Eliminate other rows
        for (size_t k = 0; k < n; k++) {
            if (k != i) {
                double factor = augmented[k][i];
                for (size_t j = 0; j < 2 * n; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }

    // Extract inverse matrix
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            result[i][j] = augmented[i][j + n];
        }
    }

    matrix_free(augmented, n);
    return 0;
}
