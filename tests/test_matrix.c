/**
 * @file test_matrix.c
 * @brief Unit tests for matrix operations library
 * @author ronnymilleo
 * @date 08/07/25
 * @version 1.0
 * 
 * This file contains comprehensive unit tests for the matrix_math library.
 * The tests verify correctness of basic matrix operations including
 * addition, subtraction, multiplication, transpose, and identity matrix
 * creation.
 * 
 * @details Test coverage includes:
 * - Matrix addition and subtraction
 * - Matrix multiplication and transpose
 * - Identity matrix creation
 * - Memory allocation and deallocation
 * - Error condition handling
 * 
 * @note Tests use simple assertions with epsilon comparisons
 * @note Memory leaks are checked through proper cleanup
 * @note Test results are reported with pass/fail status
 * 
 * @see matrix_math.h for tested functions
 */

#include <stdio.h>
#include <math.h>
#include "../matrix_math.h"

#define EPSILON 1e-9

int test_matrix_add(void) {
    printf("Testing matrix addition...\n");
    
    double **A = matrix_allocate(2, 2);
    double **B = matrix_allocate(2, 2);
    double **result = matrix_allocate(2, 2);
    
    if (!A || !B || !result) {
        printf("FAILED: Memory allocation error\n");
        return 0;
    }
    
    // Initialize matrices
    A[0][0] = 1; A[0][1] = 2;
    A[1][0] = 3; A[1][1] = 4;
    
    B[0][0] = 5; B[0][1] = 6;
    B[1][0] = 7; B[1][1] = 8;
    
    // Perform addition
    if (matrix_add(A, B, result, 2, 2) != 0) {
        printf("FAILED: Matrix addition error\n");
        matrix_free(A, 2);
        matrix_free(B, 2);
        matrix_free(result, 2);
        return 0;
    }
    
    // Verify results
    if (fabs(result[0][0] - 6) > EPSILON || fabs(result[0][1] - 8) > EPSILON ||
        fabs(result[1][0] - 10) > EPSILON || fabs(result[1][1] - 12) > EPSILON) {
        printf("FAILED: Incorrect result\n");
        matrix_free(A, 2);
        matrix_free(B, 2);
        matrix_free(result, 2);
        return 0;
    }
    
    matrix_free(A, 2);
    matrix_free(B, 2);
    matrix_free(result, 2);
    
    printf("PASSED: Matrix addition\n");
    return 1;
}

int test_matrix_subtract(void) {
    printf("Testing matrix subtraction...\n");
    
    double **A = matrix_allocate(2, 2);
    double **B = matrix_allocate(2, 2);
    double **result = matrix_allocate(2, 2);
    
    if (!A || !B || !result) {
        printf("FAILED: Memory allocation error\n");
        return 0;
    }
    
    // Initialize matrices
    A[0][0] = 5; A[0][1] = 6;
    A[1][0] = 7; A[1][1] = 8;
    
    B[0][0] = 1; B[0][1] = 2;
    B[1][0] = 3; B[1][1] = 4;
    
    // Perform subtraction
    if (matrix_subtract(A, B, result, 2, 2) != 0) {
        printf("FAILED: Matrix subtraction error\n");
        matrix_free(A, 2);
        matrix_free(B, 2);
        matrix_free(result, 2);
        return 0;
    }
    
    // Verify results
    if (fabs(result[0][0] - 4) > EPSILON || fabs(result[0][1] - 4) > EPSILON ||
        fabs(result[1][0] - 4) > EPSILON || fabs(result[1][1] - 4) > EPSILON) {
        printf("FAILED: Incorrect result\n");
        matrix_free(A, 2);
        matrix_free(B, 2);
        matrix_free(result, 2);
        return 0;
    }
    
    matrix_free(A, 2);
    matrix_free(B, 2);
    matrix_free(result, 2);
    
    printf("PASSED: Matrix subtraction\n");
    return 1;
}

int test_matrix_transpose(void) {
    printf("Testing matrix transpose...\n");
    
    double **A = matrix_allocate(2, 2);
    double **result = matrix_allocate(2, 2);
    
    if (!A || !result) {
        printf("FAILED: Memory allocation error\n");
        return 0;
    }
    
    // Initialize matrix
    A[0][0] = 1; A[0][1] = 2;
    A[1][0] = 3; A[1][1] = 4;
    
    // Perform transpose
    if (matrix_transpose(A, result, 2, 2) != 0) {
        printf("FAILED: Matrix transpose error\n");
        matrix_free(A, 2);
        matrix_free(result, 2);
        return 0;
    }
    
    // Verify results
    if (fabs(result[0][0] - 1) > EPSILON || fabs(result[0][1] - 3) > EPSILON ||
        fabs(result[1][0] - 2) > EPSILON || fabs(result[1][1] - 4) > EPSILON) {
        printf("FAILED: Incorrect result\n");
        matrix_free(A, 2);
        matrix_free(result, 2);
        return 0;
    }
    
    matrix_free(A, 2);
    matrix_free(result, 2);
    
    printf("PASSED: Matrix transpose\n");
    return 1;
}

int test_matrix_multiply(void) {
    printf("Testing matrix multiplication...\n");
    
    double **A = matrix_allocate(2, 2);
    double **B = matrix_allocate(2, 2);
    double **result = matrix_allocate(2, 2);
    
    if (!A || !B || !result) {
        printf("FAILED: Memory allocation error\n");
        return 0;
    }
    
    // Initialize matrices
    A[0][0] = 1; A[0][1] = 2;
    A[1][0] = 3; A[1][1] = 4;
    
    B[0][0] = 5; B[0][1] = 6;
    B[1][0] = 7; B[1][1] = 8;
    
    // Perform multiplication
    if (matrix_multiply(A, B, result, 2, 2, 2) != 0) {
        printf("FAILED: Matrix multiplication error\n");
        matrix_free(A, 2);
        matrix_free(B, 2);
        matrix_free(result, 2);
        return 0;
    }
    
    // Verify results (A*B = [[19, 22], [43, 50]])
    if (fabs(result[0][0] - 19) > EPSILON || fabs(result[0][1] - 22) > EPSILON ||
        fabs(result[1][0] - 43) > EPSILON || fabs(result[1][1] - 50) > EPSILON) {
        printf("FAILED: Incorrect result\n");
        printf("Expected: [[19, 22], [43, 50]]\n");
        printf("Got: [[%.3f, %.3f], [%.3f, %.3f]]\n", 
               result[0][0], result[0][1], result[1][0], result[1][1]);
        matrix_free(A, 2);
        matrix_free(B, 2);
        matrix_free(result, 2);
        return 0;
    }
    
    matrix_free(A, 2);
    matrix_free(B, 2);
    matrix_free(result, 2);
    
    printf("PASSED: Matrix multiplication\n");
    return 1;
}

int test_matrix_identity(void) {
    printf("Testing identity matrix...\n");
    
    double **result = matrix_allocate(3, 3);
    
    if (!result) {
        printf("FAILED: Memory allocation error\n");
        return 0;
    }
    
    // Create identity matrix
    if (matrix_identity(result, 3) != 0) {
        printf("FAILED: Identity matrix creation error\n");
        matrix_free(result, 3);
        return 0;
    }
    
    // Verify results
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            const double expected = (i == j) ? 1.0 : 0.0;
            if (fabs(result[i][j] - expected) > EPSILON) {
                printf("FAILED: Incorrect identity matrix\n");
                matrix_free(result, 3);
                return 0;
            }
        }
    }
    
    matrix_free(result, 3);
    
    printf("PASSED: Identity matrix\n");
    return 1;
}

int main(void) {
    printf("Running matrix math tests...\n\n");
    
    int passed = 0;
    const int total = 5;
    
    passed += test_matrix_add();
    passed += test_matrix_subtract();
    passed += test_matrix_transpose();
    passed += test_matrix_multiply();
    passed += test_matrix_identity();
    
    printf("\n=== Test Results ===\n");
    printf("Passed: %d/%d tests\n", passed, total);
    
    if (passed == total) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("Some tests failed!\n");
        return 1;
    }
}
