/**
 * @file matrix_math.h
 * @brief Matrix operations library for numerical computations
 * @author ronnymilleo
 * @date 08/07/25
 * @version 1.0
 * 
 * This file contains declarations for matrix operations including allocation,
 * deallocation, and mathematical operations such as addition, subtraction,
 * multiplication, transpose, and inversion. All functions use double precision
 * floating point arithmetic and follow C99 standards.
 * 
 * @note All matrices are represented as double** (pointer to pointer to double)
 * @note Error handling is done through return codes: 0 for success, -1 for failure
 * @note Memory allocation failures return NULL pointers
 */

#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup memory_management Memory Management
 *  @brief Functions for matrix memory allocation and deallocation
 *  @{
 */

/**
 * @brief Allocate memory for a 2D matrix
 * 
 * Allocates memory for a 2D matrix using a pointer-to-pointer structure.
 * The memory is initialized to zero using calloc().
 * 
 * @param rows Number of rows (must be > 0)
 * @param cols Number of columns (must be > 0)
 * 
 * @return Pointer to allocated matrix on success, NULL on failure
 * 
 * @note Memory is zero-initialized using calloc()
 * @note Each row is allocated separately for maximum flexibility
 * @note Must be freed using matrix_free() to avoid memory leaks
 * 
 * @warning Returns NULL if memory allocation fails
 * @warning Returns NULL if rows or cols is 0
 * 
 * @see matrix_free()
 */
double** matrix_allocate(size_t rows, size_t cols);

/**
 * Free memory for a 2D array (matrix)
 * @param matrix Matrix to free
 * @param rows Number of rows
 */
void matrix_free(double **matrix, size_t rows);

/**
 * Copy matrix A to matrix B
 * @param A Source matrix
 * @param B Destination matrix
 * @param rows Number of rows
 * @param cols Number of columns
 */
void matrix_copy(double **A, double **B, size_t rows, size_t cols);

/** @} */ // end of memory_management group

/** @defgroup basic_operations Basic Matrix Operations
 *  @brief Fundamental matrix arithmetic operations
 *  @{
 */

/**
 * Matrix addition: result = A + B
 * @param A First matrix
 * @param B Second matrix
 * @param result Result matrix
 * @param rows Number of rows
 * @param cols Number of columns
 * @return 0 on success, -1 on failure
 */
int matrix_add(double **A, double **B, double **result, size_t rows, size_t cols);

/**
 * Matrix subtraction: result = A - B
 * @param A First matrix
 * @param B Second matrix
 * @param result Result matrix
 * @param rows Number of rows
 * @param cols Number of columns
 * @return 0 on success, -1 on failure
 */
int matrix_subtract(double **A, double **B, double **result, size_t rows, size_t cols);

/**
 * Matrix transpose: result = A^T
 * @param A Input matrix
 * @param result Result matrix
 * @param rows Number of rows in A
 * @param cols Number of columns in A
 * @return 0 on success, -1 on failure
 */
int matrix_transpose(double **A, double **result, size_t rows, size_t cols);

/** @} */ // end of basic_operations group

/** @defgroup advanced_operations Advanced Matrix Operations
 *  @brief Complex matrix operations and linear algebra
 *  @{
 */

/**
 * Matrix multiplication: result = A * B
 * @param A First matrix (rowsA x colsA)
 * @param B Second matrix (colsA x colsB)
 * @param result Result matrix (rowsA x colsB)
 * @param rowsA Number of rows in A
 * @param colsA Number of columns in A (must equal rows in B)
 * @param colsB Number of columns in B
 * @return 0 on success, -1 on failure
 */
int matrix_multiply(double **A, double **B, double **result, size_t rowsA, size_t colsA, size_t colsB);

/**
 * Matrix-vector multiplication: result = A * v
 * @param A Matrix (rows x cols)
 * @param v Vector (cols elements)
 * @param result Result vector (rows elements)
 * @param rows Number of rows in A
 * @param cols Number of columns in A
 * @return 0 on success, -1 on failure
 */
int matrix_vector_multiply(double **A, const double *v, double *result, size_t rows, size_t cols);

/**
 * Create identity matrix
 * @param result Result matrix (n x n)
 * @param n Matrix size
 * @return 0 on success, -1 on failure
 */
int matrix_identity(double **result, size_t n);

/**
 * @brief Matrix inversion using Gaussian elimination
 * 
 * Computes the inverse of a square matrix using Gaussian elimination
 * with partial pivoting for numerical stability. The algorithm creates
 * an augmented matrix [A|I] and reduces it to [I|A^(-1)].
 * 
 * @param A Input matrix (n x n, must be square and non-singular)
 * @param result Result matrix where inverse will be stored (n x n)
 * @param n Matrix size (number of rows and columns)
 * 
 * @return 0 on success, -1 on failure
 * 
 * @note Uses partial pivoting for numerical stability
 * @note Checks for singular matrices (determinant near zero)
 * @note Input matrix A is not modified during computation
 * 
 * @warning Returns -1 if matrix is singular or near-singular
 * @warning Returns -1 if memory allocation for temporary matrices fails
 * @warning Returns -1 if input parameters are NULL
 * 
 * @see matrix_multiply()
 * @see matrix_identity()
 */
int matrix_inverse(double **A, double **result, size_t n);

/** @} */ // end of advanced_operations group

#ifdef __cplusplus
}
#endif

#endif //MATRIX_MATH_H
