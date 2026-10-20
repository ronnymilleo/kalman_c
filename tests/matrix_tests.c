/**
 * @file    matrix_tests.c
 * @brief   Unit tests for the matrix module.
 */

#include "matrix.h"

#include "test_support.h"
#include "unity.h"

static const double tolerance = 1e-12;

static void assert_matrix_equals(const matrix_t *expected, const matrix_t *actual) {
    TEST_ASSERT_EQUAL_UINT(expected->rows, actual->rows);
    TEST_ASSERT_EQUAL_UINT(expected->cols, actual->cols);
    for (size_t row = 0; row < expected->rows; row++) {
        for (size_t col = 0; col < expected->cols; col++) {
            TEST_ASSERT_DOUBLE_WITHIN(tolerance, expected->data[row][col], actual->data[row][col]);
        }
    }
}

static void test_a_new_matrix_has_the_requested_size_and_is_filled_with_zeros(void) {
    matrix_t matrix;
    matrix.data[0][0] = 42.0;

    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_init(&matrix, 2, 3));

    TEST_ASSERT_EQUAL_UINT(2, matrix.rows);
    TEST_ASSERT_EQUAL_UINT(3, matrix.cols);
    for (size_t row = 0; row < MATRIX_MAX_DIM; row++) {
        for (size_t col = 0; col < MATRIX_MAX_DIM; col++) {
            TEST_ASSERT_EQUAL_DOUBLE(0.0, matrix.data[row][col]);
        }
    }
}

static void test_a_matrix_cannot_be_empty_or_bigger_than_the_capacity(void) {
    matrix_t matrix;

    TEST_ASSERT_EQUAL_INT(matrix_status_invalid_size, matrix_init(&matrix, 0, 2));
    TEST_ASSERT_EQUAL_INT(matrix_status_invalid_size, matrix_init(&matrix, 2, 0));
    TEST_ASSERT_EQUAL_INT(matrix_status_invalid_size, matrix_init(&matrix, MATRIX_MAX_DIM + 1, 1));
    TEST_ASSERT_EQUAL_INT(matrix_status_invalid_size, matrix_init(&matrix, 1, MATRIX_MAX_DIM + 1));
    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_init(&matrix, MATRIX_MAX_DIM, MATRIX_MAX_DIM));
}

static void test_operations_reject_null_arguments(void) {
    matrix_t matrix = test_support_diagonal(2, 1.0);
    double vector[2] = {1.0, 2.0};

    TEST_ASSERT_EQUAL_INT(matrix_status_null_argument, matrix_init(NULL, 2, 2));
    TEST_ASSERT_EQUAL_INT(matrix_status_null_argument, matrix_from_values(&matrix, 2, 2, NULL));
    TEST_ASSERT_EQUAL_INT(matrix_status_null_argument, matrix_identity(NULL, 2));
    TEST_ASSERT_EQUAL_INT(matrix_status_null_argument, matrix_add(&matrix, NULL, &matrix));
    TEST_ASSERT_EQUAL_INT(matrix_status_null_argument, matrix_subtract(NULL, &matrix, &matrix));
    TEST_ASSERT_EQUAL_INT(matrix_status_null_argument, matrix_transpose(&matrix, NULL));
    TEST_ASSERT_EQUAL_INT(matrix_status_null_argument, matrix_multiply(&matrix, &matrix, NULL));
    TEST_ASSERT_EQUAL_INT(matrix_status_null_argument, matrix_vector_multiply(&matrix, NULL, vector));
    TEST_ASSERT_EQUAL_INT(matrix_status_null_argument, matrix_inverse(NULL, &matrix));
}

static void test_values_fill_a_matrix_row_by_row(void) {
    matrix_t matrix;

    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_from_values(&matrix, 2, 3, (const double[]){1, 2, 3, 4, 5, 6}));

    TEST_ASSERT_EQUAL_DOUBLE(3.0, matrix.data[0][2]);
    TEST_ASSERT_EQUAL_DOUBLE(4.0, matrix.data[1][0]);
}

static void test_the_identity_has_ones_on_the_diagonal_only(void) {
    matrix_t identity;

    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_identity(&identity, 3));

    const matrix_t expected = test_support_matrix(3, 3, (const double[]){1, 0, 0, 0, 1, 0, 0, 0, 1});
    assert_matrix_equals(&expected, &identity);
}

static void test_adding_matrices_adds_each_element(void) {
    const matrix_t a = test_support_matrix(2, 2, (const double[]){1, 2, 3, 4});
    const matrix_t b = test_support_matrix(2, 2, (const double[]){5, 6, 7, 8});
    matrix_t sum;

    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_add(&a, &b, &sum));

    const matrix_t expected = test_support_matrix(2, 2, (const double[]){6, 8, 10, 12});
    assert_matrix_equals(&expected, &sum);
}

static void test_subtracting_matrices_subtracts_each_element(void) {
    const matrix_t a = test_support_matrix(2, 2, (const double[]){5, 6, 7, 8});
    const matrix_t b = test_support_matrix(2, 2, (const double[]){1, 2, 3, 4});
    matrix_t difference;

    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_subtract(&a, &b, &difference));

    const matrix_t expected = test_support_matrix(2, 2, (const double[]){4, 4, 4, 4});
    assert_matrix_equals(&expected, &difference);
}

static void test_a_sum_can_be_stored_in_one_of_its_operands(void) {
    matrix_t a = test_support_matrix(1, 2, (const double[]){1, 2});
    const matrix_t b = test_support_matrix(1, 2, (const double[]){10, 20});

    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_add(&a, &b, &a));

    const matrix_t expected = test_support_matrix(1, 2, (const double[]){11, 22});
    assert_matrix_equals(&expected, &a);
}

static void test_matrices_of_different_sizes_cannot_be_added_or_subtracted(void) {
    const matrix_t a = test_support_matrix(2, 2, (const double[]){1, 2, 3, 4});
    const matrix_t b = test_support_matrix(2, 3, (const double[]){1, 2, 3, 4, 5, 6});
    matrix_t result;

    TEST_ASSERT_EQUAL_INT(matrix_status_size_mismatch, matrix_add(&a, &b, &result));
    TEST_ASSERT_EQUAL_INT(matrix_status_size_mismatch, matrix_subtract(&a, &b, &result));
}

static void test_transposing_a_rectangular_matrix_swaps_its_size(void) {
    const matrix_t a = test_support_matrix(2, 3, (const double[]){1, 2, 3, 4, 5, 6});
    matrix_t transposed;

    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_transpose(&a, &transposed));

    const matrix_t expected = test_support_matrix(3, 2, (const double[]){1, 4, 2, 5, 3, 6});
    assert_matrix_equals(&expected, &transposed);
}

static void test_a_matrix_can_be_transposed_onto_itself(void) {
    matrix_t a = test_support_matrix(2, 2, (const double[]){1, 2, 3, 4});

    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_transpose(&a, &a));

    const matrix_t expected = test_support_matrix(2, 2, (const double[]){1, 3, 2, 4});
    assert_matrix_equals(&expected, &a);
}

static void test_multiplying_matrices_follows_the_row_by_column_rule(void) {
    const matrix_t a = test_support_matrix(2, 3, (const double[]){1, 2, 3, 4, 5, 6});
    const matrix_t b = test_support_matrix(3, 2, (const double[]){7, 8, 9, 10, 11, 12});
    matrix_t product;

    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_multiply(&a, &b, &product));

    const matrix_t expected = test_support_matrix(2, 2, (const double[]){58, 64, 139, 154});
    assert_matrix_equals(&expected, &product);
}

static void test_a_product_can_be_stored_in_one_of_its_operands(void) {
    matrix_t a = test_support_matrix(2, 2, (const double[]){1, 2, 3, 4});
    const matrix_t b = test_support_matrix(2, 2, (const double[]){0, 1, 1, 0});

    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_multiply(&a, &b, &a));

    // Multiplying by a permutation swaps the columns
    const matrix_t expected = test_support_matrix(2, 2, (const double[]){2, 1, 4, 3});
    assert_matrix_equals(&expected, &a);
}

static void test_matrices_whose_inner_sizes_differ_cannot_be_multiplied(void) {
    const matrix_t a = test_support_matrix(2, 3, (const double[]){1, 2, 3, 4, 5, 6});
    matrix_t product;

    TEST_ASSERT_EQUAL_INT(matrix_status_size_mismatch, matrix_multiply(&a, &a, &product));
}

static void test_multiplying_by_a_vector_gives_one_value_per_row(void) {
    const matrix_t a = test_support_matrix(2, 3, (const double[]){1, 2, 3, 4, 5, 6});
    const double vector[3] = {1.0, 0.0, -1.0};
    double result[2] = {0.0, 0.0};

    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_vector_multiply(&a, vector, result));

    TEST_ASSERT_EQUAL_DOUBLE(-2.0, result[0]);
    TEST_ASSERT_EQUAL_DOUBLE(-2.0, result[1]);
}

static void test_a_matrix_vector_product_can_overwrite_its_vector(void) {
    const matrix_t swap = test_support_matrix(2, 2, (const double[]){0, 1, 1, 0});
    double vector[2] = {1.0, 2.0};

    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_vector_multiply(&swap, vector, vector));

    TEST_ASSERT_EQUAL_DOUBLE(2.0, vector[0]);
    TEST_ASSERT_EQUAL_DOUBLE(1.0, vector[1]);
}

static void test_the_inverse_of_a_2x2_matrix_matches_the_closed_form(void) {
    const matrix_t a = test_support_matrix(2, 2, (const double[]){4, 7, 2, 6});
    matrix_t inverse;

    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_inverse(&a, &inverse));

    const matrix_t expected = test_support_matrix(2, 2, (const double[]){0.6, -0.7, -0.2, 0.4});
    assert_matrix_equals(&expected, &inverse);
}

static void test_a_matrix_times_its_inverse_is_the_identity(void) {
    const matrix_t a = test_support_matrix(3, 3, (const double[]){2, -1, 0, -1, 2, -1, 0, -1, 2});
    matrix_t inverse;
    matrix_t product;

    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_inverse(&a, &inverse));
    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_multiply(&a, &inverse, &product));

    const matrix_t identity = test_support_diagonal(3, 1.0);
    assert_matrix_equals(&identity, &product);
}

static void test_inversion_swaps_rows_when_the_diagonal_has_a_zero(void) {
    const matrix_t a = test_support_matrix(2, 2, (const double[]){0, 1, 1, 0});
    matrix_t inverse;

    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_inverse(&a, &inverse));

    assert_matrix_equals(&a, &inverse);
}

static void test_a_matrix_with_tiny_entries_is_still_invertible(void) {
    // A converged covariance can be this small; only a matrix that is singular relative to its scale has no inverse
    const matrix_t a = test_support_diagonal(2, 1e-15);
    matrix_t inverse;

    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_inverse(&a, &inverse));

    TEST_ASSERT_DOUBLE_WITHIN(1e3, 1e15, inverse.data[0][0]);
}

static void test_a_singular_matrix_has_no_inverse_and_leaves_the_result_untouched(void) {
    const matrix_t singular = test_support_matrix(2, 2, (const double[]){1, 2, 2, 4});
    matrix_t result = test_support_diagonal(2, 7.0);

    TEST_ASSERT_EQUAL_INT(matrix_status_singular, matrix_inverse(&singular, &result));

    TEST_ASSERT_EQUAL_DOUBLE(7.0, result.data[0][0]);
}

static void test_the_zero_matrix_has_no_inverse(void) {
    matrix_t zero;
    matrix_t result;
    TEST_ASSERT_EQUAL_INT(matrix_status_ok, matrix_init(&zero, 2, 2));

    TEST_ASSERT_EQUAL_INT(matrix_status_singular, matrix_inverse(&zero, &result));
}

static void test_only_square_matrices_can_be_inverted(void) {
    const matrix_t a = test_support_matrix(2, 3, (const double[]){1, 2, 3, 4, 5, 6});
    matrix_t result;

    TEST_ASSERT_EQUAL_INT(matrix_status_size_mismatch, matrix_inverse(&a, &result));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_a_new_matrix_has_the_requested_size_and_is_filled_with_zeros);
    RUN_TEST(test_a_matrix_cannot_be_empty_or_bigger_than_the_capacity);
    RUN_TEST(test_operations_reject_null_arguments);
    RUN_TEST(test_values_fill_a_matrix_row_by_row);
    RUN_TEST(test_the_identity_has_ones_on_the_diagonal_only);
    RUN_TEST(test_adding_matrices_adds_each_element);
    RUN_TEST(test_subtracting_matrices_subtracts_each_element);
    RUN_TEST(test_a_sum_can_be_stored_in_one_of_its_operands);
    RUN_TEST(test_matrices_of_different_sizes_cannot_be_added_or_subtracted);
    RUN_TEST(test_transposing_a_rectangular_matrix_swaps_its_size);
    RUN_TEST(test_a_matrix_can_be_transposed_onto_itself);
    RUN_TEST(test_multiplying_matrices_follows_the_row_by_column_rule);
    RUN_TEST(test_a_product_can_be_stored_in_one_of_its_operands);
    RUN_TEST(test_matrices_whose_inner_sizes_differ_cannot_be_multiplied);
    RUN_TEST(test_multiplying_by_a_vector_gives_one_value_per_row);
    RUN_TEST(test_a_matrix_vector_product_can_overwrite_its_vector);
    RUN_TEST(test_the_inverse_of_a_2x2_matrix_matches_the_closed_form);
    RUN_TEST(test_a_matrix_times_its_inverse_is_the_identity);
    RUN_TEST(test_inversion_swaps_rows_when_the_diagonal_has_a_zero);
    RUN_TEST(test_a_matrix_with_tiny_entries_is_still_invertible);
    RUN_TEST(test_a_singular_matrix_has_no_inverse_and_leaves_the_result_untouched);
    RUN_TEST(test_the_zero_matrix_has_no_inverse);
    RUN_TEST(test_only_square_matrices_can_be_inverted);
    return UNITY_END();
}
