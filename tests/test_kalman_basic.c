/**
 * @file test_kalman_basic.c
 * @brief Basic functionality tests for the standard Kalman Filter implementation
 * @author ronnymilleo
 * @date 12/07/25
 * @version 1.0
 *
 * This file contains basic unit tests for the standard Kalman filter
 * functionality, focusing on core operations like initialization,
 * prediction, and update steps.
 * 
 * @details Test scenarios:
 * - Filter creation and initialization
 * - Basic prediction step validation
 * - Basic update step validation
 * - Matrix setup and configuration
 * 
 * @note Tests validate fundamental filter operations
 * @note Uses simple 2D state (position, velocity) with 1D measurement
 * 
 * @see kalman.h for API documentation
 * @see matrix_math.h for matrix operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "../kalman.h"
#include "../matrix_math.h"

// Test result tracking
static int tests_passed = 0;
static int tests_failed = 0;

// Simple color codes for test output
#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define GREEN   "\033[32m"
#define RED     "\033[31m"
#define BLUE    "\033[34m"
#define YELLOW  "\033[33m"

/**
 * @brief Print test header
 */
void print_test_header(const char* test_name) {
    printf(BOLD BLUE "\n=== %s ===" RESET "\n", test_name);
}

/**
 * @brief Print test result
 */
void print_test_result(const char* test_name, int passed) {
    if (passed) {
        printf(GREEN "[PASS]" RESET " %s\n", test_name);
        tests_passed++;
    } else {
        printf(RED "[FAIL]" RESET " %s\n", test_name);
        tests_failed++;
    }
}

/**
 * @brief Test basic Kalman filter functionality
 * 
 * This test validates basic filter operations including initialization,
 * prediction, and update steps with simple matrices.
 */
int test_basic_functionality(void) {
    print_test_header("Basic Kalman Filter Functionality");
    
    // Create a simple 2D state (position, velocity), 1D measurement (position) filter
    kalman_t *kf = kalman_create(2, 1, 0);
    if (!kf) {
        print_test_result("Basic filter creation", 0);
        return 0;
    }
    
    // Initialize state [position=1.0, velocity=0.5]
    double initial_state[2] = {1.0, 0.5};
    if (kalman_initialize_state(kf, initial_state) != 0) {
        print_test_result("State initialization", 0);
        kalman_destroy(kf);
        return 0;
    }
    
    // Set up system matrices
    double dt = 0.1;
    
    // State transition matrix (constant velocity model)
    double **F = matrix_allocate(2, 2);
    F[0][0] = 1.0; F[0][1] = dt;
    F[1][0] = 0.0; F[1][1] = 1.0;
    kalman_set_state_transition_matrix(kf, F);
    
    // Observation matrix (measure position only)
    double **H = matrix_allocate(1, 2);
    H[0][0] = 1.0; H[0][1] = 0.0;
    kalman_set_observation_matrix(kf, H);
    
    // Process noise covariance
    double **Q = matrix_allocate(2, 2);
    Q[0][0] = 0.01; Q[0][1] = 0.0;
    Q[1][0] = 0.0;  Q[1][1] = 0.01;
    kalman_set_process_noise_covariance(kf, Q);
    
    // Measurement noise covariance
    double **R = matrix_allocate(1, 1);
    R[0][0] = 0.1;
    kalman_set_measurement_noise_covariance(kf, R);
    
    // Initial error covariance
    double **P = matrix_allocate(2, 2);
    P[0][0] = 1.0; P[0][1] = 0.0;
    P[1][0] = 0.0; P[1][1] = 1.0;
    kalman_set_error_covariance(kf, P);
    
    // Test prediction step
    kalman_predict(kf, NULL);
    
    // Get predicted state
    double *predicted_state = kalman_get_state(kf);
    double expected_position = 1.0 + 0.5 * dt;  // position + velocity * dt
    double expected_velocity = 0.5;              // velocity remains constant
    
    int prediction_ok = (fabs(predicted_state[0] - expected_position) < 1e-10 &&
                        fabs(predicted_state[1] - expected_velocity) < 1e-10);
    
    // Test update step
    double measurement[1] = {1.06};  // Slightly noisy measurement
    kalman_update(kf, measurement);
    
    // Get updated state
    double *updated_state = kalman_get_state(kf);
    
    // Updated state should be between prediction and measurement
    int update_ok = (updated_state[0] > expected_position && 
                    updated_state[0] < measurement[0]);
    
    // Cleanup
    matrix_free(F, 2);
    matrix_free(H, 1);
    matrix_free(Q, 2);
    matrix_free(R, 1);
    matrix_free(P, 2);
    kalman_destroy(kf);
    
    int all_tests_ok = prediction_ok && update_ok;
    print_test_result("Basic functionality", all_tests_ok);
    
    if (all_tests_ok) {
        printf("  - Prediction step: PASS\n");
        printf("  - Update step: PASS\n");
    }
    
    return all_tests_ok;
}

/**
 * @brief Test filter creation and destruction
 */
int test_filter_lifecycle(void) {
    print_test_header("Filter Lifecycle");
    
    // Test creation
    kalman_t *kf = kalman_create(3, 2, 1);
    int creation_ok = (kf != NULL);
    print_test_result("Filter creation", creation_ok);
    
    if (creation_ok) {
        // Test state initialization
        double state[3] = {1.0, 2.0, 3.0};
        int init_ok = (kalman_initialize_state(kf, state) == 0);
        print_test_result("State initialization", init_ok);
        
        // Test state retrieval
        double *retrieved_state = kalman_get_state(kf);
        int retrieval_ok = (retrieved_state != NULL);
        print_test_result("State retrieval", retrieval_ok);
        
        // Test destruction
        kalman_destroy(kf);
        print_test_result("Filter destruction", 1); // Assuming no crash
        
        return creation_ok && init_ok && retrieval_ok;
    }
    
    return 0;
}

/**
 * @brief Main test function
 */
int main(void) {
    printf(BOLD "\nBasic Kalman Filter Test Suite\n" RESET);
    printf("==============================\n");
    
    // Run all tests
    test_filter_lifecycle();
    test_basic_functionality();
    
    // Print summary
    printf(BOLD "\n=== TEST SUMMARY ===" RESET "\n");
    printf("Tests passed: " GREEN "%d" RESET "\n", tests_passed);
    printf("Tests failed: " RED "%d" RESET "\n", tests_failed);
    printf("Total tests:  %d\n", tests_passed + tests_failed);
    
    if (tests_failed == 0) {
        printf(GREEN BOLD "\nAll basic tests PASSED! ✓" RESET "\n");
        return 0;
    } else {
        printf(RED BOLD "\nSome basic tests FAILED! ✗" RESET "\n");
        return 1;
    }
}
