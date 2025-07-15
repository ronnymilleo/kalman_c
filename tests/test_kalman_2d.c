/**
 * @file test_kalman_2d.c
 * @brief 2D tracking tests for the standard Kalman Filter implementation
 * @author ronnymilleo
 * @date 12/07/25
 * @version 1.0
 *
 * This file contains tests for 2D position and velocity tracking using
 * the standard Kalman filter, focusing on multi-dimensional state
 * estimation and convergence validation.
 * 
 * @details Test scenarios:
 * - 2D object tracking with position and velocity estimation
 * - Multi-dimensional state space validation
 * - Convergence analysis in 2D space
 * - Error calculation in 2D coordinates
 * 
 * @note Tests demonstrate practical 2D tracking applications
 * @note Uses constant velocity motion model in both X and Y directions
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
 * @brief Test 2D object tracking
 * 
 * This test demonstrates 2D tracking of an object moving with constant
 * velocity in both X and Y directions, with noisy position measurements.
 */
int test_2d_tracking(void) {
    print_test_header("2D Object Tracking");
    
    // System parameters
    const double dt = 0.2;                    // Time step
    const double true_vx = 1.5;               // True X velocity
    const double true_vy = -1.0;              // True Y velocity  
    const double process_noise = 0.05;        // Process noise
    const double measurement_noise = 0.3;     // Measurement noise
    const int num_steps = 30;                 // Number of steps
    
    // Create 4D state (x, y, vx, vy), 2D measurement (x, y) KF
    kalman_t *kf = kalman_create(4, 2, 0);
    if (!kf) {
        print_test_result("2D tracking - filter creation", 0);
        return 0;
    }
    
    // Initialize state
    double initial_state[4] = {0.0, 0.0, 0.0, 0.0}; // Poor initial guess
    kalman_initialize_state(kf, initial_state);
    
    // State transition matrix (2D constant velocity)
    double **F = matrix_allocate(4, 4);
    F[0][0] = 1.0; F[0][1] = 0.0; F[0][2] = dt;  F[0][3] = 0.0;
    F[1][0] = 0.0; F[1][1] = 1.0; F[1][2] = 0.0; F[1][3] = dt;
    F[2][0] = 0.0; F[2][1] = 0.0; F[2][2] = 1.0; F[2][3] = 0.0;
    F[3][0] = 0.0; F[3][1] = 0.0; F[3][2] = 0.0; F[3][3] = 1.0;
    kalman_set_state_transition_matrix(kf, F);
    
    // Observation matrix (measure both positions)
    double **H = matrix_allocate(2, 4);
    H[0][0] = 1.0; H[0][1] = 0.0; H[0][2] = 0.0; H[0][3] = 0.0;
    H[1][0] = 0.0; H[1][1] = 1.0; H[1][2] = 0.0; H[1][3] = 0.0;
    kalman_set_observation_matrix(kf, H);
    
    // Process noise covariance
    double q = process_noise * process_noise;
    double **Q = matrix_allocate(4, 4);
    Q[0][0] = q * dt * dt; Q[0][1] = 0.0;         Q[0][2] = q * dt; Q[0][3] = 0.0;
    Q[1][0] = 0.0;         Q[1][1] = q * dt * dt; Q[1][2] = 0.0;   Q[1][3] = q * dt;
    Q[2][0] = q * dt;      Q[2][1] = 0.0;         Q[2][2] = q;      Q[2][3] = 0.0;
    Q[3][0] = 0.0;         Q[3][1] = q * dt;      Q[3][2] = 0.0;    Q[3][3] = q;
    kalman_set_process_noise_covariance(kf, Q);
    
    // Measurement noise covariance
    double **R = matrix_allocate(2, 2);
    R[0][0] = measurement_noise * measurement_noise; R[0][1] = 0.0;
    R[1][0] = 0.0; R[1][1] = measurement_noise * measurement_noise;
    kalman_set_measurement_noise_covariance(kf, R);
    
    // Initial error covariance
    double **P = matrix_allocate(4, 4);
    P[0][0] = 5.0; P[0][1] = 0.0; P[0][2] = 0.0; P[0][3] = 0.0;
    P[1][0] = 0.0; P[1][1] = 5.0; P[1][2] = 0.0; P[1][3] = 0.0;
    P[2][0] = 0.0; P[2][1] = 0.0; P[2][2] = 5.0; P[2][3] = 0.0;
    P[3][0] = 0.0; P[3][1] = 0.0; P[3][2] = 0.0; P[3][3] = 5.0;
    kalman_set_error_covariance(kf, P);
    
    // Initialize random seed
    srand(123);
    
    printf("\nStep | True Pos (x,y) | Meas Pos (x,y) | Est Pos (x,y) | Est Vel (x,y) | Pos Error\n");
    printf("-----+----------------+----------------+---------------+---------------+----------\n");
    
    double true_x = 0.0, true_y = 0.0;
    double final_position_error = 0.0;
    
    // Simulation loop
    for (int step = 0; step < num_steps; ++step) {
        // Simulate true system
        true_x += true_vx * dt;
        true_y += true_vy * dt;
        
        // Generate noisy measurements
        double noise_x = ((double)rand() / RAND_MAX - 0.5) * 2.0 * measurement_noise;
        double noise_y = ((double)rand() / RAND_MAX - 0.5) * 2.0 * measurement_noise;
        double measured_x = true_x + noise_x;
        double measured_y = true_y + noise_y;
        
        // Kalman filter steps
        kalman_predict(kf, NULL);
        double measurement[2] = {measured_x, measured_y};
        kalman_update(kf, measurement);
        
        // Get estimates
        double *state = kalman_get_state(kf);
        double est_x = state[0], est_y = state[1];
        double est_vx = state[2], est_vy = state[3];
        
        // Calculate position error
        double pos_error = sqrt(pow(est_x - true_x, 2) + pow(est_y - true_y, 2));
        
        if (step == num_steps - 1) {
            final_position_error = pos_error;
        }
        
        // Print results every 5 steps or first/last few
        if (step % 5 == 0 || step < 2 || step >= num_steps - 2) {
            printf("%4d | (%5.2f,%5.2f) | (%5.2f,%5.2f) | (%5.2f,%5.2f) | (%5.2f,%5.2f) | %9.3f\n",
                   step, true_x, true_y, measured_x, measured_y, 
                   est_x, est_y, est_vx, est_vy, pos_error);
        }
    }
    
    // Cleanup
    matrix_free(F, 4);
    matrix_free(H, 2);
    matrix_free(Q, 4);
    matrix_free(R, 2);
    matrix_free(P, 4);
    kalman_destroy(kf);
    
    // Validate convergence
    int test_passed = (final_position_error < 1.0);
    
    print_test_result("2D tracking convergence", test_passed);
    printf("Final position error: %.3f (threshold: 1.0)\n", final_position_error);
    
    return test_passed;
}

/**
 * @brief Test 2D tracking with different motion patterns
 */
int test_2d_motion_patterns(void) {
    print_test_header("2D Tracking - Motion Patterns");
    
    const double dt = 0.1;
    const int num_steps = 25;
    
    // Test different velocity combinations
    struct {
        double vx, vy;
        const char* description;
    } patterns[] = {
        {1.0, 0.0, "Horizontal motion"},
        {0.0, 1.0, "Vertical motion"},
        {1.0, 1.0, "Diagonal motion"},
        {-1.0, 0.5, "Mixed direction"}
    };
    
    int num_patterns = sizeof(patterns) / sizeof(patterns[0]);
    int all_patterns_passed = 1;
    
    printf("\nPattern | Final Pos Error | Final Vel Error | Status\n");
    printf("--------+-----------------+-----------------+--------\n");
    
    for (int i = 0; i < num_patterns; i++) {
        double true_vx = patterns[i].vx;
        double true_vy = patterns[i].vy;
        
        kalman_t *kf = kalman_create(4, 2, 0);
        if (!kf) {
            all_patterns_passed = 0;
            continue;
        }
        
        // Initialize
        double initial_state[4] = {0.0, 0.0, 0.0, 0.0};
        kalman_initialize_state(kf, initial_state);
        
        // Standard 2D setup
        double **F = matrix_allocate(4, 4);
        F[0][0] = 1.0; F[0][1] = 0.0; F[0][2] = dt;  F[0][3] = 0.0;
        F[1][0] = 0.0; F[1][1] = 1.0; F[1][2] = 0.0; F[1][3] = dt;
        F[2][0] = 0.0; F[2][1] = 0.0; F[2][2] = 1.0; F[2][3] = 0.0;
        F[3][0] = 0.0; F[3][1] = 0.0; F[3][2] = 0.0; F[3][3] = 1.0;
        kalman_set_state_transition_matrix(kf, F);
        
        double **H = matrix_allocate(2, 4);
        H[0][0] = 1.0; H[0][1] = 0.0; H[0][2] = 0.0; H[0][3] = 0.0;
        H[1][0] = 0.0; H[1][1] = 1.0; H[1][2] = 0.0; H[1][3] = 0.0;
        kalman_set_observation_matrix(kf, H);
        
        double **Q = matrix_allocate(4, 4);
        double q = 0.01;
        Q[0][0] = q * dt * dt; Q[0][1] = 0.0;         Q[0][2] = q * dt; Q[0][3] = 0.0;
        Q[1][0] = 0.0;         Q[1][1] = q * dt * dt; Q[1][2] = 0.0;   Q[1][3] = q * dt;
        Q[2][0] = q * dt;      Q[2][1] = 0.0;         Q[2][2] = q;      Q[2][3] = 0.0;
        Q[3][0] = 0.0;         Q[3][1] = q * dt;      Q[3][2] = 0.0;    Q[3][3] = q;
        kalman_set_process_noise_covariance(kf, Q);
        
        double **R = matrix_allocate(2, 2);
        R[0][0] = 0.1; R[0][1] = 0.0;
        R[1][0] = 0.0; R[1][1] = 0.1;
        kalman_set_measurement_noise_covariance(kf, R);
        
        double **P = matrix_allocate(4, 4);
        P[0][0] = 1.0; P[0][1] = 0.0; P[0][2] = 0.0; P[0][3] = 0.0;
        P[1][0] = 0.0; P[1][1] = 1.0; P[1][2] = 0.0; P[1][3] = 0.0;
        P[2][0] = 0.0; P[2][1] = 0.0; P[2][2] = 1.0; P[2][3] = 0.0;
        P[3][0] = 0.0; P[3][1] = 0.0; P[3][2] = 0.0; P[3][3] = 1.0;
        kalman_set_error_covariance(kf, P);
        
        // Simulate
        srand(200 + i);
        double true_x = 0.0, true_y = 0.0;
        
        for (int step = 0; step < num_steps; ++step) {
            true_x += true_vx * dt;
            true_y += true_vy * dt;
            
            double noise_x = ((double)rand() / RAND_MAX - 0.5) * 0.6;
            double noise_y = ((double)rand() / RAND_MAX - 0.5) * 0.6;
            double measured_x = true_x + noise_x;
            double measured_y = true_y + noise_y;
            
            kalman_predict(kf, NULL);
            double measurement[2] = {measured_x, measured_y};
            kalman_update(kf, measurement);
        }
        
        double *final_state = kalman_get_state(kf);
        double pos_error = sqrt(pow(final_state[0] - true_x, 2) + pow(final_state[1] - true_y, 2));
        double vel_error = sqrt(pow(final_state[2] - true_vx, 2) + pow(final_state[3] - true_vy, 2));
        
        int pattern_passed = (pos_error < 0.5 && vel_error < 0.3);
        if (!pattern_passed) {
            all_patterns_passed = 0;
        }
        
        printf("%-7s | %15.4f | %15.4f | %s\n", 
               patterns[i].description, pos_error, vel_error, 
               pattern_passed ? "PASS" : "FAIL");
        
        // Cleanup
        matrix_free(F, 4);
        matrix_free(H, 2);
        matrix_free(Q, 4);
        matrix_free(R, 2);
        matrix_free(P, 4);
        kalman_destroy(kf);
    }
    
    print_test_result("Motion patterns robustness", all_patterns_passed);
    
    return all_patterns_passed;
}

/**
 * @brief Main test function
 */
int main(void) {
    printf(BOLD "\n2D Kalman Filter Tracking Test Suite\n" RESET);
    printf("=====================================\n");
    
    // Run all tests
    test_2d_tracking();
    test_2d_motion_patterns();
    
    // Print summary
    printf(BOLD "\n=== TEST SUMMARY ===" RESET "\n");
    printf("Tests passed: " GREEN "%d" RESET "\n", tests_passed);
    printf("Tests failed: " RED "%d" RESET "\n", tests_failed);
    printf("Total tests:  %d\n", tests_passed + tests_failed);
    
    if (tests_failed == 0) {
        printf(GREEN BOLD "\nAll 2D tracking tests PASSED! ✓" RESET "\n");
        return 0;
    } else {
        printf(RED BOLD "\nSome 2D tracking tests FAILED! ✗" RESET "\n");
        return 1;
    }
} 
