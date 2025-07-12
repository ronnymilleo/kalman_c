/**
 * @file test_kalman.c
 * @brief Comprehensive unit tests for the standard Kalman Filter implementation
 * @author ronnymilleo
 * @date 12/07/25
 * @version 1.0
 *
 * This file contains comprehensive unit tests for the standard Kalman filter
 * functionality, covering multiple practical scenarios to validate linear
 * state estimation and filtering capabilities.
 * 
 * @details Test scenarios:
 * - 1D position and velocity tracking with constant velocity model
 * - 2D object tracking with position and velocity estimation
 * - Sensor fusion example combining multiple measurements
 * - Performance analysis with different noise levels
 * - Basic functionality tests (predict, update, initialization)
 * 
 * @note Tests demonstrate comprehensive usage of the kalman C API
 * @note Includes noise simulation and performance evaluation
 * @note Results validate filter convergence and accuracy
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
 * @brief Test 1D position and velocity tracking
 * 
 * This test demonstrates basic Kalman filtering for tracking an object
 * moving with approximately constant velocity in 1D, with noisy position measurements.
 */
int test_1d_tracking(void) {
    print_test_header("1D Position/Velocity Tracking");
    
    // System parameters
    const double dt = 0.1;              // Time step (seconds)
    const double true_velocity = 2.0;   // True constant velocity (m/s)
    const double process_noise = 0.1;   // Process noise standard deviation
    const double measurement_noise = 0.5; // Measurement noise standard deviation
    const int num_steps = 50;           // Number of simulation steps
    
    // Create 2D state (position, velocity), 1D measurement (position) KF
    kalman_t *kf = kalman_create(2, 1, 0);
    if (!kf) {
        print_test_result("1D tracking - filter creation", 0);
        return 0;
    }
    
    // Initialize state [position=0, velocity=0] - poor initial velocity estimate
    double initial_state[2] = {0.0, 0.0};
    kalman_initialize_state(kf, initial_state);
    
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
    Q[0][0] = process_noise * process_noise * dt * dt;
    Q[0][1] = process_noise * process_noise * dt;
    Q[1][0] = process_noise * process_noise * dt;
    Q[1][1] = process_noise * process_noise;
    kalman_set_process_noise_covariance(kf, Q);
    
    // Measurement noise covariance
    double **R = matrix_allocate(1, 1);
    R[0][0] = measurement_noise * measurement_noise;
    kalman_set_measurement_noise_covariance(kf, R);
    
    // Initial error covariance (high uncertainty)
    double **P = matrix_allocate(2, 2);
    P[0][0] = 10.0; P[0][1] = 0.0;
    P[1][0] = 0.0;  P[1][1] = 10.0;
    kalman_set_error_covariance(kf, P);
    
    // Initialize random seed for reproducible tests
    srand(42);
    
    printf("\nStep | True Pos | True Vel | Meas Pos | Est Pos | Est Vel | Pos Err | Vel Err\n");
    printf("-----+----------+----------+----------+---------+---------+---------+--------\n");
    
    double true_position = 0.0;
    double final_velocity_error = 0.0;
    double final_position_error = 0.0;
    
    // Simulation loop
    for (int step = 0; step < num_steps; ++step) {
        // Simulate true system (constant velocity)
        true_position += true_velocity * dt;
        
        // Generate noisy measurement (simplified noise)
        double noise = ((double)rand() / RAND_MAX - 0.5) * 2.0 * measurement_noise;
        double measured_position = true_position + noise;
        
        // Kalman filter predict step
        kalman_predict(kf, NULL);
        
        // Kalman filter update step
        double measurement[1] = {measured_position};
        kalman_update(kf, measurement);
        
        // Get current estimates
        double *state = kalman_get_state(kf);
        double estimated_position = state[0];
        double estimated_velocity = state[1];
        
        // Calculate errors
        double position_error = fabs(estimated_position - true_position);
        double velocity_error = fabs(estimated_velocity - true_velocity);
        
        // Store final errors for validation
        if (step == num_steps - 1) {
            final_position_error = position_error;
            final_velocity_error = velocity_error;
        }
        
        // Print results every 10 steps or first/last few steps
        if (step % 10 == 0 || step < 3 || step >= num_steps - 3) {
            printf("%4d | %8.3f | %8.3f | %8.3f | %7.3f | %7.3f | %7.3f | %7.3f\n",
                   step, true_position, true_velocity, measured_position,
                   estimated_position, estimated_velocity, position_error, velocity_error);
        }
    }
    
    // Cleanup
    matrix_free(F, 2);
    matrix_free(H, 1);
    matrix_free(Q, 2);
    matrix_free(R, 1);
    matrix_free(P, 2);
    kalman_destroy(kf);
    
    // Validate convergence (final velocity estimate should be reasonably close)
    int test_passed = (final_velocity_error < 0.5 && final_position_error < 2.0);
    
    print_test_result("1D tracking convergence", test_passed);
    printf("Final velocity error: %.3f (threshold: 0.5)\n", final_velocity_error);
    printf("Final position error: %.3f (threshold: 2.0)\n", final_position_error);
    
    return test_passed;
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
 * @brief Test noise sensitivity analysis
 * 
 * This test demonstrates how the Kalman filter performs under
 * different noise conditions, showing robustness and adaptation.
 */
int test_noise_sensitivity(void) {
    print_test_header("Noise Sensitivity Analysis");
    
    const double dt = 0.1;
    const double true_velocity = 1.0;
    const int num_steps = 20;
    
    // Test different noise levels
    double noise_levels[] = {0.1, 0.5, 1.0, 2.0};
    int num_noise_levels = sizeof(noise_levels) / sizeof(noise_levels[0]);
    
    printf("\nNoise Level | Final Pos Error | Final Vel Error | Convergence Quality\n");
    printf("------------+-----------------+-----------------+--------------------\n");
    
    int all_levels_passed = 1;
    
    for (int i = 0; i < num_noise_levels; i++) {
        double noise_level = noise_levels[i];
        
        kalman_t *kf = kalman_create(2, 1, 0);
        if (!kf) {
            all_levels_passed = 0;
            continue;
        }
        
        // Initialize
        double initial_state[2] = {0.0, 0.0};
        kalman_initialize_state(kf, initial_state);
        
        double **F = matrix_allocate(2, 2);
        F[0][0] = 1.0; F[0][1] = dt;
        F[1][0] = 0.0; F[1][1] = 1.0;
        kalman_set_state_transition_matrix(kf, F);
        
        double **H = matrix_allocate(1, 2);
        H[0][0] = 1.0; H[0][1] = 0.0;
        kalman_set_observation_matrix(kf, H);
        
        double **Q = matrix_allocate(2, 2);
        Q[0][0] = 0.01; Q[0][1] = 0.0;
        Q[1][0] = 0.0;  Q[1][1] = 0.01;
        kalman_set_process_noise_covariance(kf, Q);
        
        double **R = matrix_allocate(1, 1);
        R[0][0] = noise_level * noise_level;
        kalman_set_measurement_noise_covariance(kf, R);
        
        double **P = matrix_allocate(2, 2);
        P[0][0] = 1.0; P[0][1] = 0.0;
        P[1][0] = 0.0; P[1][1] = 1.0;
        kalman_set_error_covariance(kf, P);
        
        // Simulate with this noise level
        srand(200 + i); // Different seed for each noise level
        
        double true_position = 0.0;
        double convergence_measure = 0.0;
        
        for (int step = 0; step < num_steps; ++step) {
            true_position += true_velocity * dt;
            double noise = ((double)rand() / RAND_MAX - 0.5) * 2.0 * noise_level;
            double measured_position = true_position + noise;
            
            kalman_predict(kf, NULL);
            double measurement[1] = {measured_position};
            kalman_update(kf, measurement);
            
            // Accumulate convergence measure (sum of velocity estimate errors)
            if (step > 5) { // After initial convergence
                double *state = kalman_get_state(kf);
                convergence_measure += fabs(state[1] - true_velocity);
            }
        }
        
        double *final_state = kalman_get_state(kf);
        double pos_error = fabs(final_state[0] - true_position);
        double vel_error = fabs(final_state[1] - true_velocity);
        convergence_measure /= (num_steps - 5);
        
        printf("%11.1f | %15.4f | %15.4f | %19.4f\n",
               noise_level, pos_error, vel_error, convergence_measure);
        
        // Check if this noise level test passed (reasonable thresholds)
        if (vel_error > 2.0 || convergence_measure > 3.0) {
            all_levels_passed = 0;
        }
        
        // Cleanup
        matrix_free(F, 2);
        matrix_free(H, 1);
        matrix_free(Q, 2);
        matrix_free(R, 1);
        matrix_free(P, 2);
        kalman_destroy(kf);
    }
    
    print_test_result("Noise sensitivity", all_levels_passed);
    
    return all_levels_passed;
}

/**
 * @brief Main test function
 */
int main(void) {
    printf(BOLD "\nStandard Kalman Filter Test Suite\n" RESET);
    printf("=================================\n");
    
    // Run all tests
    test_basic_functionality();
    test_1d_tracking();
    test_2d_tracking();
    test_noise_sensitivity();
    
    // Print summary
    printf(BOLD "\n=== TEST SUMMARY ===" RESET "\n");
    printf("Tests passed: " GREEN "%d" RESET "\n", tests_passed);
    printf("Tests failed: " RED "%d" RESET "\n", tests_failed);
    printf("Total tests:  %d\n", tests_passed + tests_failed);
    
    if (tests_failed == 0) {
        printf(GREEN BOLD "\nAll tests PASSED! ✓" RESET "\n");
        printf("The standard Kalman Filter implementation is working correctly.\n");
        return 0;
    } else {
        printf(RED BOLD "\nSome tests FAILED! ✗" RESET "\n");
        return 1;
    }
}
