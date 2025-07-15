/**
 * @file test_kalman_1d.c
 * @brief 1D tracking tests for the standard Kalman Filter implementation
 * @author ronnymilleo
 * @date 12/07/25
 * @version 1.0
 *
 * This file contains tests for 1D position and velocity tracking using
 * the standard Kalman filter, focusing on convergence and accuracy
 * validation.
 * 
 * @details Test scenarios:
 * - 1D position and velocity tracking with constant velocity model
 * - Convergence validation with poor initial estimates
 * - Performance analysis with noisy measurements
 * - Error threshold validation
 * 
 * @note Tests demonstrate practical 1D tracking applications
 * @note Uses constant velocity motion model with position measurements
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
 * @brief Test 1D tracking with different initial conditions
 */
int test_1d_initial_conditions(void) {
    print_test_header("1D Tracking - Initial Conditions");
    
    const double dt = 0.1;
    const double true_velocity = 1.5;
    const int num_steps = 30;
    
    // Test different initial velocity estimates
    double initial_velocities[] = {0.0, 0.5, 1.0, 2.0, 3.0};
    int num_tests = sizeof(initial_velocities) / sizeof(initial_velocities[0]);
    
    printf("\nInitial Vel | Final Vel Error | Convergence\n");
    printf("------------+-----------------+-------------\n");
    
    int all_tests_passed = 1;
    
    for (int i = 0; i < num_tests; i++) {
        double init_vel = initial_velocities[i];
        
        kalman_t *kf = kalman_create(2, 1, 0);
        if (!kf) {
            all_tests_passed = 0;
            continue;
        }
        
        // Initialize with different velocity estimate
        double initial_state[2] = {0.0, init_vel};
        kalman_initialize_state(kf, initial_state);
        
        // Standard setup
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
        R[0][0] = 0.25;
        kalman_set_measurement_noise_covariance(kf, R);
        
        double **P = matrix_allocate(2, 2);
        P[0][0] = 1.0; P[0][1] = 0.0;
        P[1][0] = 0.0; P[1][1] = 1.0;
        kalman_set_error_covariance(kf, P);
        
        // Simulate
        srand(100 + i);
        double true_position = 0.0;
        
        for (int step = 0; step < num_steps; ++step) {
            true_position += true_velocity * dt;
            double noise = ((double)rand() / RAND_MAX - 0.5) * 1.0;
            double measured_position = true_position + noise;
            
            kalman_predict(kf, NULL);
            double measurement[1] = {measured_position};
            kalman_update(kf, measurement);
        }
        
        double *final_state = kalman_get_state(kf);
        double vel_error = fabs(final_state[1] - true_velocity);
        
        printf("%11.1f | %15.4f | %11s\n", 
               init_vel, vel_error, (vel_error < 0.3) ? "Good" : "Poor");
        
        if (vel_error > 0.5) {
            all_tests_passed = 0;
        }
        
        // Cleanup
        matrix_free(F, 2);
        matrix_free(H, 1);
        matrix_free(Q, 2);
        matrix_free(R, 1);
        matrix_free(P, 2);
        kalman_destroy(kf);
    }
    
    print_test_result("Initial conditions robustness", all_tests_passed);
    
    return all_tests_passed;
}

/**
 * @brief Main test function
 */
int main(void) {
    printf(BOLD "\n1D Kalman Filter Tracking Test Suite\n" RESET);
    printf("=====================================\n");
    
    // Run all tests
    test_1d_tracking();
    test_1d_initial_conditions();
    
    // Print summary
    printf(BOLD "\n=== TEST SUMMARY ===" RESET "\n");
    printf("Tests passed: " GREEN "%d" RESET "\n", tests_passed);
    printf("Tests failed: " RED "%d" RESET "\n", tests_failed);
    printf("Total tests:  %d\n", tests_passed + tests_failed);
    
    if (tests_failed == 0) {
        printf(GREEN BOLD "\nAll 1D tracking tests PASSED! ✓" RESET "\n");
        return 0;
    } else {
        printf(RED BOLD "\nSome 1D tracking tests FAILED! ✗" RESET "\n");
        return 1;
    }
}
