/**
 * @file test_kalman_noise.c
 * @brief Noise sensitivity tests for the standard Kalman Filter implementation
 * @author ronnymilleo
 * @date 12/07/25
 * @version 1.0
 *
 * This file contains tests for noise sensitivity analysis using the
 * standard Kalman filter, focusing on robustness under different
 * noise conditions.
 * 
 * @details Test scenarios:
 * - Performance analysis with different noise levels
 * - Convergence quality assessment
 * - Robustness validation under varying conditions
 * - Error threshold analysis
 * 
 * @note Tests demonstrate filter adaptation to different noise environments
 * @note Uses 1D constant velocity model for noise analysis
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
 * @brief Test process noise sensitivity
 */
int test_process_noise_sensitivity(void) {
    print_test_header("Process Noise Sensitivity");
    
    const double dt = 0.1;
    const double true_velocity = 1.0;
    const int num_steps = 25;
    
    // Test different process noise levels
    double process_noise_levels[] = {0.01, 0.05, 0.1, 0.2, 0.5};
    int num_levels = sizeof(process_noise_levels) / sizeof(process_noise_levels[0]);
    
    printf("\nProcess Noise | Final Vel Error | Filter Responsiveness\n");
    printf("--------------+-----------------+----------------------\n");
    
    int all_levels_passed = 1;
    
    for (int i = 0; i < num_levels; i++) {
        double process_noise = process_noise_levels[i];
        
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
        
        // Process noise covariance
        double **Q = matrix_allocate(2, 2);
        Q[0][0] = process_noise * process_noise * dt * dt;
        Q[0][1] = process_noise * process_noise * dt;
        Q[1][0] = process_noise * process_noise * dt;
        Q[1][1] = process_noise * process_noise;
        kalman_set_process_noise_covariance(kf, Q);
        
        double **R = matrix_allocate(1, 1);
        R[0][0] = 0.25; // Fixed measurement noise
        kalman_set_measurement_noise_covariance(kf, R);
        
        double **P = matrix_allocate(2, 2);
        P[0][0] = 1.0; P[0][1] = 0.0;
        P[1][0] = 0.0; P[1][1] = 1.0;
        kalman_set_error_covariance(kf, P);
        
        // Simulate
        srand(300 + i);
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
        
        // Determine responsiveness (lower process noise should give lower error)
        const char* responsiveness = (vel_error < 0.2) ? "High" : 
                                   (vel_error < 0.5) ? "Medium" : "Low";
        
        printf("%13.3f | %15.4f | %20s\n", process_noise, vel_error, responsiveness);
        
        if (vel_error > 1.0) {
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
    
    print_test_result("Process noise sensitivity", all_levels_passed);
    
    return all_levels_passed;
}

/**
 * @brief Test measurement noise vs process noise trade-off
 */
int test_noise_tradeoff(void) {
    print_test_header("Measurement vs Process Noise Trade-off");
    
    const double dt = 0.1;
    const double true_velocity = 1.0;
    const int num_steps = 30;
    
    // Test different combinations
    struct {
        double measurement_noise, process_noise;
        const char* description;
    } combinations[] = {
        {0.1, 0.01, "Low meas, Low proc"},
        {0.1, 0.1,  "Low meas, High proc"},
        {1.0, 0.01, "High meas, Low proc"},
        {1.0, 0.1,  "High meas, High proc"}
    };
    
    int num_combinations = sizeof(combinations) / sizeof(combinations[0]);
    int all_combinations_passed = 1;
    
    printf("\nCombination | Final Vel Error | Final Pos Error | Performance\n");
    printf("------------+-----------------+-----------------+-------------\n");
    
    for (int i = 0; i < num_combinations; i++) {
        double meas_noise = combinations[i].measurement_noise;
        double proc_noise = combinations[i].process_noise;
        
        kalman_t *kf = kalman_create(2, 1, 0);
        if (!kf) {
            all_combinations_passed = 0;
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
        
        // Process noise covariance
        double **Q = matrix_allocate(2, 2);
        Q[0][0] = proc_noise * proc_noise * dt * dt;
        Q[0][1] = proc_noise * proc_noise * dt;
        Q[1][0] = proc_noise * proc_noise * dt;
        Q[1][1] = proc_noise * proc_noise;
        kalman_set_process_noise_covariance(kf, Q);
        
        double **R = matrix_allocate(1, 1);
        R[0][0] = meas_noise * meas_noise;
        kalman_set_measurement_noise_covariance(kf, R);
        
        double **P = matrix_allocate(2, 2);
        P[0][0] = 1.0; P[0][1] = 0.0;
        P[1][0] = 0.0; P[1][1] = 1.0;
        kalman_set_error_covariance(kf, P);
        
        // Simulate
        srand(400 + i);
        double true_position = 0.0;
        
        for (int step = 0; step < num_steps; ++step) {
            true_position += true_velocity * dt;
            double noise = ((double)rand() / RAND_MAX - 0.5) * 2.0 * meas_noise;
            double measured_position = true_position + noise;
            
            kalman_predict(kf, NULL);
            double measurement[1] = {measured_position};
            kalman_update(kf, measurement);
        }
        
        double *final_state = kalman_get_state(kf);
        double vel_error = fabs(final_state[1] - true_velocity);
        double pos_error = fabs(final_state[0] - true_position);
        
        // Determine performance
        const char* performance = (vel_error < 0.3 && pos_error < 0.5) ? "Excellent" :
                                (vel_error < 0.6 && pos_error < 1.0) ? "Good" :
                                (vel_error < 1.0 && pos_error < 2.0) ? "Fair" : "Poor";
        
        printf("%-11s | %15.4f | %15.4f | %11s\n", 
               combinations[i].description, vel_error, pos_error, performance);
        
        if (vel_error > 1.5 || pos_error > 3.0) {
            all_combinations_passed = 0;
        }
        
        // Cleanup
        matrix_free(F, 2);
        matrix_free(H, 1);
        matrix_free(Q, 2);
        matrix_free(R, 1);
        matrix_free(P, 2);
        kalman_destroy(kf);
    }
    
    print_test_result("Noise trade-off analysis", all_combinations_passed);
    
    return all_combinations_passed;
}

/**
 * @brief Main test function
 */
int main(void) {
    printf(BOLD "\nKalman Filter Noise Sensitivity Test Suite\n" RESET);
    printf("============================================\n");
    
    // Run all tests
    test_noise_sensitivity();
    test_process_noise_sensitivity();
    test_noise_tradeoff();
    
    // Print summary
    printf(BOLD "\n=== TEST SUMMARY ===" RESET "\n");
    printf("Tests passed: " GREEN "%d" RESET "\n", tests_passed);
    printf("Tests failed: " RED "%d" RESET "\n", tests_failed);
    printf("Total tests:  %d\n", tests_passed + tests_failed);
    
    if (tests_failed == 0) {
        printf(GREEN BOLD "\nAll noise sensitivity tests PASSED! ✓" RESET "\n");
        return 0;
    } else {
        printf(RED BOLD "\nSome noise sensitivity tests FAILED! ✗" RESET "\n");
        return 1;
    }
} 
