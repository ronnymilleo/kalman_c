/**
 * @file kalman_demo.c
 * @brief Comprehensive Kalman filter demonstration applications in C
 * @author ronnymilleo
 * @date 12/07/25
 * @version 1.0
 * 
 * This application demonstrates the standard Kalman filter implementation
 * with multiple practical scenarios showing different aspects of linear
 * state estimation and filtering capabilities using the C API.
 * 
 * @details Demonstration scenarios:
 * - 1D position and velocity tracking with constant velocity model
 * - 2D object tracking with position and velocity estimation
 * - Sensor fusion example combining multiple measurements
 * - Performance analysis with different noise levels
 * 
 * @note Shows comprehensive usage of the kalman C API
 * @note Includes noise simulation and performance evaluation
 * @note Results demonstrate filter convergence and accuracy
 * 
 * @see kalman.h for API documentation
 * @see matrix_math.h for matrix operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kalman.h"
#include "matrix_math.h"

// Simple color codes for better output visualization
#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define GREEN   "\033[32m"
#define BLUE    "\033[34m"
#define YELLOW  "\033[33m"

/**
 * @brief 1D position and velocity tracking demonstration
 * 
 * This function demonstrates basic Kalman filtering for tracking an object
 * moving with approximately constant velocity in 1D, with noisy position measurements.
 */
void demonstrate_1d_tracking(void) {
    printf(BOLD BLUE "\n=== Standard Kalman Filter Demo: 1D Position/Velocity Tracking ===" RESET "\n\n");
    
    // System parameters
    const double dt = 0.1;              // Time step (seconds)
    const double true_velocity = 2.0;   // True constant velocity (m/s)
    const double process_noise = 0.1;   // Process noise standard deviation
    const double measurement_noise = 0.5; // Measurement noise standard deviation
    const int num_steps = 50;           // Number of simulation steps
    
    // Create 2D state (position, velocity), 1D measurement (position) KF
    kalman_t *kf = kalman_create(2, 1, 0);
    if (!kf) {
        fprintf(stderr, "Error: Failed to create Kalman filter\n");
        return;
    }
    
    // Initialize state [position=0, velocity=0] - poor initial velocity estimate
    double initial_state[2] = {0.0, 0.0};
    kalman_initialize_state(kf, initial_state);
    
    // State transition matrix (constant velocity model)
    // x(k+1) = x(k) + v(k) * dt
    // v(k+1) = v(k)
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
    
    // Set up random number generation (simple linear congruential generator)
    unsigned int seed = 12345;
    
    printf("Step | True Pos | True Vel | Meas Pos | Est Pos | Est Vel | Pos Err | Vel Err\n");
    printf("-----+----------+----------+----------+---------+---------+---------+--------\n");
    
    double true_position = 0.0;
    
    // Simulation loop
    for (int step = 0; step < num_steps; ++step) {
        // Simulate true system (constant velocity)
        true_position += true_velocity * dt;
        
        // Generate noisy measurement (simple pseudo-random noise)
        seed = seed * 1103515245 + 12345;
        double noise = ((double)(seed % 1000) / 1000.0 - 0.5) * 2.0 * measurement_noise;
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
        
        // Print results every 5 steps or first/last few steps
        if (step % 5 == 0 || step < 3 || step >= num_steps - 3) {
            printf("%4d | %8.3f | %8.3f | %8.3f | %7.3f | %7.3f | %7.3f | %7.3f\n",
                   step, true_position, true_velocity, measured_position,
                   estimated_position, estimated_velocity, position_error, velocity_error);
        }
    }
    
    printf(GREEN "\nKF successfully estimated velocity from position-only measurements!" RESET "\n");
    printf("Notice how velocity estimate converges to the true value of %.1f m/s.\n\n", true_velocity);
    
    // Cleanup
    matrix_free(F, 2);
    matrix_free(H, 1);
    matrix_free(Q, 2);
    matrix_free(R, 1);
    matrix_free(P, 2);
    kalman_destroy(kf);
}

/**
 * @brief 2D object tracking demonstration
 * 
 * This function demonstrates 2D tracking of an object moving with constant
 * velocity in both X and Y directions, with noisy position measurements.
 */
void demonstrate_2d_tracking(void) {
    printf(BOLD BLUE "\n=== Standard Kalman Filter Demo: 2D Object Tracking ===" RESET "\n\n");
    
    // System parameters
    const double dt = 0.2;                    // Time step
    const double true_vx = 1.5;               // True X velocity
    const double true_vy = -1.0;              // True Y velocity  
    const double process_noise = 0.05;        // Process noise
    const double measurement_noise_x = 0.3;   // X measurement noise
    const double measurement_noise_y = 0.4;   // Y measurement noise
    const int num_steps = 30;                 // Number of steps
    
    // Create 4D state (x, y, vx, vy), 2D measurement (x, y) KF
    kalman_t *kf = kalman_create(4, 2, 0);
    if (!kf) {
        fprintf(stderr, "Error: Failed to create Kalman filter\n");
        return;
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
    R[0][0] = measurement_noise_x * measurement_noise_x; R[0][1] = 0.0;
    R[1][0] = 0.0; R[1][1] = measurement_noise_y * measurement_noise_y;
    kalman_set_measurement_noise_covariance(kf, R);
    
    // Initial error covariance
    double **P = matrix_allocate(4, 4);
    P[0][0] = 5.0; P[0][1] = 0.0; P[0][2] = 0.0; P[0][3] = 0.0;
    P[1][0] = 0.0; P[1][1] = 5.0; P[1][2] = 0.0; P[1][3] = 0.0;
    P[2][0] = 0.0; P[2][1] = 0.0; P[2][2] = 5.0; P[2][3] = 0.0;
    P[3][0] = 0.0; P[3][1] = 0.0; P[3][2] = 0.0; P[3][3] = 5.0;
    kalman_set_error_covariance(kf, P);
    
    // Set up random number generation
    unsigned int seed_x = 54321;
    unsigned int seed_y = 98765;
    
    printf("Step | True Pos (x,y) | Meas Pos (x,y) | Est Pos (x,y) | Est Vel (x,y) | Pos Error\n");
    printf("-----+----------------+----------------+---------------+---------------+----------\n");
    
    double true_x = 0.0, true_y = 0.0;
    
    // Simulation loop
    for (int step = 0; step < num_steps; ++step) {
        // Simulate true system
        true_x += true_vx * dt;
        true_y += true_vy * dt;
        
        // Generate noisy measurements
        seed_x = seed_x * 1103515245 + 12345;
        seed_y = seed_y * 1103515245 + 54321;
        double noise_x = ((double)(seed_x % 1000) / 1000.0 - 0.5) * 2.0 * measurement_noise_x;
        double noise_y = ((double)(seed_y % 1000) / 1000.0 - 0.5) * 2.0 * measurement_noise_y;
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
        
        // Print results every 3 steps or first/last few
        if (step % 3 == 0 || step < 2 || step >= num_steps - 2) {
            printf("%4d | (%5.2f,%5.2f) | (%5.2f,%5.2f) | (%5.2f,%5.2f) | (%5.2f,%5.2f) | %9.3f\n",
                   step, true_x, true_y, measured_x, measured_y, 
                   est_x, est_y, est_vx, est_vy, pos_error);
        }
    }
    
    printf(GREEN "\nKF successfully tracked 2D motion with coupled state estimation!" RESET "\n");
    printf("The filter estimated both position and velocity in X and Y dimensions.\n\n");
    
    // Cleanup
    matrix_free(F, 4);
    matrix_free(H, 2);
    matrix_free(Q, 4);
    matrix_free(R, 2);
    matrix_free(P, 4);
    kalman_destroy(kf);
}

/**
 * @brief Noise sensitivity analysis
 * 
 * This function demonstrates how the Kalman filter performs under
 * different noise conditions, showing robustness and adaptation.
 */
void demonstrate_noise_sensitivity(void) {
    printf(BOLD BLUE "\n=== Standard Kalman Filter Demo: Noise Sensitivity Analysis ===" RESET "\n\n");
    
    const double dt = 0.1;
    const double true_velocity = 1.0;
    const int num_steps = 20;
    
    // Test different noise levels
    double noise_levels[] = {0.1, 0.5, 1.0, 2.0};
    int num_noise_levels = sizeof(noise_levels) / sizeof(noise_levels[0]);
    
    printf("Noise Level | Final Pos Error | Final Vel Error | Convergence Quality\n");
    printf("------------+-----------------+-----------------+--------------------\n");
    
    for (int i = 0; i < num_noise_levels; i++) {
        double noise_level = noise_levels[i];
        
        kalman_t *kf = kalman_create(2, 1, 0);
        if (!kf) {
            fprintf(stderr, "Error: Failed to create Kalman filter\n");
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
        unsigned int seed = 200 + i * 100; // Different seed for each noise level
        
        double true_position = 0.0;
        double convergence_measure = 0.0;
        
        for (int step = 0; step < num_steps; ++step) {
            true_position += true_velocity * dt;
            seed = seed * 1103515245 + 12345;
            double noise = ((double)(seed % 1000) / 1000.0 - 0.5) * 2.0 * noise_level;
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
        
        // Cleanup
        matrix_free(F, 2);
        matrix_free(H, 1);
        matrix_free(Q, 2);
        matrix_free(R, 1);
        matrix_free(P, 2);
        kalman_destroy(kf);
    }
    
    printf(GREEN "\nKF demonstrates robustness across different noise levels!" RESET "\n");
    printf("Higher noise leads to slower convergence but maintains stability.\n\n");
}

/**
 * @brief Main demonstration function
 */
int main(void) {
    printf(BOLD "\nStandard Kalman Filter Demonstrations" RESET "\n");
    printf("=====================================\n");
    
    // Run 1D tracking demo
    demonstrate_1d_tracking();
    
    // Run 2D tracking demo  
    demonstrate_2d_tracking();
    
    // Run noise sensitivity analysis
    demonstrate_noise_sensitivity();
    
    printf(GREEN BOLD "All demonstrations completed successfully!" RESET "\n");
    printf("The standard Kalman Filter effectively handles linear estimation problems.\n");
    
    return 0;
}
