/**
 * @file main.c
 * @brief Kalman filter demonstration and example usage
 * @author ronnymilleo
 * @date 08/07/25
 * @version 1.0
 * 
 * This file demonstrates the usage of the Kalman filter library with
 * a practical example of 1D position tracking with velocity estimation.
 * The demo shows how to configure and use the filter for real-time
 * state estimation.
 * 
 * @details Example features:
 * - 1D position tracking with velocity
 * - Constant velocity motion model
 * - Simulated noisy position measurements
 * - Real-time filter performance visualization
 * - Complete setup and cleanup procedures
 * 
 * @note This example uses a simple constant velocity model
 * @note Measurement noise and process noise are configurable
 * @note Color output is used for better visualization
 * 
 * @see kalman.h for API documentation
 * @see matrix_math.h for matrix operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "kalman.h"
#include "matrix_math.h"

// Simple color codes
#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define GREEN   "\033[32m"
#define BLUE    "\033[34m"
#define YELLOW  "\033[33m"

void printHeader(void) {
    printf(BOLD "\n=== KALMAN FILTER DEMO ===" RESET "\n");
    printf("1D Position Tracking with Velocity\n\n");
}

void printSection(const char* title) {
    printf(BLUE "%s" RESET "\n", title);
    size_t len = strlen(title);
    for (size_t i = 0; i < len; i++) {
        printf("-");
    }
    printf("\n");
}

int main(void) {
    printHeader();

    printSection("Filter Initialization");

    // Example: 1D position tracking with velocity
    // State: [position, velocity]
    // Measurement: [position]

    // Create Kalman filter (2D state, 1D measurement, no control)
    kalman_t *kf = kalman_create(2, 1, 0);
    if (!kf) {
        fprintf(stderr, "Error: Failed to create Kalman filter\n");
        return 1;
    }

    // Initialize state [position=0, velocity=0]
    double initialState[2] = {0.0, 0.0};
    if (kalman_initialize_state(kf, initialState) != 0) {
        fprintf(stderr, "Error: Failed to initialize state\n");
        kalman_destroy(kf);
        return 1;
    }

    // State transition matrix (constant velocity model)
    // x(k+1) = x(k) + v(k) * dt
    // v(k+1) = v(k)
    double dt = 1.0; // time step
    double **F = matrix_allocate(2, 2);
    if (!F) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        kalman_destroy(kf);
        return 1;
    }
    F[0][0] = 1.0; F[0][1] = dt;
    F[1][0] = 0.0; F[1][1] = 1.0;
    
    if (kalman_set_state_transition_matrix(kf, F) != 0) {
        fprintf(stderr, "Error: Failed to set state transition matrix\n");
        matrix_free(F, 2);
        kalman_destroy(kf);
        return 1;
    }

    // Observation matrix (we can only measure position)
    double **H = matrix_allocate(1, 2);
    if (!H) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        matrix_free(F, 2);
        kalman_destroy(kf);
        return 1;
    }
    H[0][0] = 1.0; H[0][1] = 0.0;
    
    if (kalman_set_observation_matrix(kf, H) != 0) {
        fprintf(stderr, "Error: Failed to set observation matrix\n");
        matrix_free(F, 2);
        matrix_free(H, 1);
        kalman_destroy(kf);
        return 1;
    }

    // Process noise covariance
    double **Q = matrix_allocate(2, 2);
    if (!Q) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        matrix_free(F, 2);
        matrix_free(H, 1);
        kalman_destroy(kf);
        return 1;
    }
    Q[0][0] = 0.1; Q[0][1] = 0.0;
    Q[1][0] = 0.0; Q[1][1] = 0.1;
    
    if (kalman_set_process_noise_covariance(kf, Q) != 0) {
        fprintf(stderr, "Error: Failed to set process noise covariance\n");
        matrix_free(F, 2);
        matrix_free(H, 1);
        matrix_free(Q, 2);
        kalman_destroy(kf);
        return 1;
    }

    // Measurement noise covariance
    double **R = matrix_allocate(1, 1);
    if (!R) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        matrix_free(F, 2);
        matrix_free(H, 1);
        matrix_free(Q, 2);
        kalman_destroy(kf);
        return 1;
    }
    R[0][0] = 1.0;
    
    if (kalman_set_measurement_noise_covariance(kf, R) != 0) {
        fprintf(stderr, "Error: Failed to set measurement noise covariance\n");
        matrix_free(F, 2);
        matrix_free(H, 1);
        matrix_free(Q, 2);
        matrix_free(R, 1);
        kalman_destroy(kf);
        return 1;
    }

    // Initial error covariance
    double **P = matrix_allocate(2, 2);
    if (!P) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        matrix_free(F, 2);
        matrix_free(H, 1);
        matrix_free(Q, 2);
        matrix_free(R, 1);
        kalman_destroy(kf);
        return 1;
    }
    P[0][0] = 1.0; P[0][1] = 0.0;
    P[1][0] = 0.0; P[1][1] = 1.0;
    
    if (kalman_set_error_covariance(kf, P) != 0) {
        fprintf(stderr, "Error: Failed to set error covariance\n");
        matrix_free(F, 2);
        matrix_free(H, 1);
        matrix_free(Q, 2);
        matrix_free(R, 1);
        matrix_free(P, 2);
        kalman_destroy(kf);
        return 1;
    }

    printf(GREEN "Filter configured successfully!" RESET "\n");
    printf("State: [position, velocity] - Measurement: [position]\n");
    printf("Initial Position: %.3f units\n", kalman_get_state_element(kf, 0));
    printf("Initial Velocity: %.3f units/s\n\n", kalman_get_state_element(kf, 1));

    // Simulate some measurements
    double measurements[] = {1.2, 2.8, 4.1, 5.9, 7.8, 9.2, 11.1, 12.9};
    int numMeasurements = sizeof(measurements) / sizeof(measurements[0]);

    printSection("Processing Measurements");

    printf(BOLD "Step  Measurement  Position  Velocity" RESET "\n");
    printf("----  -----------  --------  --------\n");

    for (int i = 0; i < numMeasurements; i++) {
        // Predict step
        if (kalman_predict(kf, NULL) != 0) {
            fprintf(stderr, "Error: Prediction failed at step %d\n", i + 1);
            break;
        }

        // Update step with measurement
        double z[1] = {measurements[i]};
        if (kalman_update(kf, z) != 0) {
            fprintf(stderr, "Error: Update failed at step %d\n", i + 1);
            break;
        }

        printf("%4d  %11.3f  %8.3f  %8.3f\n", 
               i + 1, measurements[i], 
               kalman_get_state_element(kf, 0), 
               kalman_get_state_element(kf, 1));
    }

    printSection("Final Results");

    // Calculate and display performance metrics
    double finalPos = kalman_get_state_element(kf, 0);
    double finalVel = kalman_get_state_element(kf, 1);
    double lastMeasurement = measurements[numMeasurements - 1];
    double estimationError = fabs(finalPos - lastMeasurement);

    printf(BOLD "Final Estimated State:" RESET "\n");
    printf("Position: %.4f units\n", finalPos);
    printf("Velocity: %.4f units/s\n\n", finalVel);

    printf("Performance:\n");
    printf("Last measurement: %.3f\n", lastMeasurement);
    printf("Position error: %.4f\n", estimationError);

    printf("\n" GREEN "Processing completed successfully!" RESET "\n");

    // Cleanup
    matrix_free(F, 2);
    matrix_free(H, 1);
    matrix_free(Q, 2);
    matrix_free(R, 1);
    matrix_free(P, 2);
    kalman_destroy(kf);

    return 0;
}
