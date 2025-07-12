# Kalman and Extended Kalman Filter Implementation in C

This repository contains a pure C implementation of a Kalman Filter (KF) and an Extended Kalman Filter (EKF) for real-time state estimation.

## Features

- **Dual Filter Support**: Implements both standard KF for linear systems and EKF for non-linear systems.
- **Pure C Implementation**: No C++ dependencies, compatible with the C99 standard.
- **Dynamic Memory Management**: Efficient allocation and deallocation of matrices.
- **Comprehensive Matrix Operations**: Includes addition, subtraction, multiplication, transpose, and inversion.
- **Robust Error Handling**: Provides error checking and safe memory cleanup.
- **Well-Documented API**: Clear function signatures with comprehensive Doxygen-style comments.
- **Unit Tests**: Contains test suites for both matrix operations and the EKF implementation.
- **Example Application**: A demonstration of 1D position tracking with velocity estimation.

## Project Structure

```
├── kalman.h          # Main Kalman filter interface
├── kalman.c          # Kalman filter implementation
├── matrix_math.h     # Matrix operations interface
├── matrix_math.c     # Matrix operations implementation
├── main.c            # Example usage and demonstration
├── tests/
│   ├── test_matrix.c # Unit tests for matrix operations
│   ├── test_ekf.c    # Unit tests for EKF setup
│   └── CMakeLists.txt
└── CMakeLists.txt    # Build configuration
└── .gitattributes    # Git line ending configuration
```

## Building

### Prerequisites

- CMake 3.14 or higher
- A C99-compatible compiler (such as GCC, Clang, or MSVC)
- Doxygen (optional, for generating documentation)

### Build Instructions

The project uses CMake with presets for easy configuration and building.

1.  **Configure the project:**

    ```bash
    # For a debug build
    cmake --preset debug

    # For a release build
    cmake --preset release
    ```

2.  **Build the project:**

    ```bash
    # Build the debug configuration
    cmake --build --preset debug

    # Build the release configuration
    cmake --build --preset release
    ```

    Executables will be placed in the `build/<preset>/bin` directory.

## Usage Example (Linear KF)

```c
#include "kalman.h"
#include "matrix_math.h"

// Create a 2D state, 1D measurement Kalman filter
kalman_t *kf = kalman_create(2, 1, 0);

// Initialize state [position=0, velocity=0]
double initialState[2] = {0.0, 0.0};
kalman_initialize_state(kf, initialState);

// Set up state transition matrix (constant velocity model)
double **F = matrix_allocate(2, 2);
F[0][0] = 1.0; F[0][1] = 1.0;  // dt = 1.0
F[1][0] = 0.0; F[1][1] = 1.0;
kalman_set_state_transition_matrix(kf, F);

// Set up observation matrix (measure position only)
double **H = matrix_allocate(1, 2);
H[0][0] = 1.0; H[0][1] = 0.0;
kalman_set_observation_matrix(kf, H);

// ... set other matrices (Q, R, P) ...

// Prediction and update cycle
kalman_predict(kf, NULL);      // Predict step
double measurement[1] = {1.2};
kalman_update(kf, measurement); // Update step

// Get results
double position = kalman_get_state_element(kf, 0);
double velocity = kalman_get_state_element(kf, 1);

// Cleanup
matrix_free(F, 2);
matrix_free(H, 1);
kalman_destroy(kf);
```

## Usage Example (Extended EKF)

```c
#include "kalman.h"
#include "matrix_math.h"
#include <math.h>

// Non-linear state transition function
void state_transition_fn(const double *state_in, const double *control_in, double *state_out) {
    state_out[0] = state_in[0] + state_in[1]; // position += velocity
    state_out[1] = state_in[1];              // velocity remains constant
}

// Jacobian of the state transition function
void state_jacobian_fn(const double *state_in, double **jacobian_out) {
    jacobian_out[0][0] = 1.0;
    jacobian_out[0][1] = 1.0;
    jacobian_out[1][0] = 0.0;
    jacobian_out[1][1] = 1.0;
}

// Non-linear observation function (e.g., measuring range)
void observation_fn(const double *state_in, double *measurement_out) {
    measurement_out[0] = sqrt(state_in[0] * state_in[0]);
}

// Jacobian of the observation function
void observation_jacobian_fn(const double *state_in, double **jacobian_out) {
    jacobian_out[0][0] = state_in[0] / sqrt(state_in[0] * state_in[0]);
    jacobian_out[0][1] = 0.0;
}

// Create a 2D state, 1D measurement EKF
kalman_t *ekf = kalman_create(2, 1, 0);

// Set EKF functions
kalman_set_state_transition_function(ekf, state_transition_fn);
kalman_set_state_jacobian_function(ekf, state_jacobian_fn);
kalman_set_observation_function(ekf, observation_fn);
kalman_set_observation_jacobian_function(ekf, observation_jacobian_fn);

// ... initialize state and set other matrices (Q, R, P) ...

// Prediction and update cycle
kalman_predict(ekf, NULL);
kalman_update(ekf, &measurement);

// ... get results and cleanup ...
```

## API Reference

### Kalman Filter Functions

- `kalman_create()`: Creates a new filter instance.
- `kalman_destroy()`: Frees all memory associated with the filter.
- `kalman_initialize_state()`: Sets the initial state vector.
- `kalman_set_state_transition_matrix()`: Sets the `F` matrix for a linear KF.
- `kalman_set_observation_matrix()`: Sets the `H` matrix for a linear KF.
- `kalman_set_process_noise_covariance()`: Sets the `Q` matrix.
- `kalman_set_measurement_noise_covariance()`: Sets the `R` matrix.
- `kalman_set_error_covariance()`: Sets the initial `P` matrix.
- `kalman_predict()`: Performs the prediction step.
- `kalman_update()`: Performs the update/correction step.
- `kalman_get_state()`: Returns the current state vector.
- `kalman_get_state_element()`: Returns a specific element from the state vector.

### EKF-Specific Functions

- `kalman_set_state_transition_function()`: Sets the non-linear state transition function `f(x, u)`.
- `kalman_set_state_jacobian_function()`: Sets the function to compute the Jacobian of `f(x, u)`.
- `kalman_set_observation_function()`: Sets the non-linear observation function `h(x)`.
- `kalman_set_observation_jacobian_function()`: Sets the function to compute the Jacobian of `h(x)`.

### Matrix Functions

- `matrix_allocate()`: Allocates memory for a new matrix.
- `matrix_free()`: Frees memory used by a matrix.
- `matrix_add()`: Performs matrix addition.
- `matrix_subtract()`: Performs matrix subtraction.
- `matrix_multiply()`: Performs matrix multiplication.
- `matrix_transpose()`: Transposes a matrix.
- `matrix_inverse()`: Computes the inverse of a matrix.
- `matrix_identity()`: Creates an identity matrix.

## Mathematical Background

The filter implementation follows the standard discrete-time formulations.

### Prediction Step

- **KF**: `x(k|k-1) = F * x(k-1|k-1) + B * u(k-1)`
- **EKF**: `x(k|k-1) = f(x(k-1|k-1), u(k-1))`
- **Covariance**: `P(k|k-1) = F * P(k-1|k-1) * F^T + Q` (where `F` is the Jacobian for EKF)

### Update Step

- **Innovation (KF)**: `y(k) = z(k) - H * x(k|k-1)`
- **Innovation (EKF)**: `y(k) = z(k) - h(x(k|k-1))`
- **Innovation Covariance**: `S(k) = H * P(k|k-1) * H^T + R` (where `H` is the Jacobian for EKF)
- **Kalman Gain**: `K(k) = P(k|k-1) * H^T * S(k)^(-1)`
- **State Update**: `x(k|k) = x(k|k-1) + K(k) * y(k)`
- **Covariance Update**: `P(k|k) = (I - K(k) * H) * P(k|k-1)`

## License

This project is licensed under the GNU GPLv3. See the [LICENSE](LICENSE) file for details.