# Kalman Filter Implementation in C

This repository contains a pure C implementation of a Kalman filter for real-time state estimation.

## Features

- **Pure C Implementation**: No C++ dependencies, compatible with C99 standard
- **Dynamic Memory Management**: Efficient allocation and deallocation of matrices
- **Comprehensive Matrix Operations**: Addition, subtraction, multiplication, transpose, and inversion
- **Error Handling**: Robust error checking and memory cleanup
- **Well-Documented API**: Clear function signatures with comprehensive documentation
- **Unit Tests**: Basic test suite for matrix operations
- **Example Application**: 1D position tracking with velocity estimation

## Project Structure

```
├── kalman.h          # Main Kalman filter interface
├── kalman.c          # Kalman filter implementation
├── matrix_math.h     # Matrix operations interface
├── matrix_math.c     # Matrix operations implementation
├── main.c            # Example usage and demonstration
├── tests/
│   ├── test_matrix.c # Unit tests for matrix operations
│   └── CMakeLists.txt
└── CMakeLists.txt    # Build configuration
└── .gitattributes    # Git line ending configuration
```

## Building

### Prerequisites
- CMake 3.14 or higher
- C99-compatible compiler (GCC, Clang, MSVC)
- Doxygen (optional, for generating documentation)

### Build Instructions

```bash
# Configure
cmake --preset debug
cmake --preset release

# Build
cmake --build --preset debug
cmake --build --preset release
```

## Usage Example

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

## Documentation

This project includes comprehensive Doxygen documentation.

### Documentation Features
- **Complete API Reference**: All functions, structures, and parameters
- **Usage Examples**: Practical code examples for each function
- **Mathematical Background**: Detailed explanation of Kalman filter equations
- **Group Organization**: Functions organized by purpose (memory, operations, etc.)
- **Cross-References**: Links between related functions and files

## API Reference

### Kalman Filter Functions

- `kalman_create()` - Create new filter instance
- `kalman_destroy()` - Free filter memory
- `kalman_initialize_state()` - Set initial state
- `kalman_set_state_transition_matrix()` - Set F matrix
- `kalman_set_observation_matrix()` - Set H matrix
- `kalman_set_process_noise_covariance()` - Set Q matrix
- `kalman_set_measurement_noise_covariance()` - Set R matrix
- `kalman_set_error_covariance()` - Set P matrix
- `kalman_predict()` - Prediction step
- `kalman_update()` - Correction step
- `kalman_get_state()` - Get current state vector
- `kalman_get_state_element()` - Get specific state element

### Matrix Functions

- `matrix_allocate()` - Allocate matrix memory
- `matrix_free()` - Free matrix memory
- `matrix_add()` - Matrix addition
- `matrix_subtract()` - Matrix subtraction
- `matrix_multiply()` - Matrix multiplication
- `matrix_transpose()` - Matrix transpose
- `matrix_inverse()` - Matrix inversion using Gaussian elimination
- `matrix_identity()` - Create identity matrix

## Mathematical Background

The Kalman filter implementation follows the standard discrete-time formulation:

**Prediction Step:**
- State prediction: `x(k|k-1) = F * x(k-1|k-1) + B * u(k-1)`
- Covariance prediction: `P(k|k-1) = F * P(k-1|k-1) * F^T + Q`

**Update Step:**
- Innovation (residual): `y(k) = z(k) - H * x(k|k-1)`
- Innovation covariance: `S(k) = H * P(k|k-1) * H^T + R`
- Kalman gain: `K(k) = P(k|k-1) * H^T * S(k)^(-1)`
- State update: `x(k|k) = x(k|k-1) + K(k) * y(k)`
- Covariance update: `P(k|k) = (I - K(k) * H) * P(k|k-1)`

Where:
- `x` = state vector
- `F` = state transition matrix
- `B` = control matrix (optional)
- `u` = control vector (optional)
- `Q` = process noise covariance
- `H` = observation matrix
- `R` = measurement noise covariance
- `P` = error covariance matrix
- `z` = measurement vector
- `I` = identity matrix

## License

This project is licensed under the GNU GPLv3 - see the LICENSE file for details.
