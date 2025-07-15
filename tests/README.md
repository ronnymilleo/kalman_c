# Kalman Filter Test Suite

This directory contains comprehensive unit tests for the Kalman filter implementation, organized into focused test modules to reduce redundancy and improve maintainability.

## Test Organization

### Core Test Files

- **`test_matrix.c`** - Matrix math operations tests
  - Matrix addition, subtraction, multiplication
  - Matrix transpose and identity matrix creation
  - Memory allocation and error handling

- **`test_ekf.c`** - Extended Kalman Filter tests
  - Non-linear state transition and observation functions
  - Jacobian matrix validation
  - EKF-specific functionality

### Standard Kalman Filter Tests (Split into focused modules)

- **`test_kalman_basic.c`** - Basic functionality tests
  - Filter creation and initialization
  - Basic prediction and update steps
  - Matrix setup and configuration validation

- **`test_kalman_1d.c`** - 1D tracking tests
  - 1D position and velocity tracking with constant velocity model
  - Convergence validation with poor initial estimates
  - Initial conditions robustness testing

- **`test_kalman_2d.c`** - 2D tracking tests
  - 2D object tracking with position and velocity estimation
  - Multi-dimensional state space validation
  - Motion patterns testing (horizontal, vertical, diagonal)

- **`test_kalman_noise.c`** - Noise sensitivity tests
  - Performance analysis with different noise levels
  - Process noise vs measurement noise trade-off analysis
  - Convergence quality assessment

## Building and Running Tests

### Build all tests
```bash
cd tests
mkdir build && cd build
cmake ..
make
```

### Run individual test modules
```bash
# Run basic functionality tests
./test_kalman_basic

# Run 1D tracking tests
./test_kalman_1d

# Run 2D tracking tests
./test_kalman_2d

# Run noise sensitivity tests
./test_kalman_noise

# Run matrix math tests
./test_matrix

# Run EKF tests
./test_ekf
```

### Run all tests using CMake targets
```bash
# Run all tests sequentially
make run_tests

# Run with memory checking (requires Valgrind)
make run_tests_memcheck

# Run with code coverage (if enabled)
make coverage
```

## Test Coverage

The test suite covers:

1. **Basic Operations**
   - Filter creation and destruction
   - State initialization and retrieval
   - Matrix configuration

2. **Core Functionality**
   - Prediction step validation
   - Update step validation
   - State estimation accuracy

3. **Practical Applications**
   - 1D constant velocity tracking
   - 2D multi-dimensional tracking
   - Different motion patterns

4. **Robustness**
   - Noise sensitivity analysis
   - Initial condition variations
   - Process vs measurement noise trade-offs

5. **Advanced Features**
   - Extended Kalman Filter functionality
   - Non-linear system handling
   - Matrix operations validation

## Adding New Tests

To add new test scenarios:

1. Create a new test file following the naming convention `test_kalman_<category>.c`
2. Include the standard test infrastructure (headers, result tracking, color codes)
3. Add the executable to `CMakeLists.txt`
4. Update the test runner in `test_kalman.c` if needed
5. Follow the existing test patterns for consistency

## Test Output

All tests provide:
- Colored output for easy identification of pass/fail status
- Detailed progress information
- Summary statistics
- Error thresholds and validation criteria
- Performance metrics where applicable 