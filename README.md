# KalmanC

[![CI](https://github.com/ronnymilleo/kalman_c/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ronnymilleo/kalman_c/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/ronnymilleo/kalman_c/badge.svg?branch=main)](https://coveralls.io/github/ronnymilleo/kalman_c?branch=main)
[![CodeFactor](https://www.codefactor.io/repository/github/ronnymilleo/kalman_c/badge)](https://www.codefactor.io/repository/github/ronnymilleo/kalman_c)

Kalman and Extended Kalman filters in C, for state estimation on hosts and embedded targets.

## Features

- **Kalman and Extended Kalman filter**: the same `kalman_t` runs the linear model, or the non-linear one once the state
  transition and observation functions and their Jacobians are set.
- **No dynamic allocation**: every matrix lives inside `kalman_t`, which the caller places on the stack or in static
  memory. Capacity is fixed at compile time (see [Capacity](#capacity)).
- **Status codes**: every function reports success or the kind of failure through `kalman_status_t` or
  `matrix_status_t`; results come back through out-parameters.
- **Small matrix module**: addition, subtraction, transpose, multiplication and inversion by Gauss-Jordan elimination
  with partial pivoting.
- **Unit tests with Unity** that run on the host and can be reused on a target.

## Project structure

```
├── src/
│   ├── matrix.h / matrix.c   # Fixed-capacity matrices and linear algebra
│   └── kalman.h / kalman.c   # Kalman and Extended Kalman filter
├── tests/                    # Unity tests, one executable per topic
├── vendor/unity              # Unity test framework (git submodule)
└── CMakeLists.txt
```

## Building

You need CMake 3.28 or newer, Ninja and a C17 compiler (GCC, Clang or MSVC).

```bash
# Unity is only needed for the tests
git submodule update --init vendor/unity

cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

The presets are `debug`, `release`, `clang-debug` and `clang-release`, and build into `build/<preset>`. Binaries and
libraries end up in `out/<build type>/bin` and `out/<build type>/lib`.

Without presets:

```bash
cmake -S . -B build -G Ninja -DKALMAN_C_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

| Option                        | Default | Purpose                                                    |
|-------------------------------|---------|------------------------------------------------------------|
| `KALMAN_C_BUILD_TESTS`        | `OFF`   | Build the unit tests                                       |
| `KALMAN_C_WARNINGS_AS_ERRORS` | `OFF`   | Make warnings in the project's code fail the build         |
| `KALMAN_C_ENABLE_SANITIZERS`  | `OFF`   | AddressSanitizer and UndefinedBehaviorSanitizer            |
| `KALMAN_C_ENABLE_COVERAGE`    | `OFF`   | Instrument the libraries for `gcov` (GCC or Clang)         |

Building also runs `clang-format` over the sources when it is installed, through the `format` target.

To use the library in another CMake project, add this folder with `add_subdirectory` and link `kalman-c-filter-lib`,
which brings `kalman-c-matrix-lib` along.

## Capacity

`MATRIX_MAX_DIM` (default 8) is the largest number of rows or columns of a matrix, and so the largest number of
states, measurements or controls of a filter. Set the `KALMAN_C_MATRIX_MAX_DIM` cache variable to fit the target's
memory; it is passed to everything that links the library, so `matrix_t` has the same size everywhere:

```bash
cmake -S . -B build -G Ninja -DKALMAN_C_MATRIX_MAX_DIM=4
```

When the sources are built some other way, define `MATRIX_MAX_DIM` for every file that includes `matrix.h`.

A `matrix_t` takes `MATRIX_MAX_DIM² * 8` bytes plus two sizes, and a `kalman_t` holds six of them. `kalman_update()`
keeps about a dozen more on the stack while it runs, so budget the stack accordingly on small targets.

## Usage

### Linear filter

```c
#include "kalman.h"

// Track position and velocity from position measurements
const double dt = 0.1;

kalman_t kf;
if (kalman_init(&kf, 2, 1, 0) != kalman_status_ok) {
    return;
}

matrix_t f;
matrix_t h;
matrix_t q;
matrix_t r;
matrix_t p;
matrix_from_values(&f, 2, 2, (const double[]){1.0, dt, 0.0, 1.0});
matrix_from_values(&h, 1, 2, (const double[]){1.0, 0.0});
matrix_from_values(&q, 2, 2, (const double[]){0.01, 0.0, 0.0, 0.01});
matrix_from_values(&r, 1, 1, (const double[]){0.25});
matrix_from_values(&p, 2, 2, (const double[]){1.0, 0.0, 0.0, 1.0});

const double initial_state[2] = {0.0, 0.0};
kalman_initialize_state(&kf, initial_state);
kalman_set_state_transition_matrix(&kf, &f);
kalman_set_observation_matrix(&kf, &h);
kalman_set_process_noise_covariance(&kf, &q);
kalman_set_measurement_noise_covariance(&kf, &r);
kalman_set_error_covariance(&kf, &p); // the filter is ready once P is set

const double measurement[1] = {1.2};
if (kalman_predict(&kf, nullptr) == kalman_status_ok && kalman_update(&kf, measurement) == kalman_status_ok) {
    double position = 0.0;
    double velocity = 0.0;
    kalman_get_state_element(&kf, 0, &position);
    kalman_get_state_element(&kf, 1, &velocity);
}

kalman_deinit(&kf);
```

Check the status of each call in real code; they are ignored here to keep the example short.

### Extended filter

Set the four functions and the filter switches to the non-linear model. The Jacobian callbacks receive a matrix that is
already sized, so they only fill in the entries:

```c
static void next_state(const double *state, const double *control, double *next_state_out) {
    control;
    next_state_out[0] = state[0] + state[1];
    next_state_out[1] = state[1];
}

static void next_state_jacobian(const double *state, matrix_t *jacobian) {
    state;
    jacobian->data[0][0] = 1.0;
    jacobian->data[0][1] = 1.0;
    jacobian->data[1][0] = 0.0;
    jacobian->data[1][1] = 1.0;
}

static void range(const double *state, double *measurement) {
    measurement[0] = sqrt(state[0] * state[0] + state[1] * state[1]);
}

static void range_jacobian(const double *state, matrix_t *jacobian) {
    const double distance = sqrt(state[0] * state[0] + state[1] * state[1]);
    jacobian->data[0][0] = state[0] / distance;
    jacobian->data[0][1] = state[1] / distance;
}

kalman_set_state_transition_function(&ekf, next_state);
kalman_set_state_jacobian_function(&ekf, next_state_jacobian);
kalman_set_observation_function(&ekf, range);
kalman_set_observation_jacobian_function(&ekf, range_jacobian);
```

A function without its Jacobian, or the other way round, makes `kalman_predict()` and `kalman_update()` return
`kalman_status_incomplete_model` rather than quietly using the linear model.

## Mathematical background

### Prediction

- **Linear**: `x = F x + B u`
- **Extended**: `x = f(x, u)`, with `F` the Jacobian of `f` evaluated at the estimate the step starts from
- **Covariance**: `P = F P Fᵀ + Q`

### Update

- **Innovation**: `y = z - H x` (linear) or `y = z - h(x)` (extended, with `H` the Jacobian of `h`)
- **Innovation covariance**: `S = H P Hᵀ + R`
- **Kalman gain**: `K = P Hᵀ S⁻¹`
- **State**: `x = x + K y`
- **Covariance**: `P = (I - K H) P`

## License

This project is licensed under the GNU GPLv3. See the [LICENSE](LICENSE) file for details.
