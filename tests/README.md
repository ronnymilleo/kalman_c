# Tests

The tests use [Unity](https://github.com/ThrowTheSwitch/Unity), a git submodule in `vendor/unity`. Each executable holds
the tests of one topic and is registered with CTest.

| Executable               | Covers                                                                              |
|--------------------------|-------------------------------------------------------------------------------------|
| `matrix_tests`           | Construction, arithmetic, transpose, multiplication and inversion of matrices       |
| `kalman_tests`           | Setup, validation, prediction and update of the linear filter                       |
| `kalman_ekf_tests`       | Non-linear prediction and update, Jacobian evaluation, incomplete models            |
| `kalman_tracking_tests`  | The filter following a moving object through noisy measurements                     |

`test_support.[ch]` holds what they share: a pseudo-random generator, so noisy scenarios give the same numbers on every
platform, and builders for the usual position and velocity models.

## Running

```bash
git submodule update --init vendor/unity
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

## Adding a test

1. Write a `static void test_<what happens>(void)` function in the executable of the topic. Name it as a sentence that
   states the behavior.
2. Register it with `RUN_TEST` in that file's `main`.
3. A new executable goes into `tests/CMakeLists.txt` through `kalman_c_add_test()` and into
   `KALMAN_C_FORMAT_SOURCES` in the top-level `CMakeLists.txt`.
