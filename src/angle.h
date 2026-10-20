/**
 * @file    angle.h
 * @brief   Angles in radians that wrap around, for filters whose states or measurements are headings and bearings.
 */

#ifndef KALMAN_C_ANGLE_H
#define KALMAN_C_ANGLE_H

// Half a turn in radians
#define ANGLE_PI 3.14159265358979323846

// Wrapping
double angle_wrap_radians(double angle_radians);
double angle_difference_radians(double minuend_radians, double subtrahend_radians);

#endif // KALMAN_C_ANGLE_H
