/**
 * @file    angle.c
 * @brief   Angles in radians that wrap around, for filters whose states or measurements are headings and bearings.
 */

#include "angle.h"

#include <math.h>

static const double full_turn_radians = 2.0 * ANGLE_PI;

/**
 * @brief   Brings an angle into the range from just above -pi up to and including pi.
 * @param[in] angle_radians  Angle of any size.
 * @return  The same direction, within one turn centered on zero.
 */
double angle_wrap_radians(const double angle_radians) {
    double wrapped = fmod(angle_radians + ANGLE_PI, full_turn_radians);
    if (wrapped <= 0.0) {
        wrapped += full_turn_radians;
    }
    return wrapped - ANGLE_PI;
}

/**
 * @brief   Subtracts two angles along the shorter way around the circle.
 * @param[in] minuend_radians     Angle to subtract from.
 * @param[in] subtrahend_radians  Angle to subtract.
 * @return  The signed difference, within one turn centered on zero: 179 degrees minus -179 degrees is -2 degrees.
 */
double angle_difference_radians(const double minuend_radians, const double subtrahend_radians) {
    return angle_wrap_radians(minuend_radians - subtrahend_radians);
}
