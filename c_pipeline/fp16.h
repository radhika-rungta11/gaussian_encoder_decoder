#ifndef FP16_H
#define FP16_H

#include <stdint.h>

float    fp16_to_float(uint16_t h);
uint16_t float_to_fp16(float f);

#endif
