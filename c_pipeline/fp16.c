#include "fp16.h"

float fp16_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits;
    union { uint32_t u; float f; } out;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    out.u = bits;
    return out.f;
}

uint16_t float_to_fp16(float f) {
    union { float f; uint32_t u; } in;
    in.f = f;
    uint32_t bits = in.u;
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x007FFFFFu;

    if (((bits >> 23) & 0xFFu) == 0xFF) {
        /* inf or NaN */
        return (uint16_t)(sign | 0x7C00u | (mant ? (mant >> 13 | 0x200u) : 0));
    }
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        /* round to nearest even */
        uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        half_mant += round_bit;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    uint32_t half_mant = mant >> 13;
    uint32_t round_bit = (mant >> 12) & 1u;
    uint16_t out_bits = (uint16_t)(sign | (((uint32_t)exp) << 10) | half_mant);
    out_bits = (uint16_t)(out_bits + (uint16_t)round_bit);
    return out_bits;
}
