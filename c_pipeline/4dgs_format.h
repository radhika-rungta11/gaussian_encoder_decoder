#ifndef FOURDGS_FORMAT_H
#define FOURDGS_FORMAT_H

#include <stdint.h>

#define FOURDGS_MAGIC_LE   0x53474434u  /* "4DGS" little-endian */
#define FOURDGS_MAGIC_STR  "4DGS"
#define FOURDGS_VERSION    3
#define FOURDGS_HEADER     12

/* per-attribute VQ dims, fixed by format */
#define FOURDGS_DIM_SCALE     3
#define FOURDGS_DIM_ROTATION  4
#define FOURDGS_DIM_OMEGA     4
#define FOURDGS_DIM_TFEA      3

/* Sandwich MLP weight shapes */
#define FOURDGS_RGB_W1_COUNT  (6 * 12)
#define FOURDGS_RGB_W2_COUNT  (3 * 6)

/* Sentinel emitted by encoder for the "_EOF" key in the dahuffman codec */
#define FOURDGS_HUFFMAN_EOF_SYMBOL  0xFFFFu
#define FOURDGS_HUFFMAN_EOF_INTERN  256

#endif
