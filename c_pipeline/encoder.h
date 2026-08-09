#ifndef ENCODER_H
#define ENCODER_H

#include "scene.h"
#include <stddef.h>
#include <stdint.h>

/* Build .4dgs body bytes from decoded scene. Re-encodes:
 *   - xyz, motion: float -> f16
 *   - opacity, tcen, tsca: min-max quantize uint8 + canonical Huffman
 *   - scale, rotation, omega, tfea: keep codebooks, greedy nearest residual
 *     index assignment, then bit-pack + canonical Huffman
 *   - features_dc / rgb_dec: float -> f16 if present
 *
 * Caller frees *out_bytes. */
int encoder_emit_4dgs(const scene_t *s,
                      uint8_t **out_bytes, size_t *out_size);

/* Convenience: emit then optionally gzip-wrap, then write to disk. */
int encoder_write_file(const scene_t *s, const char *path, int do_gzip);

#endif
