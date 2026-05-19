#ifndef DECODER_H
#define DECODER_H

#include "scene.h"
#include <stdint.h>
#include <stddef.h>

/* Decode raw 4DGS bytes (already inflated) into scene_t.
 * If verbose != 0, prints validation summary (Stage 3).
 * Returns 0 on success, non-zero on error. */
int decoder_parse_4dgs(const uint8_t *data, size_t size,
                       scene_t *out, int verbose);

/* Convenience: read a file (handles gzip auto), then parse. */
int decoder_load_file(const char *path, scene_t *out, int verbose);

#endif
