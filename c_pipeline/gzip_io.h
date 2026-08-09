#ifndef GZIP_IO_H
#define GZIP_IO_H

#include <stddef.h>
#include <stdint.h>

/* Read a file fully into memory. Caller frees *out_data. */
int read_file_all(const char *path, uint8_t **out_data, size_t *out_size);

/* Write entire buffer to file. */
int write_file_all(const char *path, const uint8_t *data, size_t size);

/* If src starts with 0x1F 0x8B, gzip-inflate it; otherwise duplicate as-is.
 * Allocates *out_data; caller frees. */
int gzip_inflate_if_needed(const uint8_t *src, size_t src_size,
                           uint8_t **out_data, size_t *out_size);

/* Always gzip-deflate src into *out_data (allocated). */
int gzip_deflate(const uint8_t *src, size_t src_size,
                 uint8_t **out_data, size_t *out_size);

#endif
