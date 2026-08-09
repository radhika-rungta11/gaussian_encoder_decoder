#ifndef HUFFMAN_H
#define HUFFMAN_H

#include <stddef.h>
#include <stdint.h>

/* A single entry exactly as it appears on disk. */
typedef struct {
    uint16_t symbol;       /* 0..255 byte symbol or 0xFFFF EOF */
    uint8_t  bit_len;
    uint32_t code_bits;    /* MSB-first packed code, length = bit_len */
} huff_entry_t;

typedef struct {
    huff_entry_t *entries;
    uint16_t      count;
} huff_table_t;

/* Decode. Reader API matches decoder.c byte-cursor. */
typedef struct {
    const uint8_t *data;
    size_t         size;
    size_t         offset;
} byte_reader_t;

typedef struct {
    uint8_t *data;
    size_t   size;
    size_t   capacity;
} byte_writer_t;

void byte_writer_init(byte_writer_t *w);
void byte_writer_free(byte_writer_t *w);
int  byte_writer_append(byte_writer_t *w, const void *src, size_t n);
int  byte_writer_u8 (byte_writer_t *w, uint8_t  v);
int  byte_writer_u16(byte_writer_t *w, uint16_t v);
int  byte_writer_u32(byte_writer_t *w, uint32_t v);
int  byte_writer_f32(byte_writer_t *w, float    v);

int  byte_reader_take(byte_reader_t *r, size_t n, const uint8_t **out);
int  byte_reader_u8 (byte_reader_t *r, uint8_t  *out);
int  byte_reader_u16(byte_reader_t *r, uint16_t *out);
int  byte_reader_u32(byte_reader_t *r, uint32_t *out);
int  byte_reader_f32(byte_reader_t *r, float    *out);

/* Read a Huffman block (table + payload) and return decoded byte array.
 * Caller frees *out_values. */
int huffman_decode_block(byte_reader_t *r,
                         uint8_t **out_values, size_t *out_count);

/* Build a canonical Huffman table from byte frequencies (256 bins).
 * Adds an EOF symbol (0xFFFF) with count==1 so decoders terminate cleanly. */
int huffman_build_table(const uint8_t *src, size_t n, huff_table_t *out);
void huffman_table_free(huff_table_t *t);

/* Encode src using table and append a complete Huffman block to writer. */
int huffman_encode_block(byte_writer_t *w,
                         const uint8_t *src, size_t n,
                         const huff_table_t *table);

#endif
