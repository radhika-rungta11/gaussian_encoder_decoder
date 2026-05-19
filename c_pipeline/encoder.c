#include "encoder.h"
#include "4dgs_format.h"
#include "fp16.h"
#include "gzip_io.h"
#include "huffman.h"
#include "rvq.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int write_f16_array(byte_writer_t *w, const float *src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (byte_writer_u16(w, float_to_fp16(src[i])) != 0) return -1;
    }
    return 0;
}

static int write_scalar_block(byte_writer_t *w, const scalar_block_t *blk,
                              uint32_t N) {
    /* Re-quantize from decoded floats so we can be fully self-contained. */
    float mn = FLT_MAX, mx = -FLT_MAX;
    for (uint32_t i = 0; i < N; ++i) {
        float v = blk->values[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    /* Use a tight new minmax (matches what min-max quantize would produce). */
    if (byte_writer_f32(w, mn) != 0) return -1;
    if (byte_writer_f32(w, mx) != 0) return -1;

    uint8_t *q = (uint8_t *)malloc(N);
    if (!q) return -1;
    float span = mx - mn;
    if (span < 1e-12f) {
        memset(q, 0, N);
    } else {
        float inv = 255.0f / span;
        for (uint32_t i = 0; i < N; ++i) {
            float t = (blk->values[i] - mn) * inv;
            int qi = (int)(t + 0.5f);
            if (qi < 0) qi = 0; if (qi > 255) qi = 255;
            q[i] = (uint8_t)qi;
        }
    }

    huff_table_t table = {0};
    if (huffman_build_table(q, N, &table) != 0) { free(q); return -1; }
    int rc = huffman_encode_block(w, q, N, &table);
    huffman_table_free(&table);
    free(q);
    return rc;
}

static int write_vq_block(byte_writer_t *w, const vq_block_t *blk, uint32_t N) {
    if (byte_writer_u8 (w, blk->num_layers)    != 0) return -1;
    if (byte_writer_u16(w, blk->codebook_size) != 0) return -1;
    if (byte_writer_u16(w, blk->dim)           != 0) return -1;

    /* codebooks back to f16 */
    size_t cb_count = (size_t)blk->num_layers * blk->codebook_size * blk->dim;
    if (write_f16_array(w, blk->codebooks, cb_count) != 0) return -1;

    /* greedy nearest-neighbor residual indices */
    uint32_t *indices = (uint32_t *)malloc(sizeof(uint32_t) * N * blk->num_layers);
    if (!indices) return -1;
    if (rvq_encode_indices(blk->values, N, blk->num_layers,
                           blk->codebook_size, blk->dim,
                           blk->codebooks, indices) != 0) {
        free(indices); return -1;
    }

    int rvq_bit;
    if (rvq_log2_check(blk->codebook_size, &rvq_bit) != 0) {
        free(indices); return -1;
    }
    uint8_t *packed = NULL;
    size_t   packed_n = 0;
    if (rvq_pack_indices(indices, N, blk->num_layers, rvq_bit,
                         &packed, &packed_n) != 0) {
        free(indices); return -1;
    }
    free(indices);

    huff_table_t table = {0};
    if (huffman_build_table(packed, packed_n, &table) != 0) {
        free(packed); return -1;
    }
    int rc = huffman_encode_block(w, packed, packed_n, &table);
    huffman_table_free(&table);
    free(packed);
    return rc;
}

int encoder_emit_4dgs(const scene_t *s,
                      uint8_t **out_bytes, size_t *out_size) {
    byte_writer_t w; byte_writer_init(&w);

    if (byte_writer_u32(&w, FOURDGS_MAGIC_LE) != 0) goto fail;
    if (byte_writer_u32(&w, s->version)       != 0) goto fail;
    if (byte_writer_u32(&w, s->N)             != 0) goto fail;

    if (write_f16_array(&w, s->xyz,    (size_t)s->N * 3) != 0) goto fail;
    if (write_f16_array(&w, s->motion, (size_t)s->N * 9) != 0) goto fail;

    if (write_scalar_block(&w, &s->opacity, s->N) != 0) goto fail;
    if (write_scalar_block(&w, &s->tcen,    s->N) != 0) goto fail;
    if (write_scalar_block(&w, &s->tsca,    s->N) != 0) goto fail;

    if (write_vq_block(&w, &s->scale,    s->N) != 0) goto fail;
    if (write_vq_block(&w, &s->rotation, s->N) != 0) goto fail;
    if (write_vq_block(&w, &s->omega,    s->N) != 0) goto fail;
    if (write_vq_block(&w, &s->tfea,     s->N) != 0) goto fail;

    if (byte_writer_u8(&w, (uint8_t)(s->has_features ? 1 : 0)) != 0) goto fail;
    if (s->has_features) {
        if (write_f16_array(&w, s->features_dc, (size_t)s->N * 6) != 0) goto fail;
    }
    if (byte_writer_u8(&w, (uint8_t)(s->has_rgb_dec ? 1 : 0)) != 0) goto fail;
    if (s->has_rgb_dec) {
        if (write_f16_array(&w, s->rgb_w1, FOURDGS_RGB_W1_COUNT) != 0) goto fail;
        if (write_f16_array(&w, s->rgb_w2, FOURDGS_RGB_W2_COUNT) != 0) goto fail;
    }

    *out_bytes = w.data;
    *out_size  = w.size;
    /* Detach buffer ownership */
    w.data = NULL; w.capacity = 0; w.size = 0;
    return 0;

fail:
    byte_writer_free(&w);
    return -1;
}

int encoder_write_file(const scene_t *s, const char *path, int do_gzip) {
    uint8_t *body = NULL; size_t body_n = 0;
    if (encoder_emit_4dgs(s, &body, &body_n) != 0) return -1;

    if (do_gzip) {
        uint8_t *gz = NULL; size_t gz_n = 0;
        if (gzip_deflate(body, body_n, &gz, &gz_n) != 0) {
            free(body); return -1;
        }
        int rc = write_file_all(path, gz, gz_n);
        free(gz); free(body);
        return rc;
    }
    int rc = write_file_all(path, body, body_n);
    free(body);
    return rc;
}
