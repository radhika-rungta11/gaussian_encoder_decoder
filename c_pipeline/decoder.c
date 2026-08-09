#include "decoder.h"
#include "4dgs_format.h"
#include "fp16.h"
#include "gzip_io.h"
#include "huffman.h"
#include "rvq.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_f16_array(byte_reader_t *r, size_t count, float *out) {
    const uint8_t *ptr = NULL;
    if (byte_reader_take(r, count * 2, &ptr) != 0) return -1;
    for (size_t i = 0; i < count; ++i) {
        uint16_t h = (uint16_t)(ptr[i * 2] | (ptr[i * 2 + 1] << 8));
        out[i] = fp16_to_float(h);
    }
    return 0;
}

static int read_scalar_block(byte_reader_t *r, uint32_t expected_count,
                             scalar_block_t *out, const char *name,
                             int verbose) {
    if (byte_reader_f32(r, &out->min_val) != 0) return -1;
    if (byte_reader_f32(r, &out->max_val) != 0) return -1;

    uint8_t *q = NULL;
    size_t   q_count = 0;
    if (huffman_decode_block(r, &q, &q_count) != 0) return -1;
    if (q_count != expected_count) {
        fprintf(stderr, "[decoder] %s: expected %u symbols, got %zu\n",
                name, expected_count, q_count);
        free(q);
        return -1;
    }

    out->values = (float *)malloc(sizeof(float) * expected_count);
    if (!out->values) { free(q); return -1; }
    float scale = (out->max_val - out->min_val) / 255.0f;
    for (uint32_t i = 0; i < expected_count; ++i) {
        out->values[i] = out->min_val + scale * (float)q[i];
    }
    free(q);

    if (verbose) {
        float mn = FLT_MAX, mx = -FLT_MAX, sum = 0.0f;
        for (uint32_t i = 0; i < expected_count; ++i) {
            float v = out->values[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        fprintf(stderr, "  %-8s N=%u min=%-12.6g max=%-12.6g mean=%-12.6g qrange=[%.4g, %.4g]\n",
                name, expected_count, mn, mx, sum / (float)expected_count,
                out->min_val, out->max_val);
    }
    return 0;
}

static int read_vq_block(byte_reader_t *r, uint32_t count,
                         uint16_t expected_dim, vq_block_t *out,
                         const char *name, int verbose) {
    uint8_t  num_layers = 0;
    uint16_t cb_size = 0;
    uint16_t dim = 0;
    if (byte_reader_u8(r, &num_layers) != 0)  return -1;
    if (byte_reader_u16(r, &cb_size)  != 0)   return -1;
    if (byte_reader_u16(r, &dim)      != 0)   return -1;
    if (dim != expected_dim) {
        fprintf(stderr, "[decoder] %s: dim mismatch %u vs %u\n",
                name, dim, expected_dim);
        return -1;
    }
    int rvq_bit;
    if (rvq_log2_check(cb_size, &rvq_bit) != 0) {
        fprintf(stderr, "[decoder] %s: cb_size %u not power of two\n",
                name, cb_size);
        return -1;
    }

    out->num_layers    = num_layers;
    out->codebook_size = cb_size;
    out->dim           = dim;

    size_t cb_count = (size_t)num_layers * cb_size * dim;
    out->codebooks = (float *)malloc(sizeof(float) * cb_count);
    if (!out->codebooks) return -1;
    if (read_f16_array(r, cb_count, out->codebooks) != 0) return -1;

    uint8_t *packed = NULL;
    size_t   packed_n = 0;
    if (huffman_decode_block(r, &packed, &packed_n) != 0) return -1;

    out->values = (float *)calloc((size_t)count * dim, sizeof(float));
    if (!out->values) { free(packed); return -1; }
    if (rvq_reconstruct(packed, packed_n, count, num_layers, cb_size, dim,
                        out->codebooks, out->values) != 0) {
        free(packed); return -1;
    }
    free(packed);

    if (verbose) {
        float mn = FLT_MAX, mx = -FLT_MAX, sum = 0.0f;
        size_t total = (size_t)count * dim;
        for (size_t i = 0; i < total; ++i) {
            float v = out->values[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        fprintf(stderr, "  %-8s shape=[%u,%u] layers=%u cb=%u rvq_bit=%d "
                "min=%-10.4g max=%-10.4g mean=%-10.4g\n",
                name, count, dim, num_layers, cb_size, rvq_bit,
                mn, mx, sum / (float)total);
    }
    return 0;
}

int decoder_parse_4dgs(const uint8_t *data, size_t size,
                       scene_t *out, int verbose) {
    scene_init(out);

    if (size < FOURDGS_HEADER) {
        fprintf(stderr, "[decoder] file too short\n");
        return -1;
    }

    byte_reader_t r = { data, size, 0 };
    uint32_t magic;
    if (byte_reader_u32(&r, &magic) != 0) return -1;
    if (magic != FOURDGS_MAGIC_LE) {
        fprintf(stderr, "[decoder] bad magic 0x%08x (want 0x%08x)\n",
                magic, FOURDGS_MAGIC_LE);
        return -1;
    }
    if (byte_reader_u32(&r, &out->version) != 0) return -1;
    if (byte_reader_u32(&r, &out->N)       != 0) return -1;

    if (verbose) {
        fprintf(stderr, "[decoder] magic=4DGS version=%u N=%u\n",
                out->version, out->N);
    }

    out->xyz    = (float *)malloc(sizeof(float) * out->N * 3);
    out->motion = (float *)malloc(sizeof(float) * out->N * 9);
    if (!out->xyz || !out->motion) return -1;
    if (read_f16_array(&r, (size_t)out->N * 3, out->xyz)    != 0) return -1;
    if (read_f16_array(&r, (size_t)out->N * 9, out->motion) != 0) return -1;

    if (verbose) {
        float bounds[6];
        scene_bounds_xyz(out, bounds);
        fprintf(stderr, "  xyz      shape=[%u,3] min=[%.4f,%.4f,%.4f] max=[%.4f,%.4f,%.4f]\n",
                out->N, bounds[0], bounds[1], bounds[2],
                bounds[3], bounds[4], bounds[5]);
        float m_mn = FLT_MAX, m_mx = -FLT_MAX;
        size_t total = (size_t)out->N * 9;
        for (size_t i = 0; i < total; ++i) {
            float v = out->motion[i];
            if (v < m_mn) m_mn = v;
            if (v > m_mx) m_mx = v;
        }
        fprintf(stderr, "  motion   shape=[%u,9] min=%.4g max=%.4g\n",
                out->N, m_mn, m_mx);
    }   

    if (read_scalar_block(&r, out->N, &out->opacity, "opacity", verbose) != 0) return -1;
    if (read_scalar_block(&r, out->N, &out->tcen,    "tcen",    verbose) != 0) return -1;
    if (read_scalar_block(&r, out->N, &out->tsca,    "tsca",    verbose) != 0) return -1;

    if (read_vq_block(&r, out->N, FOURDGS_DIM_SCALE,    &out->scale,    "scale",    verbose) != 0) return -1;
    if (read_vq_block(&r, out->N, FOURDGS_DIM_ROTATION, &out->rotation, "rotation", verbose) != 0) return -1;
    if (read_vq_block(&r, out->N, FOURDGS_DIM_OMEGA,    &out->omega,    "omega",    verbose) != 0) return -1;
    if (read_vq_block(&r, out->N, FOURDGS_DIM_TFEA,     &out->tfea,     "tfea",     verbose) != 0) return -1;

    uint8_t hf;
    if (byte_reader_u8(&r, &hf) != 0) return -1;
    out->has_features = hf;
    if (hf) {
        out->features_dc = (float *)malloc(sizeof(float) * out->N * 6);
        if (!out->features_dc) return -1;
        if (read_f16_array(&r, (size_t)out->N * 6, out->features_dc) != 0)
            return -1;
        if (verbose)
            fprintf(stderr, "  features_dc shape=[%u,6] (baked, present)\n", out->N);
    } else if (verbose) {
        fprintf(stderr, "  features_dc absent\n");
    }

    uint8_t hr;
    if (byte_reader_u8(&r, &hr) != 0) return -1;
    out->has_rgb_dec = hr;
    if (hr) {
        out->rgb_w1 = (float *)malloc(sizeof(float) * FOURDGS_RGB_W1_COUNT);
        out->rgb_w2 = (float *)malloc(sizeof(float) * FOURDGS_RGB_W2_COUNT);
        if (!out->rgb_w1 || !out->rgb_w2) return -1;
        if (read_f16_array(&r, FOURDGS_RGB_W1_COUNT, out->rgb_w1) != 0) return -1;
        if (read_f16_array(&r, FOURDGS_RGB_W2_COUNT, out->rgb_w2) != 0) return -1;
        if (verbose)
            fprintf(stderr, "  rgb_dec shapes w1=[6,12] w2=[3,6]\n");
    } else if (verbose) {
        fprintf(stderr, "  rgb_dec absent\n");
    }

    if (verbose) {
        fprintf(stderr, "[decoder] consumed %zu of %zu bytes (%s)\n",
                r.offset, r.size,
                r.offset == r.size ? "EXACT" : "leftover");
    }
    if (r.offset != r.size) {
        fprintf(stderr, "[decoder] WARN: %zu trailing bytes after parse\n",
                r.size - r.offset);
    }
    return 0;
}

int decoder_load_file(const char *path, scene_t *out, int verbose) {
    uint8_t *raw = NULL; size_t raw_n = 0;
    if (read_file_all(path, &raw, &raw_n) != 0) return -1;
    uint8_t *body = NULL; size_t body_n = 0;
    if (gzip_inflate_if_needed(raw, raw_n, &body, &body_n) != 0) {
        free(raw); return -1;
    }
    free(raw);
    int rc = decoder_parse_4dgs(body, body_n, out, verbose);
    free(body);
    return rc;
}
