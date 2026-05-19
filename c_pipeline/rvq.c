#include "rvq.h"

#include <stdlib.h>
#include <string.h>

int rvq_log2_check(uint16_t codebook_size, int *out_bits) {
    int b = 0;
    while ((1u << b) < codebook_size) b++;
    if ((1u << b) != codebook_size) return -1;
    *out_bits = b;
    return 0;
}

int rvq_reconstruct(const uint8_t *packed_indices, size_t packed_size,
                    uint32_t count, uint8_t num_layers,
                    uint16_t codebook_size, uint16_t dim,
                    const float *codebooks, float *out) {
    int rvq_bit = 0;
    if (rvq_log2_check(codebook_size, &rvq_bit) != 0) return -1;

    memset(out, 0, (size_t)count * dim * sizeof(float));

    for (uint32_t i = 0; i < count; ++i) {
        for (uint8_t layer = 0; layer < num_layers; ++layer) {
            uint32_t index = 0;
            size_t logical = ((size_t)i * num_layers + layer) * (size_t)rvq_bit;
            for (int b = 0; b < rvq_bit; ++b) {
                size_t pos = logical + b;
                size_t byte_pos = pos / 8;
                int bit_in_byte = (int)(pos % 8);
                if (byte_pos >= packed_size) return -1;
                if ((packed_indices[byte_pos] >> bit_in_byte) & 1u)
                    index |= (1u << b);
            }
            const float *cb_row = codebooks
                + ((size_t)layer * codebook_size + index) * dim;
            float *out_row = out + (size_t)i * dim;
            for (uint16_t d = 0; d < dim; ++d) out_row[d] += cb_row[d];
        }
    }
    return 0;
}

int rvq_encode_indices(const float *vectors,
                       uint32_t count, uint8_t num_layers,
                       uint16_t codebook_size, uint16_t dim,
                       const float *codebooks,
                       uint32_t *out_idx) {
    float *residual = (float *)malloc(sizeof(float) * dim);
    if (!residual) return -1;

    for (uint32_t i = 0; i < count; ++i) {
        for (uint16_t d = 0; d < dim; ++d)
            residual[d] = vectors[(size_t)i * dim + d];

        for (uint8_t layer = 0; layer < num_layers; ++layer) {
            uint32_t best = 0;
            float best_dist = 0.0f;
            int found = 0;
            for (uint16_t k = 0; k < codebook_size; ++k) {
                const float *cb_row = codebooks
                    + ((size_t)layer * codebook_size + k) * dim;
                float dist = 0.0f;
                for (uint16_t d = 0; d < dim; ++d) {
                    float diff = residual[d] - cb_row[d];
                    dist += diff * diff;
                }
                if (!found || dist < best_dist) {
                    best_dist = dist; best = k; found = 1;
                }
            }
            out_idx[(size_t)i * num_layers + layer] = best;
            const float *cb_row = codebooks
                + ((size_t)layer * codebook_size + best) * dim;
            for (uint16_t d = 0; d < dim; ++d)
                residual[d] -= cb_row[d];
        }
    }
    free(residual);
    return 0;
}

int rvq_pack_indices(const uint32_t *indices,
                     uint32_t count, uint8_t num_layers, int rvq_bit,
                     uint8_t **out_bytes, size_t *out_bytes_size) {
    size_t total_bits = (size_t)count * num_layers * (size_t)rvq_bit;
    size_t total_bytes = (total_bits + 7) / 8;
    uint8_t *buf = (uint8_t *)calloc(total_bytes, 1);
    if (!buf) return -1;

    for (uint32_t i = 0; i < count; ++i) {
        for (uint8_t layer = 0; layer < num_layers; ++layer) {
            uint32_t idx = indices[(size_t)i * num_layers + layer];
            size_t logical = ((size_t)i * num_layers + layer) * (size_t)rvq_bit;
            for (int b = 0; b < rvq_bit; ++b) {
                if ((idx >> b) & 1u) {
                    size_t pos = logical + b;
                    buf[pos / 8] |= (uint8_t)(1u << (pos % 8));
                }
            }
        }
    }
    *out_bytes = buf;
    *out_bytes_size = total_bytes;
    return 0;
}
