#ifndef RVQ_H
#define RVQ_H

#include <stddef.h>
#include <stdint.h>

/* Reconstruct vectors from packed RVQ indices and codebooks.
 * packed_indices: byte stream from huffman_decode_block, LSB-first per byte
 * codebooks:      [num_layers, codebook_size, dim] float32
 * out:            [count, dim] float32 (caller-allocated)
 */
int rvq_reconstruct(const uint8_t *packed_indices, size_t packed_size,
                    uint32_t count, uint8_t num_layers,
                    uint16_t codebook_size, uint16_t dim,
                    const float *codebooks, float *out);

/* Greedy nearest-neighbor RVQ encode: pick best codebook entry at each layer
 * and pass residual to next. Returns indices [count, num_layers] in *out_idx. */
int rvq_encode_indices(const float *vectors,
                       uint32_t count, uint8_t num_layers,
                       uint16_t codebook_size, uint16_t dim,
                       const float *codebooks,
                       uint32_t *out_idx /* size count*num_layers */);

/* Pack [count, num_layers] indices into a byte stream LSB-first.
 * rvq_bit = log2(codebook_size). Caller frees *out_bytes. */
int rvq_pack_indices(const uint32_t *indices,
                     uint32_t count, uint8_t num_layers, int rvq_bit,
                     uint8_t **out_bytes, size_t *out_bytes_size);

/* log2 helper that errors if codebook_size is not a power of two. */
int rvq_log2_check(uint16_t codebook_size, int *out_bits);

#endif
