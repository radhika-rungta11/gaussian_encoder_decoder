#ifndef SCENE_H
#define SCENE_H

#include <stddef.h>
#include <stdint.h>

/* Decoded RVQ block keeps both raw and reconstructed data so the encoder
 * can either re-emit decoded floats (Level B) or just reuse codebooks. */
typedef struct {
    uint8_t   num_layers;
    uint16_t  codebook_size;
    uint16_t  dim;
    float    *codebooks;       /* [num_layers, codebook_size, dim] */
    float    *values;          /* [N, dim] reconstructed, float32  */
} vq_block_t;

typedef struct {
    float min_val;
    float max_val;
    float *values;             /* [N] dequantized, float32 */
} scalar_block_t;

typedef struct {
    uint32_t version;
    uint32_t N;

    float *xyz;                /* [N, 3] float32 (from f16) */
    float *motion;             /* [N, 9] float32 (from f16) */

    scalar_block_t opacity;
    scalar_block_t tcen;
    scalar_block_t tsca;

    vq_block_t scale;          /* dim=3, log-space; apply exp() before render */
    vq_block_t rotation;       /* dim=4, quaternion (w,x,y,z), un-normalized */
    vq_block_t omega;          /* dim=4 */
    vq_block_t tfea;           /* dim=3 */

    int      has_features;
    float   *features_dc;      /* [N, 6] float32 (from f16); NULL if none */

    int      has_rgb_dec;
    float   *rgb_w1;           /* [6,12] float32 (from f16); NULL if none */
    float   *rgb_w2;           /* [3,6]  float32 (from f16); NULL if none */
} scene_t;

void scene_init(scene_t *s);
void scene_free(scene_t *s);

/* Apply motion polynomial coefficients to xyz at given t.
 * out must be sized [N*3]. */
void scene_xyz_at_time(const scene_t *s, float t, float *out);

/* Min/max bounds over xyz at t=0 (writes 6 floats into bounds). */
void scene_bounds_xyz(const scene_t *s, float *bounds);

/* sigmoid(features_dc[:, :3]) fallback color into rgb [N*3].
 * If features_dc is NULL, fills white. */
void scene_fallback_rgb(const scene_t *s, float *rgb);

#endif
