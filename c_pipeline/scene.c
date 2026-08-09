#include "scene.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void scene_init(scene_t *s) { memset(s, 0, sizeof(*s)); }

static void vq_free(vq_block_t *v) {
    free(v->codebooks); v->codebooks = NULL;
    free(v->values);    v->values = NULL;
}
static void sca_free(scalar_block_t *s) {
    free(s->values); s->values = NULL;
}

void scene_free(scene_t *s) {
    free(s->xyz);    s->xyz = NULL;
    free(s->motion); s->motion = NULL;
    sca_free(&s->opacity);
    sca_free(&s->tcen);
    sca_free(&s->tsca);
    vq_free(&s->scale);
    vq_free(&s->rotation);
    vq_free(&s->omega);
    vq_free(&s->tfea);
    free(s->features_dc); s->features_dc = NULL;
    free(s->rgb_w1);      s->rgb_w1 = NULL;
    free(s->rgb_w2);      s->rgb_w2 = NULL;
}

void scene_xyz_at_time(const scene_t *s, float t, float *out) {
    float t2 = t * t;
    float t3 = t2 * t;
    for (uint32_t i = 0; i < s->N; ++i) {
        const float *x = s->xyz + (size_t)i * 3;
        const float *m = s->motion + (size_t)i * 9;
        for (int k = 0; k < 3; ++k) {
            out[(size_t)i * 3 + k] =
                x[k]
                + m[k]     * t
                + m[3 + k] * t2
                + m[6 + k] * t3;
        }
    }
}

void scene_bounds_xyz(const scene_t *s, float *bounds) {
    float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    float mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (uint32_t i = 0; i < s->N; ++i) {
        for (int k = 0; k < 3; ++k) {
            float v = s->xyz[(size_t)i * 3 + k];
            if (v < mn[k]) mn[k] = v;
            if (v > mx[k]) mx[k] = v;
        }
    }
    bounds[0]=mn[0]; bounds[1]=mn[1]; bounds[2]=mn[2];
    bounds[3]=mx[0]; bounds[4]=mx[1]; bounds[5]=mx[2];
}

static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

void scene_fallback_rgb(const scene_t *s, float *rgb) {
    if (!s->features_dc) {
        for (uint32_t i = 0; i < s->N; ++i) {
            rgb[(size_t)i * 3 + 0] = 1.0f;
            rgb[(size_t)i * 3 + 1] = 1.0f;
            rgb[(size_t)i * 3 + 2] = 1.0f;
        }
        return;
    }
    for (uint32_t i = 0; i < s->N; ++i) {
        const float *f = s->features_dc + (size_t)i * 6;
        rgb[(size_t)i * 3 + 0] = sigmoidf(f[0]);
        rgb[(size_t)i * 3 + 1] = sigmoidf(f[1]);
        rgb[(size_t)i * 3 + 2] = sigmoidf(f[2]);
    }
}
