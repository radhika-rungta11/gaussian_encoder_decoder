#include "renderer_cpu.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void v3_sub(const float *a, const float *b, float *o) {
    o[0]=a[0]-b[0]; o[1]=a[1]-b[1]; o[2]=a[2]-b[2];
}
static float v3_dot(const float *a, const float *b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
static void v3_cross(const float *a, const float *b, float *o) {
    o[0] = a[1]*b[2] - a[2]*b[1];
    o[1] = a[2]*b[0] - a[0]*b[2];
    o[2] = a[0]*b[1] - a[1]*b[0];
}
static void v3_norm(float *v) {
    float n = sqrtf(v3_dot(v,v));
    if (n > 1e-12f) { v[0]/=n; v[1]/=n; v[2]/=n; }
}

typedef struct {
    int x, y;
    float depth;
    float r, g, b, a;
} sample_t;

static int sample_cmp(const void *a, const void *b) {
    const sample_t *sa = (const sample_t *)a;
    const sample_t *sb = (const sample_t *)b;
    if (sa->depth < sb->depth) return  1;
    if (sa->depth > sb->depth) return -1;
    return 0;
}

int renderer_cpu_render_to_ppm(const scene_t *s, const cpu_render_opts_t *opt,
                               const char *out_path) {
    int W = opt->width;
    int H = opt->height;
    if (W <= 0 || H <= 0) return -1;

    /* View matrix (look-at). */
    float fwd[3], right[3], up[3];
    v3_sub(opt->look_at, opt->cam, fwd);
    v3_norm(fwd);
    v3_cross(fwd, opt->up, right);
    v3_norm(right);
    v3_cross(right, fwd, up);

    float fov_rad = opt->fov_y_deg * (float)M_PI / 180.0f;
    float focal = (float)H * 0.5f / tanf(fov_rad * 0.5f);
    float cx = (float)W * 0.5f;
    float cy = (float)H * 0.5f;

    /* xyz at time t */
    float *pos = (float *)malloc(sizeof(float) * s->N * 3);
    if (!pos) return -1;
    scene_xyz_at_time(s, opt->t, pos);

    /* fallback rgb */
    float *rgb = (float *)malloc(sizeof(float) * s->N * 3);
    if (!rgb) { free(pos); return -1; }
    scene_fallback_rgb(s, rgb);

    /* z-buffered point splat to image */
    float *fb_r = (float *)calloc((size_t)W * H, sizeof(float));
    float *fb_g = (float *)calloc((size_t)W * H, sizeof(float));
    float *fb_b = (float *)calloc((size_t)W * H, sizeof(float));
    float *fb_a = (float *)calloc((size_t)W * H, sizeof(float));
    if (!fb_r || !fb_g || !fb_b || !fb_a) {
        free(pos); free(rgb);
        free(fb_r); free(fb_g); free(fb_b); free(fb_a);
        return -1;
    }
    /* Clear bg dark gray */
    for (int i = 0; i < W*H; ++i) {
        fb_r[i] = 0.05f; fb_g[i] = 0.05f; fb_b[i] = 0.08f; fb_a[i] = 0.0f;
    }

    /* Sort by depth (back-to-front) for alpha-over compositing. */
    sample_t *samples = (sample_t *)malloc(sizeof(sample_t) * s->N);
    if (!samples) {
        free(pos); free(rgb);
        free(fb_r); free(fb_g); free(fb_b); free(fb_a);
        return -1;
    }
    int sn = 0;

    int radius = opt->splat_radius > 0.0f ? (int)opt->splat_radius : 1;

    for (uint32_t i = 0; i < s->N; ++i) {
        float p[3] = { pos[i*3+0] - opt->cam[0],
                       pos[i*3+1] - opt->cam[1],
                       pos[i*3+2] - opt->cam[2] };
        float vx = v3_dot(right, p);
        float vy = v3_dot(up, p);
        float vz = v3_dot(fwd, p);
        if (vz <= 0.001f) continue;

        float sx = cx + (vx / vz) * focal;
        float sy = cy - (vy / vz) * focal;
        int ix = (int)(sx + 0.5f);
        int iy = (int)(sy + 0.5f);
        if (ix < 0 || ix >= W || iy < 0 || iy >= H) continue;

        float opacity = s->opacity.values ? s->opacity.values[i] : 1.0f;
        if (opacity < 0.0f) opacity = 0.0f;
        if (opacity > 1.0f) opacity = 1.0f;

        sample_t sm;
        sm.x = ix; sm.y = iy; sm.depth = vz;
        sm.r = rgb[i*3+0];
        sm.g = rgb[i*3+1];
        sm.b = rgb[i*3+2];
        sm.a = opacity;
        samples[sn++] = sm;
        (void)radius;
    }

    /* Sort descending by depth using qsort + sample_cmp. */
    qsort(samples, sn, sizeof(sample_t), sample_cmp);

    for (int k = 0; k < sn; ++k) {
        sample_t sm = samples[k];
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx*dx + dy*dy > radius*radius) continue;
                int xx = sm.x + dx;
                int yy = sm.y + dy;
                if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;
                int idx = yy * W + xx;
                float a = sm.a;
                fb_r[idx] = a * sm.r + (1 - a) * fb_r[idx];
                fb_g[idx] = a * sm.g + (1 - a) * fb_g[idx];
                fb_b[idx] = a * sm.b + (1 - a) * fb_b[idx];
                fb_a[idx] = a + (1 - a) * fb_a[idx];
            }
        }
    }
    free(samples);

    /* Write PPM */
    FILE *fp = fopen(out_path, "wb");
    if (!fp) {
        free(pos); free(rgb);
        free(fb_r); free(fb_g); free(fb_b); free(fb_a);
        return -1;
    }
    fprintf(fp, "P6\n%d %d\n255\n", W, H);
    for (int i = 0; i < W*H; ++i) {
        uint8_t r8 = (uint8_t)fminf(255.0f, fmaxf(0.0f, fb_r[i] * 255.0f + 0.5f));
        uint8_t g8 = (uint8_t)fminf(255.0f, fmaxf(0.0f, fb_g[i] * 255.0f + 0.5f));
        uint8_t b8 = (uint8_t)fminf(255.0f, fmaxf(0.0f, fb_b[i] * 255.0f + 0.5f));
        uint8_t out_px[3] = { r8, g8, b8 };
        fwrite(out_px, 1, 3, fp);
    }
    fclose(fp);

    free(pos); free(rgb);
    free(fb_r); free(fb_g); free(fb_b); free(fb_a);
    return 0;
}
