#include "4dgs_format.h"
#include "decoder.h"
#include "encoder.h"
#include "renderer_cpu.h"
#include "scene.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(stderr,
        "Stages of the 4DGS pipeline:\n"
        "  decode      <in.4dgs[.gz]>\n"
        "       parse + validate, print summary\n"
        "  render      <in.4dgs[.gz]> <out.ppm> [t]\n"
        "       decode then render single frame at time t (default 0.0)\n"
        "  encode      <in.4dgs[.gz]> <out.4dgs[.gz]>\n"
        "       decode then re-emit (Stage 5: container/true encoder)\n"
        "       output is gzip-wrapped if name ends with .gz\n"
        "  roundtrip   <in.4dgs[.gz]> <re.4dgs[.gz]>\n"
        "       decode -> encode -> decode -> compare attributes (Stage 6)\n"
        "  render-pair <in.4dgs[.gz]> <out_a.ppm> <out_b.ppm> [t]\n"
        "       Stage 6 visual: render(orig) vs render(round-tripped)\n"
        "Usage: %s <command> [args]\n", argv0);
}

/* ---------------- Stage 6 helpers ---------------- */

static void diff_floats(const float *a, const float *b, size_t n,
                        float *out_max, float *out_mean) {
    float mx = 0.0f, sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
        sum += d;
    }
    *out_max  = mx;
    *out_mean = (n > 0) ? sum / (float)n : 0.0f;
}

static void compare_attr(const char *name, const float *a, const float *b,
                         size_t n) {
    float mx, mn;
    diff_floats(a, b, n, &mx, &mn);
    printf("  %-12s n=%-10zu max_err=%.6g  mean_err=%.6g\n",
           name, n, mx, mn);
}

static int cmd_decode(int argc, char **argv) {
    if (argc < 1) return 2;
    scene_t s; scene_init(&s);
    if (decoder_load_file(argv[0], &s, 1) != 0) {
        scene_free(&s); return 1;
    }
    scene_free(&s);
    return 0;
}

static cpu_render_opts_t default_render_opts(const scene_t *s, float t) {
    cpu_render_opts_t opt;
    opt.width = 512;
    opt.height = 512;
    opt.fov_y_deg = 45.0f;
    opt.up[0] = 0; opt.up[1] = 1; opt.up[2] = 0;
    opt.t = t;
    opt.splat_radius = 1;

    /* Camera: place at scene center + (0, 0, +radius) along z */
    float bounds[6];
    scene_bounds_xyz(s, bounds);
    float cx = 0.5f * (bounds[0] + bounds[3]);
    float cy = 0.5f * (bounds[1] + bounds[4]);
    float cz = 0.5f * (bounds[2] + bounds[5]);
    float ex = bounds[3] - bounds[0];
    float ey = bounds[4] - bounds[1];
    float ez = bounds[5] - bounds[2];
    float diag = sqrtf(ex*ex + ey*ey + ez*ez);
    if (diag < 1e-3f) diag = 5.0f;
    opt.cam[0]    = cx;
    opt.cam[1]    = cy;
    opt.cam[2]    = cz + diag * 1.2f;
    opt.look_at[0]= cx;
    opt.look_at[1]= cy;
    opt.look_at[2]= cz;
    return opt;
}

static int cmd_render(int argc, char **argv) {
    if (argc < 2) { usage("4dgs"); return 2; }
    const char *in_path = argv[0];
    const char *out_path = argv[1];
    float t = (argc >= 3) ? (float)atof(argv[2]) : 0.0f;

    scene_t s; scene_init(&s);
    if (decoder_load_file(in_path, &s, 1) != 0) {
        scene_free(&s); return 1;
    }
    cpu_render_opts_t opt = default_render_opts(&s, t);
    if (renderer_cpu_render_to_ppm(&s, &opt, out_path) != 0) {
        fprintf(stderr, "render failed\n");
        scene_free(&s);
        return 1;
    }
    fprintf(stderr, "[render] wrote %s (t=%.3f, %dx%d)\n",
            out_path, t, opt.width, opt.height);
    scene_free(&s);
    return 0;
}

static int cmd_encode(int argc, char **argv) {
    if (argc < 2) { usage("4dgs"); return 2; }
    const char *in_path = argv[0];
    const char *out_path = argv[1];
    int gz = 0;
    size_t L = strlen(out_path);
    if (L >= 3 && strcmp(out_path + L - 3, ".gz") == 0) gz = 1;

    scene_t s; scene_init(&s);
    if (decoder_load_file(in_path, &s, 1) != 0) {
        scene_free(&s); return 1;
    }
    if (encoder_write_file(&s, out_path, gz) != 0) {
        fprintf(stderr, "encode failed\n");
        scene_free(&s); return 1;
    }
    fprintf(stderr, "[encode] wrote %s (gzip=%d)\n", out_path, gz);
    scene_free(&s);
    return 0;
}

static int cmd_roundtrip(int argc, char **argv) {
    if (argc < 2) { usage("4dgs"); return 2; }
    const char *in_path = argv[0];
    const char *re_path = argv[1];
    int gz = 0;
    size_t L = strlen(re_path);
    if (L >= 3 && strcmp(re_path + L - 3, ".gz") == 0) gz = 1;

    scene_t a; scene_init(&a);
    fprintf(stderr, "=== decode original ===\n");
    if (decoder_load_file(in_path, &a, 1) != 0) {
        scene_free(&a); return 1;
    }
    fprintf(stderr, "=== encode -> %s ===\n", re_path);
    if (encoder_write_file(&a, re_path, gz) != 0) {
        scene_free(&a); return 1;
    }
    scene_t b; scene_init(&b);
    fprintf(stderr, "=== decode re-encoded ===\n");
    if (decoder_load_file(re_path, &b, 1) != 0) {
        scene_free(&a); scene_free(&b); return 1;
    }
    if (a.N != b.N) {
        fprintf(stderr, "N mismatch: orig=%u re=%u\n", a.N, b.N);
        scene_free(&a); scene_free(&b); return 1;
    }

    printf("\n=== Stage 6 attribute comparison (orig vs re-decoded) ===\n");
    compare_attr("xyz",      a.xyz,      b.xyz,      (size_t)a.N * 3);
    compare_attr("motion",   a.motion,   b.motion,   (size_t)a.N * 9);
    compare_attr("opacity",  a.opacity.values, b.opacity.values, a.N);
    compare_attr("tcen",     a.tcen.values,    b.tcen.values,    a.N);
    compare_attr("tsca",     a.tsca.values,    b.tsca.values,    a.N);
    compare_attr("scale",    a.scale.values,    b.scale.values,    (size_t)a.N * 3);
    compare_attr("rotation", a.rotation.values, b.rotation.values, (size_t)a.N * 4);
    compare_attr("omega",    a.omega.values,    b.omega.values,    (size_t)a.N * 4);
    compare_attr("tfea",     a.tfea.values,     b.tfea.values,     (size_t)a.N * 3);
    if (a.has_features && b.has_features) {
        compare_attr("features_dc", a.features_dc, b.features_dc, (size_t)a.N * 6);
    }
    if (a.has_rgb_dec && b.has_rgb_dec) {
        compare_attr("rgb_w1", a.rgb_w1, b.rgb_w1, FOURDGS_RGB_W1_COUNT);
        compare_attr("rgb_w2", a.rgb_w2, b.rgb_w2, FOURDGS_RGB_W2_COUNT);
    }
    printf("\nNote: non-zero errors are expected. Sources of loss:\n"
           "  - opacity/tcen/tsca: 8-bit min-max requantization (~step/2)\n"
           "  - scale/rotation/omega/tfea: greedy nearest-neighbor RVQ "
           "re-assignment\n"
           "  - features_dc / rgb_dec / xyz / motion: f16 round-trip "
           "(should be 0)\n");

    scene_free(&a); scene_free(&b);
    return 0;
}

static int cmd_render_pair(int argc, char **argv) {
    if (argc < 3) { usage("4dgs"); return 2; }
    const char *in_path = argv[0];
    const char *out_a = argv[1];
    const char *out_b = argv[2];
    float t = (argc >= 4) ? (float)atof(argv[3]) : 0.0f;

    scene_t a; scene_init(&a);
    if (decoder_load_file(in_path, &a, 1) != 0) { scene_free(&a); return 1; }
    cpu_render_opts_t opt = default_render_opts(&a, t);
    if (renderer_cpu_render_to_ppm(&a, &opt, out_a) != 0) {
        fprintf(stderr, "render orig failed\n"); scene_free(&a); return 1;
    }
    fprintf(stderr, "[render-pair] wrote %s (orig)\n", out_a);

    /* round trip in memory */
    uint8_t *body = NULL; size_t body_n = 0;
    if (encoder_emit_4dgs(&a, &body, &body_n) != 0) {
        scene_free(&a); return 1;
    }
    scene_t b; scene_init(&b);
    if (decoder_parse_4dgs(body, body_n, &b, 0) != 0) {
        free(body); scene_free(&a); return 1;
    }
    free(body);
    if (renderer_cpu_render_to_ppm(&b, &opt, out_b) != 0) {
        fprintf(stderr, "render re failed\n");
        scene_free(&a); scene_free(&b); return 1;
    }
    fprintf(stderr, "[render-pair] wrote %s (round-tripped)\n", out_b);

    /* show overall pixel diff stats by re-running decoder pixel diff */
    scene_free(&a); scene_free(&b);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 2; }
    const char *cmd = argv[1];
    int sub_argc = argc - 2;
    char **sub_argv = argv + 2;

    if (strcmp(cmd, "decode") == 0)      return cmd_decode(sub_argc, sub_argv);
    if (strcmp(cmd, "render") == 0)      return cmd_render(sub_argc, sub_argv);
    if (strcmp(cmd, "encode") == 0)      return cmd_encode(sub_argc, sub_argv);
    if (strcmp(cmd, "roundtrip") == 0)   return cmd_roundtrip(sub_argc, sub_argv);
    if (strcmp(cmd, "render-pair") == 0) return cmd_render_pair(sub_argc, sub_argv);

    usage(argv[0]);
    return 2;
}
