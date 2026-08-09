#ifndef RENDERER_CPU_H
#define RENDERER_CPU_H

#include "scene.h"
#include <stdint.h>

typedef struct {
    int   width;
    int   height;
    float fov_y_deg;     /* vertical field of view (degrees) */
    float cam[3];        /* camera position */
    float look_at[3];    /* target */
    float up[3];         /* up vector (typically 0,1,0) */
    float t;             /* timestep */
    float splat_radius;  /* in pixels; if <=0 use scene scale heuristic */
} cpu_render_opts_t;

/* Render scene as colored disks/points, write to a PPM file. */
int renderer_cpu_render_to_ppm(const scene_t *s, const cpu_render_opts_t *opt,
                               const char *out_path);

#endif
