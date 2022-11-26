#define FP_SCALE 1000.
#define TO_FIXED(_X) ((int)((_X) * FP_SCALE))
#define TO_FLOAT(_X) ((float)((_X) / FP_SCALE))

typedef struct __attribute__((packed)) _Particle {
    int id;
    float2 position;
    float radius; // 4
    float2 velocity;
    float mass;
    float heat; // 8
    float4 types; // 12 // x:rock, y:oil, z:fire/smoke, w:water/steam
} Particle;

typedef struct __attribute__((packed)) _GridCell {
    int mass;
    int heat;
    int2 velocity; // 4
    int4 types; // 8 // x:rock, y:oil, z:fire/smoke, w:water/steam
} GridCell;

__kernel void clear_grids( __global GridCell * grid,
                           int2 grid_size ) {
    int id = get_global_id(0);
    int n = grid_size.x * grid_size.y;

    if (id < n) {
        __global int * GC = (__global int*)(grid + id);
        GC[0] = GC[1] = GC[2] = GC[3] = GC[4] = GC[5] = GC[6] = GC[7] = 0;
    }
}

__kernel void update_grids( __global Particle * particles,
                            __global GridCell * grid,
                           int num_particles,
                           int2 grid_size ) {
    int id = get_global_id(0);

    if (id < num_particles) {

        Particle P = particles[id];
        if (P.id < 0) {
            return;
        }
        int xc = (int)floor(P.position.x);
        int yc = (int)floor(P.position.y);
        int r = (int)ceil(P.radius + 0.5);

        for (int x=xc - r; x<=(xc + r); x++) {
            for (int y=yc - r; y<=(yc + r); y++) {
                if (x >= 0 && y >= 0 && x < grid_size.x && y < grid_size.y) {
                    float dx = ((float)(x) + 0.5) - P.position.x, dy = ((float)(y) + 0.5) - P.position.y;
                    float t = 1. - sqrt(dx*dx+dy*dy) / P.radius;
                    if (t > 0.) {
                        int grid_index = y * grid_size.x + x;
                        __global int * GC = (__global int*)(grid + grid_index);
                        atomic_add(GC + 0, TO_FIXED(P.mass * t));
                        atomic_add(GC + 1, TO_FIXED(P.heat * t));
                        atomic_add(GC + 2, TO_FIXED(P.velocity.x * t));
                        atomic_add(GC + 3, TO_FIXED(P.velocity.y * t));
                        atomic_add(GC + 4, TO_FIXED(P.types.x * t));
                        atomic_add(GC + 5, TO_FIXED(P.types.y * t));
                        atomic_add(GC + 6, TO_FIXED(P.types.z * t));
                        atomic_add(GC + 7, TO_FIXED(P.types.w * t));
                    }
                }
            }
        }

    }
}

__kernel void update_particles( __global Particle * particles,
                                __global GridCell * grid,
                                int num_particles,
                                int2 grid_size,
                                float delta_time,
                                float gravity ) {
    int id = get_global_id(0);

    if (id < num_particles) {

        Particle P = particles[id];
        if (P.id < 0) {
            return;
        }

        P.velocity.x -= P.velocity.x * 0.1 * delta_time;
        P.velocity.y -= P.velocity.y * 0.1 * delta_time;
        P.velocity.y += gravity * delta_time;

        int xc = (int)floor(P.position.x);
        int yc = (int)floor(P.position.y);
        int r = (int)ceil(P.radius + 0.5);

        float wPressX = 0.;
        float wPressY = 0.;
        float totalHeat = 0.;

        for (int x=xc - r; x<=(xc + r); x++) {
            for (int y=yc - r; y<=(yc + r); y++) {
                if (x >= 0 && y >= 0 && x < grid_size.x && y < grid_size.y) {
                    float dx = ((float)(x) + 0.5) - P.position.x, dy = ((float)(y) + 0.5) - P.position.y;
                    float t = 1. - sqrt(dx*dx+dy*dy) / P.radius;
                    if (t > 0.) {
                        int grid_index = y * grid_size.x + x;
                        __global int * GC = (__global int*)(grid + grid_index);
                        float mass = TO_FLOAT(GC[0]) * t;
                        float heat = TO_FLOAT(GC[1]) * t;
                        float2 velocity = (float2)(TO_FLOAT(GC[2]) * t, TO_FLOAT(GC[3]) * t);
                        float4 types = (float4)(TO_FLOAT(GC[4]) * t, TO_FLOAT(GC[5]) * t, TO_FLOAT(GC[6]) * t, TO_FLOAT(GC[7]) * t);

                        totalHeat += heat;

                        wPressX += -dx * mass;
                        wPressY += -dy * mass;
                    }
                }
            }
        }

        P.heat += (totalHeat / (M_PI*r*r) - P.heat) * delta_time * 0.1;
        P.heat -= P.heat * 0.025 * delta_time;
        if (P.heat <= 0.) {
            P.heat = 0.;
            if (P.types.z > 0.5) {
                P.id = -1;
            }
        }
        if (P.types.x > 0.5 && P.heat > 10.) {
            P.types = (float4)(0., 1., 0., 0.);
        }

        wPressX /= P.mass;
        wPressY /= P.mass;

        P.velocity.x += wPressX * delta_time;
        P.velocity.y += wPressY * delta_time;

        if (P.types.x > 0.5) {
            P.velocity = (float2)(0., 0.);
        }

        P.position.x += P.velocity.x * delta_time;
        P.position.y += P.velocity.y * delta_time;

        if (P.position.y >= (float)grid_size.y) {
            P.id = -1;
        }

        particles[id] = P;

    }                                    
}

#define CAMX(_X) (((float)(_X) - (float)camera.x) / camera.z + ((float)render_size.x) * 0.5)
#define CAMY(_X) (((float)(_X) - (float)camera.y) / camera.z + ((float)render_size.y) * 0.5)
#define ICAMX(_X) (((float)(_X) - 0.5 * (float)render_size.x) * camera.z + ((float)camera.x))
#define ICAMY(_X) (((float)(_X) - 0.5 * (float)render_size.y) * camera.z + ((float)camera.y))

__kernel void render_main( __write_only image2d_t out_color,
                             int2 render_size,
                             __global GridCell * grid,
                             int2 grid_size,
                             float3 camera ) {
    int id = get_global_id(0);
    int n = render_size.x * render_size.y;

    for (int k=0; k<2; k++) {
        if (CAMX(0) >= 0 && CAMX(grid_size.x) <= (float)(render_size.x)) {
            camera.x = 0.5 * (float)grid_size.x;
            camera.z = min(camera.z, ((float)grid_size.x) / ((float)render_size.x));
            break;
        }
        else if (CAMX(0) > 0) {
            camera.x += CAMX(0);
        }
        else if (CAMX(grid_size.x) < (float)(grid_size.x)) {
            camera.x -= (float)(grid_size.x) - CAMX(grid_size.x);
        }
    }

    for (int k=0; k<2; k++) {
        if (CAMY(0) >= 0 && CAMY(grid_size.y) <= (float)(render_size.y)) {
            camera.y = 0.5 * (float)grid_size.y;
            camera.z = min(camera.z, ((float)grid_size.y) / ((float)render_size.y));
            break;
        }
        else if (CAMY(0) > 0) {
            camera.y += CAMY(0);
        }
        else if (CAMY(grid_size.y) < (float)(grid_size.y)) {
            camera.y -= (float)(grid_size.y) - CAMY(grid_size.y);
        }
    }

    if (id < n) {

        int sx = id % render_size.x;
        int sy = (id - sx) / render_size.x;

        float cx = ICAMX(sx);
        float cy = ICAMY(sy);

        int x = (int)round(cx);
        int y = (int)round(cy);

        float yt = 1. - (float)y / (float)render_size.y;
        float4 clr = (float4)(yt + (1. - yt) * 0.5, yt + (1. - yt) * 0.5, yt + (1. - yt) * 0.1, 1.) * (float4)(0.2);

        if (x >= 0 && y >= 0 && x < grid_size.x && y < grid_size.y) {
            int grid_index = y * grid_size.x + x;
            __global GridCell * GC = grid + grid_index;
            float heat = TO_FLOAT(GC->heat);
            float rocks = TO_FLOAT(GC->types.x);
            float oil = TO_FLOAT(GC->types.y);
            float fire = TO_FLOAT(GC->types.z);
            float water = TO_FLOAT(GC->types.w);

            if (rocks > 0.5) {
                clr.x = 0.4;
                clr.y = 0.5;
                clr.z = 0.35;
            }

            clr.x += heat / 5.;
        }

        clr = clamp(clr, (float4)(0.), (float4)(1.));
        write_imagef(out_color, (int2)(sx, sy), clr);

    }
}