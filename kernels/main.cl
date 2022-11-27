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
    int maxID;
    int3 dummy; // 12
} GridCell;

__kernel void clear_grids( __global GridCell * grid,
                           int2 grid_size ) {
    int id = get_global_id(0);
    int n = grid_size.x * grid_size.y;

    if (id < n) {
        __global int * GC = (__global int*)(grid + id);
        GC[0] = GC[1] = GC[2] = GC[3] = GC[4] = GC[5] = GC[6] = GC[7] = 0;
        GC[8] = -1;
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
        int r = (int)ceil(P.radius + 1.);

        for (int x=xc - r; x<=(xc + r); x++) {
            for (int y=yc - r; y<=(yc + r); y++) {
                if (x >= 0 && y >= 0 && x < grid_size.x && y < grid_size.y) {
                    float dx = ((float)(x) + 0.5) - P.position.x, dy = ((float)(y) + 0.5) - P.position.y;
                    float t = 1. - sqrt(dx*dx+dy*dy) / P.radius;
                    if (t > 0.) {
                        t = (float)pow((double)t, 0.5);
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
                        atomic_max(GC + 8, P.id);
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

        if (P.types.z > 0.5) {
            P.velocity.y -= 0.5 * gravity * delta_time;
        }
        else {
            P.velocity.y += gravity * delta_time;
        }
        P.velocity.x -= P.velocity.x * P.radius / P.mass * delta_time;
        P.velocity.y -= P.velocity.y * P.radius / P.mass * delta_time;

        int xc = (int)floor(P.position.x);
        int yc = (int)floor(P.position.y);
        int r = (int)ceil(P.radius + 0.5);

        float wPressX = 0.;
        float wPressY = 0.;
        float totalHeat = 0.;
        float stick = 0.;
        float totalT = 0.;

        float myHeat = P.heat * sqrt(P.velocity.x * P.velocity.x + P.velocity.y * P.velocity.y) * P.mass / 10.f;

        for (int x=xc - r; x<=(xc + r); x++) {
            for (int y=yc - r; y<=(yc + r); y++) {
                if (x >= 0 && y >= 0 && x < grid_size.x && y < grid_size.y) {
                    float dx = (float)(x) - floor(P.position.x), dy = (float)(y) - floor(P.position.y);
                    float t = 1. - sqrt(dx*dx+dy*dy) / P.radius;
                    if (t > 0. && t < 1.) {
                        int grid_index = y * grid_size.x + x;
                        __global int * GC = (__global int*)(grid + grid_index);
                        float mass = TO_FLOAT(GC[0]) * t;
                        float heat = TO_FLOAT(GC[1]) * t;
                        float2 velocity = (float2)(TO_FLOAT(GC[2]) * t, TO_FLOAT(GC[3]) * t);
                        float4 types = (float4)(TO_FLOAT(GC[4]) * t, TO_FLOAT(GC[5]) * t, TO_FLOAT(GC[6]) * t, TO_FLOAT(GC[7]) * t);
                        int maxID = GC[8];

                        totalHeat += heat * sqrt(velocity.x * velocity.x + velocity.y * velocity.y) * mass / 10.f;

                        totalT += t;

                        if (types.x > 0.01) {
                            mass *= 100.;
                            stick += types.x * 10.;
                        }
                        if (types.y > 0.01) {
                            stick += types.y * 0.025;
                        }
                        if (types.z > 0.01) {
                            stick += types.y * 0.01;
                        }

                        if (fabs(dx - 0.f) < 0.0001f) {
                            int r1 = (x + maxID) % 13;
                            dx += (r1 / 12.) * 0.8 - 0.4;
                        }

                        if (fabs(dy - 0.f) < 0.0001f) {
                            int r1 = (y + maxID) % 13;
                            dy += (r1 / 12.) * 0.8 - 0.4;
                        }

                        wPressX += -dx * mass;
                        wPressY += -dy * mass;
                    }
                }
            }
        }

        float avgHeat = totalHeat / totalT;

        P.heat += (avgHeat * 0.5 - myHeat) * delta_time * 8.;
        if (P.heat > 8.) {
            P.heat = 8.;
        }
        P.heat -= delta_time * max(P.heat, 1.f);
        if (P.types.z > 0.5) {
            P.radius -= P.radius * delta_time;
            if (P.radius < 0.01f) {
                P.id = -1;
            }
        }
        if (P.types.y > 1.5) {
            P.radius -= P.radius * delta_time * 0.1;
            if (P.radius < 1.5f) {
                P.id = -1;
            }
        }
        if (P.heat <= 0.f) {
            P.heat = 0.;
            if (P.types.z > 0.5) {
                P.id = -1;
            }
        }
        if (P.types.x > 0.5 && P.heat > 5.) {
            P.types = (float4)(0., 0., 1., 0.);
        }
        if (P.types.y > 0.5 && P.heat > 0.1) {
            P.heat = 1.;
            P.radius *= 1.75;
            P.types = (float4)(0., 0., 1., 0.);
        }

        wPressX /= P.mass;
        wPressY /= P.mass;

        P.velocity.x += wPressX * delta_time;
        P.velocity.y += wPressY * delta_time;

        if (totalT) {
            stick /= totalT;
            if (stick > 1.) {
                stick = 1.;
            }

            P.velocity.x -= stick / 10. * P.velocity.x;
            P.velocity.y -= stick / 10. * P.velocity.y;
        }

        if (P.types.x > 0.5 && P.heat < 1.) {
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
        float4 clr = (float4)(yt + (1. - yt) * 0.5, yt + (1. - yt) * 0.5, yt + (1. - yt) * 0.1, 1.) * (float4)(0.4);

        if (x >= 0 && y >= 0 && x < grid_size.x && y < grid_size.y) {
            int grid_index = y * grid_size.x + x;
            __global GridCell * GC = grid + grid_index;
            float heat = TO_FLOAT(GC->heat);
            float rocks = TO_FLOAT(GC->types.x);
            float oil = TO_FLOAT(GC->types.y);
            float fire = TO_FLOAT(GC->types.z);
            float water = TO_FLOAT(GC->types.w);

            if (rocks > 0.0) {
                int rand1 = (GC->maxID * 17) % 3;
                if (rand1 == 0) {
                    clr.x = 0.366;
                    clr.y = 0.289;
                    clr.z = 0.289;
                }
                else if (rand1 == 1) {
                    clr.x = 0.511;
                    clr.y = 0.429;
                    clr.z = 0.428;
                }
                else if (rand1 == 2) {
                    clr.x = 0.444;
                    clr.y = 0.364;
                    clr.z = 0.256;
                }
                clr.xyz *= (float3)(min(rocks / 2.5f, 1.f));
            }

            if (oil > 0.5) {
                float3 t = clamp((float3)oil / 2.5f, (float3)0., (float3)1.);
                clr.xyz = ((float3)1. - t) * clr.xyz;
            }

            float heatT = clamp(heat / 10.f, 0.f, 1.f);
            clr.x += min(heatT * 4., 1.);
            clr.y += min(heatT * 2., 1.);
            clr.z += min(heatT * 1., 1.);
        }

        clr = clamp(clr, (float4)(0.), (float4)(1.));
        write_imagef(out_color, (int2)(sx, sy), clr);

    }
}