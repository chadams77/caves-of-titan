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
    int trace;
    int2 dummy; // 12
} GridCell;

typedef struct __attribute__((packed)) _Trace {
    int num;
    float2 position;
    int dummy; // 4
} Trace;

typedef struct __attribute__((packed)) _Player {
    float2 pos;
    float2 vel; // 4
    float radius;
    float health;
    int moving;
    int dummy; // 8
} Player;

__kernel void clear_grids( __global GridCell * grid,
                           int2 grid_size ) {
    int id = get_global_id(0);
    int n = grid_size.x * grid_size.y;

    if (id < n) {
        __global int * GC = (__global int*)(grid + id);
        GC[0] = GC[1] = GC[2] = GC[3] = GC[4] = GC[5] = GC[6] = GC[7] = 0;
        GC[8] = -1;
        GC[9] = 0;
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

bool collisionDirRock ( __global GridCell * grid, int2 grid_size, float2 pos, float radius, int2 dir ) {

    int xc = (int)floor(pos.x);
    int yc = (int)floor(pos.y);
    int r = (int)ceil(radius + 1.);

    for (int x=xc - r; x<=(xc + r); x++) {
        for (int y=yc - r; y<=(yc + r); y++) {
            if (dir.x < 0 && x >= xc) {
                continue;
            }
            if (dir.y < 0 && y >= yc) {
                continue;
            }
            if (dir.x > 0 && x <= xc) {
                continue;
            }
            if (dir.y > 0 && y <= yc) {
                continue;
            }
            if (x >= 0 && y >= 0 && x < grid_size.x && y < grid_size.y) {
                float dx = ((float)(x) + 0.5) - pos.x, dy = ((float)(y) + 0.5) - pos.y;
                float t = 1. - (dx*dx+dy*dy / radius*radius);
                if (t > 0.) {
                    int grid_index = y * grid_size.x + x;
                    __global int * GC = (__global int*)(grid + grid_index);
                    if (GC[4] > 0) {
                        return true;
                    }
                }
            }
        }
    }

    return false;

}

__kernel void update_trace( __global GridCell * grid,
                            int2 grid_size,
                            __global Trace * trace,
                            int num_trace,
                            float2 world_mouse,
                            float delta_time,
                            float2 player0,
                            float gravity ) {

    int id = get_global_id(0);

    if (id == 0) {

        float traceR = 4.;
        float2 vel = (world_mouse - player0) * (float2)2.;
        float speed = length(vel);
        if (speed > 1.) {
            if (speed > 400.) {
                vel /= speed;
                speed = 400.;
                vel *= speed;
            }

            for (int i=0; i<num_trace; i++) {
                vel.y += gravity * delta_time;
                vel.x -= vel.x * 0.5 * delta_time;
                vel.y -= vel.y * 0.5 * delta_time;

                player0 += vel * delta_time;

                if (vel.y < 0.) {
                    if (collisionDirRock(grid, grid_size, player0, traceR, (int2)(0, -1))) {
                        player0.y += traceR;
                        vel.y = -vel.y * 0.5;
                    }
                }
                else if (vel.y > 0.) {
                    if (collisionDirRock(grid, grid_size, player0, traceR, (int2)(0, 1))) {
                        player0.y -= traceR;
                        vel.y = -vel.y * 0.5;
                        break;
                    }
                }
                if (vel.x < 0.) {
                    if (collisionDirRock(grid, grid_size, player0, traceR, (int2)(-1, 0))) {
                        player0.x += traceR;
                        vel.x = -vel.x * 0.5;
                    }
                }
                else if (vel.x > 0.) {
                    if (collisionDirRock(grid, grid_size, player0, traceR, (int2)(1, 0))) {
                        player0.x -= traceR;
                        vel.x = -vel.x * 0.5;
                    }
                }

                trace[i].num = i;
                trace[i].position = player0;

                int xc = (int)floor(player0.x);
                int yc = (int)floor(player0.y);
                int r = (int)ceil(traceR + 1.);

                for (int x=xc - r; x<=(xc + r); x++) {
                    for (int y=yc - r; y<=(yc + r); y++) {
                        if (x >= 0 && y >= 0 && x < grid_size.x && y < grid_size.y) {
                            float dx = ((float)(x) + 0.5) - player0.x, dy = ((float)(y) + 0.5) - player0.y;
                            float t = 1. - sqrt(dx*dx+dy*dy) / traceR;
                            if (t > 0.) {
                                t = (float)pow((double)t, 0.5);
                                int grid_index = y * grid_size.x + x;
                                __global int * GC = (__global int*)(grid + grid_index);
                                atomic_add(GC + 9, TO_FIXED(0.25));
                            }
                        }
                    }
                }
            }
        }

    }

}

__kernel void update_player( __global GridCell * grid,
                             int2 grid_size,
                             float delta_time,
                             float gravity,
                             __global Player * player ) {

    if (player->moving == 0) {
        return;
    }

    int id = get_global_id(0);

    if (id == 0) {

        float traceR = 4.;
        float2 vel = player->vel;
        float2 player0 = player->pos;

        float dt = delta_time / 10.;

        for (int i=0; i<10; i++) {

            vel.y += gravity * dt;
            vel.x -= vel.x * 0.5 * dt;
            vel.y -= vel.y * 0.5 * dt;

            player0 += vel * dt;

            if (vel.y < 0.) {
                if (collisionDirRock(grid, grid_size, player0, traceR, (int2)(0, -1))) {
                    player0.y += traceR;
                    vel.y = -vel.y * 0.5;
                }
            }
            else if (vel.y > 0.) {
                if (collisionDirRock(grid, grid_size, player0, traceR, (int2)(0, 1))) {
                    player0.y -= traceR;
                    vel.y = -vel.y * 0.5;
                    player->moving = 0;
                    break;
                }
            }
            if (vel.x < 0.) {
                if (collisionDirRock(grid, grid_size, player0, traceR, (int2)(-1, 0))) {
                    player0.x += traceR;
                    vel.x = -vel.x * 0.5;
                }
            }
            else if (vel.x > 0.) {
                if (collisionDirRock(grid, grid_size, player0, traceR, (int2)(1, 0))) {
                    player0.x -= traceR;
                    vel.x = -vel.x * 0.5;
                }
            }

        }

        player->pos = player0;
        player->vel = vel;

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
                            stick += types.x * 100.;
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
            if (P.types.y > 2.5) {
                P.radius -= P.radius * delta_time * 0.02;
                if (P.radius < 1.5f) {
                    P.id = -1;
                }
            }
            else {
                P.radius -= P.radius * delta_time * 0.2;
                if (P.radius < 1.5f) {
                    P.id = -1;
                }
            }
        }
        if (P.heat <= 0.f) {
            P.heat = 0.;
            if (P.types.z > 0.5) {
                P.id = -1;
            }
        }
        if (P.types.x > 0.5 && P.heat > 2.) {
            P.types = (float4)(0., 0., 1., 0.);
        }
        if (P.types.y > 0.5 && P.heat > 0.1) {
            P.heat = 1.;
            P.radius *= 1.75;
            P.mass *= 10.;
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

    if (id < n) {

        int sx = id % render_size.x;
        int sy = (id - sx) / render_size.x;

        float cx = ICAMX(sx);
        float cy = ICAMY(sy);

        int x = (int)round(cx);
        int y = (int)round(cy);

        float yt = 1. - ((float)y / (float)render_size.y) * 0.5;
        float4 clr = (float4)(yt + (1. - yt) * 0.5, yt + (1. - yt) * 0.5, yt + (1. - yt) * 0.1, 1.) * (float4)(0.4);

        if (x >= 0 && y >= 0 && x < grid_size.x && y < grid_size.y) {
            int grid_index = y * grid_size.x + x;
            __global GridCell * GC = grid + grid_index;
            float heat = TO_FLOAT(GC->heat);
            float rocks = TO_FLOAT(GC->types.x);
            float oil = TO_FLOAT(GC->types.y);
            float fire = TO_FLOAT(GC->types.z);
            float water = TO_FLOAT(GC->types.w);
            float trace = TO_FLOAT(GC->trace);

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

            clr.g += trace;
        }

        clr = clamp(clr, (float4)(0.), (float4)(1.));
        write_imagef(out_color, (int2)(sx, sy), clr);

    }
}