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
                           int2 grid_size )
{
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
                           int2 grid_size )
{
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
            for (int y=xc - r; y<=(yc + r); y++) {
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

__kernel void render_main( __write_only image2d_t out_color,
                             int2 render_size,
                             __global GridCell * grid,
                             int2 grid_size )
{
    int id = get_global_id(0);
    int n = render_size.x * render_size.y;

    if (id < n) {

        int x = id % render_size.x;
        int y = (id - x) / render_size.x;

        int grid_index = y * grid_size.x + x;
        __global GridCell * GC = grid + grid_index;
        float heat = TO_FLOAT(GC->heat);

        float4 clr = (float4)(heat, 0.1 * (float)y / (float)render_size.y, 0.1 * (float)x / (float)render_size.x, 1.);

        clr = clamp(clr, (float4)(0.), (float4)(1.));
        write_imagef(out_color, (int2)(x, y), clr);

    }
}