__kernel void render_main( __write_only image2d_t out_color,
                             int2 render_size)
{
    int id = get_global_id(0);
    int n = render_size.x * render_size.y;

    if (id < n) {

        int x = id % render_size.x;
        int y = (id - x) / render_size.x;

        float4 clr = (float4)(1., 0., 0., 1.);

        clr = clamp(clr, (float4)(0.), (float4)(1.));
        write_imagef(out_color, (int2)(x, y), clr);

    }
}