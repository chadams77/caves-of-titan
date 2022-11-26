#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>

#include "vec_math.h"
#include "cl_wrapper.h"

using std::cerr;
using std::cout;
using std::endl;
using std::map;
using std::vector;
using std::unordered_map;
using std::string;
using std::stringstream;
using std::ifstream;
using std::ofstream;

int main(void)
{
    const GLenum PIXEL_FORMAT = GL_RGBA;
    GLuint WINDOW_WIDTH = 1024;
    GLuint WINDOW_HEIGHT = 1024;
    GLuint DATA_SIZE = WINDOW_WIDTH * WINDOW_HEIGHT * 4;

    GLFWwindow* window;

    if (!glfwInit()) {
        return -1;
    }

    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Caves of Titan", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    CLContext cl_context;

    CLProgram renderProgram(cl_context, "render_main");
    CLImageGL outImage(&renderProgram, WINDOW_WIDTH, WINDOW_HEIGHT, MEMORY_WRITE);

    int index = 0;

    while (!glfwWindowShouldClose(window)) {

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, 1., 0.0, 1., -1.0, 1.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        CLInt2 renderSize(WINDOW_WIDTH, WINDOW_HEIGHT);

        renderProgram.setArg("render_main", 0, &outImage);
        renderProgram.setArg("render_main", 1, renderSize);

        renderProgram.acquireImageGL(&outImage);

        if (!renderProgram.callFunction("render_main", WINDOW_WIDTH * WINDOW_HEIGHT)) {
            exit(0);
        }

        renderProgram.releaseImageGL(&outImage);

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, outImage.glTex);

        glBegin(GL_QUADS);
            glTexCoord2f(0., 0.); glVertex2f(0., 0.);
            glTexCoord2f(0., 1.); glVertex2f(0., 1.);
            glTexCoord2f(1., 1.); glVertex2f(1., 1.);
            glTexCoord2f(1., 0.); glVertex2f(1., 0.);
        glEnd();

        glfwSwapBuffers(window);

        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}