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

GLuint WINDOW_WIDTH = 1024;
GLuint WINDOW_HEIGHT = 1024;
GLuint DATA_SIZE = WINDOW_WIDTH * WINDOW_HEIGHT * 4;
bool WINDOW_RESIZED = false;
bool FULLSCREEN = false;

map<GLuint, bool> keyDown, lastKeyDown;

// Callbacks
void onWindowResize (GLFWwindow* window, int width, int height)
{
    WINDOW_WIDTH = (GLuint)width;
    WINDOW_HEIGHT = (GLuint)height;
    WINDOW_RESIZED = true;
}

void onKeyboard (GLFWwindow* window, int key, int scancode, int action, int mods)
{
    keyDown[key] = action == GLFW_PRESS;
}
////////////

// Mirror classes for transfering data to/from kernels
class GridCell {
    CLInt mass;
    CLInt heat;
    CLInt2 velocity; // 4
    CLInt4 types; // 8 // x:rock, y:oil, z:fire/smoke, w:water/steam
    GridCell() { }
};

class Particle {
    CLInt id;
    CLFloat2 position;
    CLFloat radius; // 4
    CLFloat2 velocity;
    CLFloat mass;
    CLFloat heat; // 8
    CLFloat4 types; // 12 // x:rock, y:oil, z:fire/smoke, w:water/steam
    Particle() {
        id = -1;
    }
};

CLInt NUM_PARTICLES = 32768;
CLInt2 GRID_SIZE(2048, 2048);
////////////

int main(void)
{
    const GLenum PIXEL_FORMAT = GL_RGBA;

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

    glfwSetWindowSizeCallback(window, onWindowResize);
    glfwSetKeyCallback(window, onKeyboard);

    CLContext cl_context;

    CLProgram program(cl_context, "main");

    CLImageGL *outImage = new CLImageGL(&program, WINDOW_WIDTH, WINDOW_HEIGHT, MEMORY_WRITE);

    CLBuffer particleBfr(&program, NUM_PARTICLES, sizeof(Particle), MEMORY_READ_WRITE);
    CLBuffer gridBfr(&program, GRID_SIZE.x * GRID_SIZE.y, sizeof(GridCell), MEMORY_READ_WRITE);

    particleBfr.writeSync();
    gridBfr.writeSync();

    GLFWmonitor * monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode * mode = glfwGetVideoMode(monitor);

    double deltaTime = 1. / (double)(mode->refreshRate);

    while (!glfwWindowShouldClose(window)) {

        if (lastKeyDown[GLFW_KEY_F11] && !keyDown[GLFW_KEY_F11]) {
            if (!FULLSCREEN) {
                glfwMaximizeWindow(window);
                mode = glfwGetVideoMode(monitor);
                deltaTime = 1. / (double)(mode->refreshRate);
                glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
                FULLSCREEN = true;
            }
            else {
                mode = glfwGetVideoMode(monitor);
                deltaTime = 1. / (double)(mode->refreshRate);
                glfwSetWindowMonitor(window, NULL, 0, 0, mode->width, mode->height, mode->refreshRate);
                glfwRestoreWindow(window);
                FULLSCREEN = false;
            }
        }

        if (WINDOW_RESIZED) {
            delete outImage;
            outImage = new CLImageGL(&program, WINDOW_WIDTH, WINDOW_HEIGHT, MEMORY_WRITE);
            glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
            WINDOW_RESIZED = false;
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, 1., 0.0, 1., -1.0, 1.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        CLInt2 renderSize(WINDOW_WIDTH, WINDOW_HEIGHT);
        
        program.setArg("update_grids", 0, &particleBfr);
        program.setArg("update_grids", 1, &gridBfr);
        program.setArg("update_grids", 2, NUM_PARTICLES);
        program.setArg("update_grids", 3, GRID_SIZE);

        program.setArg("clear_grids", 0, &gridBfr);
        program.setArg("clear_grids", 1, GRID_SIZE);

        program.setArg("render_main", 0, outImage);
        program.setArg("render_main", 1, renderSize);
        program.setArg("render_main", 2, &gridBfr);
        program.setArg("render_main", 3, GRID_SIZE);

        program.acquireImageGL(outImage);

        if (!program.callFunction("clear_grids", GRID_SIZE.x * GRID_SIZE.y)) {
            exit(0);
        }

        if (!program.callFunction("update_grids", NUM_PARTICLES)) {
            exit(0);
        }

        if (!program.callFunction("render_main", WINDOW_WIDTH * WINDOW_HEIGHT)) {
            exit(0);
        }

        program.releaseImageGL(outImage);

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, outImage->glTex);

        glBegin(GL_QUADS);
            glTexCoord2f(0., 0.); glVertex2f(0., 0.);
            glTexCoord2f(0., 1.); glVertex2f(0., 1.);
            glTexCoord2f(1., 1.); glVertex2f(1., 1.);
            glTexCoord2f(1., 0.); glVertex2f(1., 0.);
        glEnd();

        glfwSwapBuffers(window);

        lastKeyDown = keyDown;

        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}