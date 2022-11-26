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

    CLProgram renderProgram(cl_context, "render_main");
    CLImageGL *outImage = new CLImageGL(&renderProgram, WINDOW_WIDTH, WINDOW_HEIGHT, MEMORY_WRITE);

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
            outImage = new CLImageGL(&renderProgram, WINDOW_WIDTH, WINDOW_HEIGHT, MEMORY_WRITE);
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

        renderProgram.setArg("render_main", 0, outImage);
        renderProgram.setArg("render_main", 1, renderSize);

        renderProgram.acquireImageGL(outImage);

        if (!renderProgram.callFunction("render_main", WINDOW_WIDTH * WINDOW_HEIGHT)) {
            exit(0);
        }

        renderProgram.releaseImageGL(outImage);

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