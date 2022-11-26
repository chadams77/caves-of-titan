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
using std::string;

// Mirror classes for transfering data to/from kernels
class GridCell {
public:
    CLInt mass;
    CLInt heat;
    CLInt2 velocity; // 4
    CLInt4 types; // 8 // x:rock, y:oil, z:fire/smoke, w:water/steam
    GridCell() { }
};

class Particle {
public:
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

CLInt NUM_PARTICLES = 512 * 512;
CLInt2 GRID_SIZE(2048, 2048);
CLFloat GRAVITY = 8.;
CLFloat3 CAMERA;

////////////

CLContext * clContext;
CLProgram * program;
CLImageGL * outImage;
CLBuffer * particleBfr;
CLBuffer * gridBfr;
GLFWwindow * window;
GLFWmonitor * monitor;
const GLFWvidmode * mode;
double deltaTime = 1. / 60.;
double time = 0.;
int newParticleIndex = 0;
bool anyParticlesAdded = false;

GLuint WINDOW_WIDTH = 1024;
GLuint WINDOW_HEIGHT = 1024;
GLuint DATA_SIZE = WINDOW_WIDTH * WINDOW_HEIGHT * 4;
bool WINDOW_RESIZED = false;
bool FULLSCREEN = false;

map<GLuint, bool> keyDown, lastKeyDown;

//////////

void onWindowResize (GLFWwindow* window, int width, int height) {
    WINDOW_WIDTH = (GLuint)width;
    WINDOW_HEIGHT = (GLuint)height;
    WINDOW_RESIZED = true;
}

void onKeyboard (GLFWwindow* window, int key, int scancode, int action, int mods) {
    keyDown[key] = action == GLFW_PRESS;
}

void setFullscreen (bool flag) {
    if (flag) {
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

void handleWindowResize () {
    if (WINDOW_RESIZED) {
        delete outImage;
        outImage = new CLImageGL(program, WINDOW_WIDTH, WINDOW_HEIGHT, MEMORY_WRITE);
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        WINDOW_RESIZED = false;
    }
}

void clearParticles () {
    Particle * data = new Particle[NUM_PARTICLES];
    for (size_t i=0; i<NUM_PARTICLES; i++) {
        data[i].id = -1;
    }
    particleBfr->writeSync(0, NUM_PARTICLES * sizeof(Particle), (void *)data);
    newParticleIndex = 0;
}

void addParticle (Particle & P) {
    anyParticlesAdded = true;
    P.id = newParticleIndex;
    particleBfr->writeSync(newParticleIndex * sizeof(Particle), sizeof(Particle), (void *)&P);
    newParticleIndex += 1;
    if (newParticleIndex >= NUM_PARTICLES) {
        newParticleIndex = 0;
    }
}

void addParticles (Particle * data, int count) {
    for (size_t i=0; i<count; i++) {
        data[i].id = (newParticleIndex + i) % NUM_PARTICLES;
    }
    if ((newParticleIndex + count) < NUM_PARTICLES) {
        particleBfr->writeSync(newParticleIndex * sizeof(Particle), count * sizeof(Particle), (void *)data);
    }
    else {
        int over = (newParticleIndex + count) - NUM_PARTICLES;
        particleBfr->writeSync(newParticleIndex * sizeof(Particle), (count - over) * sizeof(Particle), (void *)data);
        particleBfr->writeSync(0, over * sizeof(Particle), (void *)(data + (count - over)));
    }
    newParticleIndex = (newParticleIndex + count) % NUM_PARTICLES;
}

#define RAND ((float)(rand() % 12347) / 12347.)

void initLevel() {
    clearParticles();

    int size = 512;
    int * grid = new int[size * size];
    int * grid2 = new int[size * size];
    for (int x=0; x<size; x++) {
        for (int y=0; y<size; y++) {
            grid[x + y * size] = (RAND < 0.49 || x <= 5 || y <= 5 || x >= (size - 5) || y >= (size - 5)) ? 1 : 0;
        }
    }
    for (int k=0; k<20; k++) {
        for (int x=0; x<size; x++) {
            for (int y=0; y<size; y++) {
                int off = x + y * size;
                int ncount = 0;
                for (int dx=-1; dx<=1; dx++) {
                    for (int dy=-1; dy<=1; dy++) {
                        int nx = x + dx, ny = y + dy;
                        ncount += (nx < 0 || ny < 0 || nx >= size || ny >= size || grid[nx+ny*size] == 1) ? 1 : 0;
                    }
                }
                if (ncount == 5) {
                    grid2[off] = grid[off];
                }
                else if (ncount > 5) {
                    grid2[off] = 1;
                }
                else {
                    grid2[off] = 0;
                }
            }
        }
        int * tmp = grid;
        grid = grid2;
        grid2 = tmp;
    }
    int count = 0;
    for (int x=0; x<size; x++) {
        for (int y=0; y<size; y++) {
            int rock = grid[x + y*size];
            count += rock > 0 ? 1 : 0;
        }
    }
    Particle * newPrt = new Particle[count];
    int idx = 0;
    for (int x=0; x<size; x++) {
        for (int y=0; y<size; y++) {
            int rock = grid[x + y*size];
            if (rock) {
                Particle P;
                P.position.x = ((float)x + 0.5f) / (float)size * (float)GRID_SIZE.x;
                P.position.y = ((float)y + 0.5f) / (float)size * (float)GRID_SIZE.y;
                P.velocity.x = P.velocity.y = 0.;
                P.heat = 0.;
                P.mass = 100.;
                P.radius = (float)GRID_SIZE.x / (float)size;
                P.types.x = 1.;
                P.types.y = 0.;
                P.types.z = 0.;
                P.types.w = 0.;
                newPrt[idx] = P;
                idx ++;
            }
        }
    }
    addParticles(newPrt, count);
    delete newPrt;
    delete grid;
    delete grid2;
    time = 0.;
}

int main (void)
{
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

    clContext = new CLContext();

    program = new CLProgram(clContext, "main");

    outImage = new CLImageGL(program, WINDOW_WIDTH, WINDOW_HEIGHT, MEMORY_WRITE);

    particleBfr = new CLBuffer(program, NUM_PARTICLES, sizeof(Particle), MEMORY_READ_WRITE);
    gridBfr     = new CLBuffer(program, GRID_SIZE.x * GRID_SIZE.y, sizeof(GridCell), MEMORY_READ_WRITE);

    particleBfr->writeSync();
    gridBfr->writeSync();

    monitor = glfwGetPrimaryMonitor();
    mode = glfwGetVideoMode(monitor);

    deltaTime = 1. / (double)(mode->refreshRate);

    initLevel();

    CAMERA.x = (float)GRID_SIZE.x * 0.5;
    CAMERA.y = (float)GRID_SIZE.y * 0.5;
    CAMERA.z = 1.;

    while (!glfwWindowShouldClose(window)) {

        if (lastKeyDown[GLFW_KEY_F11] && !keyDown[GLFW_KEY_F11]) {
            setFullscreen(!FULLSCREEN);
        }

        handleWindowResize();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, 1., 0.0, 1., -1.0, 1.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        float _r = sin(time) * 0.4 + 1.3;
        float _a = (time * 1.37);
        CAMERA.x = (float)GRID_SIZE.x * 0.5 + cos(_a) * _r * 512.;
        CAMERA.y = (float)GRID_SIZE.y * 0.5 + sin(_a) * _r * 512.;

        CLInt2 renderSize(WINDOW_WIDTH, WINDOW_HEIGHT);
        
        program->setArg("update_grids", 0, particleBfr);
        program->setArg("update_grids", 1, gridBfr);
        program->setArg("update_grids", 2, NUM_PARTICLES);
        program->setArg("update_grids", 3, GRID_SIZE);

        program->setArg("clear_grids", 0, gridBfr);
        program->setArg("clear_grids", 1, GRID_SIZE);

        program->setArg("render_main", 0, outImage);
        program->setArg("render_main", 1, renderSize);
        program->setArg("render_main", 2, gridBfr);
        program->setArg("render_main", 3, GRID_SIZE);
        program->setArg("render_main", 4, CAMERA);

        program->setArg("update_particles", 0, particleBfr);
        program->setArg("update_particles", 1, gridBfr);
        program->setArg("update_particles", 2, NUM_PARTICLES);
        program->setArg("update_particles", 3, GRID_SIZE);
        program->setArg("update_particles", 4, (CLFloat)deltaTime);
        program->setArg("update_particles", 5, GRAVITY);

        program->acquireImageGL(outImage);

        if (!program->callFunction("clear_grids", GRID_SIZE.x * GRID_SIZE.y)) {
            exit(0);
        }

        if (!program->callFunction("update_grids", NUM_PARTICLES)) {
            exit(0);
        }

        if (!program->callFunction("update_particles", NUM_PARTICLES)) {
            exit(0);
        }

        if (!program->callFunction("render_main", WINDOW_WIDTH * WINDOW_HEIGHT)) {
            exit(0);
        }

        program->releaseImageGL(outImage);

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, outImage->glTex);

        glBegin(GL_QUADS);
            glTexCoord2f(0., 1.); glVertex2f(0., 0.);
            glTexCoord2f(0., 0.); glVertex2f(0., 1.);
            glTexCoord2f(1., 0.); glVertex2f(1., 1.);
            glTexCoord2f(1., 1.); glVertex2f(1., 0.);
        glEnd();

        glfwSwapBuffers(window);

        lastKeyDown = keyDown;

        glfwPollEvents();

        time += deltaTime;
    }

    delete gridBfr;
    delete particleBfr;
    delete outImage;
    delete program;
    delete clContext;

    glfwTerminate();
    return 0;
}