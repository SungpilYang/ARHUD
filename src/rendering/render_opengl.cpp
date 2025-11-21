#define GLM_ENABLE_EXPERIMENTAL

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <locale>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#pragma comment (lib, "glfw3")
#pragma comment (lib, "OpenGL32")
#pragma comment (lib, "GlU32")

#ifdef _DEBUG
#pragma comment(lib,"opencv_world4100d")

#else
#pragma comment(lib,"opencv_world4100")
#endif

using namespace std;
using namespace cv;
using namespace Eigen;

// ============================= Struct =============================
struct Pose {
    glm::mat3 R;
    glm::vec3 t;
    glm::quat q;
    double time;
};

struct TimedPoint {
    double time;
    glm::vec3 pos;
};

// ============================= Shader =============================
const char* bgVS = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTex;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTex;
}
)";

const char* bgFS = R"(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D frameTex;
void main() {
    FragColor = texture(frameTex, TexCoord);
}
)";

const char* hudVS = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 MVP;
void main() {
    gl_Position = MVP * vec4(aPos, 1.0);
}
)";

const char* hudFS = R"(
#version 330 core
out vec4 FragColor;
uniform vec4 color;
void main() {
    FragColor = color;
}
)";

unsigned int compileShader(unsigned int type, const char* src) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    int success; glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, NULL, log);
        cerr << "X Shader compile error:\n" << log << endl;
    }
    return shader;
}

unsigned int createProgram(const char* vs, const char* fs) {
    unsigned int v = compileShader(GL_VERTEX_SHADER, vs);
    unsigned int f = compileShader(GL_FRAGMENT_SHADER, fs);
    unsigned int prog = glCreateProgram();
    glAttachShader(prog, v);
    glAttachShader(prog, f);
    glLinkProgram(prog);
    glDeleteShader(v);
    glDeleteShader(f);
    return prog;
}

// ============================= Loader =============================
vector<Pose> loadPoses(const string& path) {
    vector<Pose> poses;
    ifstream f(path);
    string line;
    while (getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        stringstream ss(line);
        double t, tx, ty, tz, qx, qy, qz, qw;
        if (!(ss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) continue;
        glm::quat q((float)qw, (float)qx, (float)qy, (float)qz);
        glm::mat3 R = glm::mat3_cast(q);
        Pose p{ R, glm::vec3(tx, ty, tz), q, t };
        poses.push_back(p);
    }
    cout << "> Loaded " << poses.size() << " poses\n";
    return poses;
}

vector<TimedPoint> loadTimedPoints(const string& path) {
    ifstream f(path);
    vector<TimedPoint> pts;
    double t, x, y, z;
    while (f >> t >> x >> y >> z)
        pts.push_back({ t, glm::vec3(x, y, z) });
    cout << "> Loaded " << pts.size() << " timestamped 3D points\n";
    return pts;
}

Pose interpolatePose(const vector<Pose>& poses, double t) {
    if (poses.empty()) return {};
    if (t <= poses.front().time) return poses.front();
    if (t >= poses.back().time) return poses.back();
    for (size_t i = 0; i < poses.size() - 1; i++) {
        const Pose& a = poses[i], & b = poses[i + 1];
        if (a.time <= t && t <= b.time) {
            float r = float((t - a.time) / (b.time - a.time));
            glm::quat q = glm::slerp(a.q, b.q, r);
            glm::vec3 tt = (1 - r) * a.t + r * b.t;
            glm::mat3 R = glm::mat3_cast(q);
            return { R, tt, q, t };
        }
    }
    return poses.back();
}

glm::mat4 makeProjectionMatrix(float fx, float fy, float cx, float cy,
    float w, float h, float near_z, float far_z) {
    glm::mat4 P(0.0f);
    P[0][0] = 2.0f * fx / w;
    P[0][2] = 1.0f - 2.0f * cx / w;
    P[1][1] = -2.0f * fy / h;
    P[1][2] = -1.0f + 2.0f * cy / h;
    float inv_depth = 1.0f / (near_z - far_z);
    P[2][2] = (far_z + near_z) * inv_depth;
    P[2][3] = -1.0f;
    P[3][2] = 2.0f * far_z * near_z * inv_depth;
    return P;
}

int main() {
    // ===== 비디오 =====
    VideoCapture cap("test1.MOV");
    if (!cap.isOpened()) {
        cerr << "> Failed to open video.\n";
        return -1;
    }
    int W = cap.get(CAP_PROP_FRAME_WIDTH);
    int H = cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(W, H, "ARHUD Renderer (timestamped)", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    unsigned int tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    float quad[] = { -1,-1,0,1, 1,-1,1,1, -1,1,0,0, 1,1,1,0 };
    unsigned int bgVAO, bgVBO;
    glGenVertexArrays(1, &bgVAO);
    glGenBuffers(1, &bgVBO);
    glBindVertexArray(bgVAO);
    glBindBuffer(GL_ARRAY_BUFFER, bgVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // ===== 데이터 로드 =====`
    auto poses = loadPoses("KeyFrameTrajectory.txt");
    auto points = loadTimedPoints("points_3d_full_timestamped.txt");
    if (poses.empty() || points.empty()) return -1;

    unsigned int bgShader = createProgram(bgVS, bgFS);
    unsigned int hudShader = createProgram(hudVS, hudFS);
    int mvpLoc = glGetUniformLocation(hudShader, "MVP");
    int colorLoc = glGetUniformLocation(hudShader, "color");

    unsigned int ribbonVAO, ribbonVBO;
    glGenVertexArrays(1, &ribbonVAO);
    glGenBuffers(1, &ribbonVBO);
    glBindVertexArray(ribbonVAO);
    glBindBuffer(GL_ARRAY_BUFFER, ribbonVBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    // ===== Projection =====
    glm::mat4 proj = makeProjectionMatrix(
        1464.357f, 1463.089f,
        961.935f, 521.971f,
        (float)W, (float)H,
        0.01f, 10000.0f);

    // ===== HUD Offset / Scale =====
   glm::vec3 hud_offset(0.0f, 1.3f, -1.5f);

    float scale = 0.7f;
    float half_w = 0.2f;


    // ===== 루프 =====
    Mat frame;
    size_t frameIdx = 0;
    double startTime = poses.front().time;
    double offset_sec = 0.0;

    while (!glfwWindowShouldClose(window)) {
        cap >> frame;
        if (frame.empty()) break;
        cvtColor(frame, frame, COLOR_BGR2RGB);

        double frameTime = startTime + (frameIdx / fps) + offset_sec;
        Pose interp = interpolatePose(poses, frameTime);
    
  
        glm::mat3 Rcw = glm::transpose(interp.R);
        glm::vec3 tcw = -Rcw * interp.t;
        glm::mat4 view = glm::mat4(1.0f);  
       
        view = glm::translate(view, glm::vec3(-interp.t.x, -interp.t.y, -interp.t.z));
        view *= glm::mat4_cast(interp.q);
        glm::mat4 camOffset = glm::translate(glm::mat4(1.0f), hud_offset);
        view = view * camOffset;

         // ===== Pitch 보정 =====
        float pitch_correction_deg = -30.0f;  
        float pitch_rad = glm::radians(pitch_correction_deg);
        glm::mat4 pitchAdjust = glm::rotate(glm::mat4(1.0f), pitch_rad, glm::vec3(1, 0, 0));
        view = view * pitchAdjust;


        glm::mat4 MVP = proj * view;

        // ===== timestamp 기반 포인트 필터링 =====
        vector<glm::vec3> active;
        for (auto& tp : points) {
            if (fabs(tp.time - frameTime) < 0.015) //  타임 윈도우 축소 (겹침 방지)
                active.push_back(tp.pos * 0.1f);
        }
        if (active.size() < 2) {
            glfwSwapBuffers(window);
            glfwPollEvents();
            frameIdx++;
            continue;
        }

        // ===== 리본 생성 =====
        vector<glm::vec3> ribbon;
        for (size_t i = 0; i + 1 < active.size(); ++i) {
            glm::vec3 c = active[i];
            glm::vec3 n = active[i + 1];
            glm::vec3 dir = glm::normalize(n - c);
            glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0, 1, 0), dir));
            ribbon.push_back(c - half_w * right * scale);
            ribbon.push_back(c + half_w * right * scale);
        }

        // ===== GPU 업데이트 =====
        glBindBuffer(GL_ARRAY_BUFFER, ribbonVBO);
        glBufferData(GL_ARRAY_BUFFER, ribbon.size() * sizeof(glm::vec3), ribbon.data(), GL_DYNAMIC_DRAW);

        glClear(GL_COLOR_BUFFER_BIT);

        //  비디오 (배경)
        glUseProgram(bgShader);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, frame.data);
        glBindVertexArray(bgVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        //  리본
        glUseProgram(hudShader);
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(MVP));
        glUniform4f(colorLoc, 0.0f, 1.0f, 0.0f, 0.6f);
        glBindVertexArray(ribbonVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLsizei)ribbon.size());

        glfwSwapBuffers(window);
        glfwPollEvents();
        frameIdx++;
    }

    glfwTerminate();
    return 0;
}