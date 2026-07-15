// Builds a BVH on the CPU, FLATTENS it into an array (no pointers -- GPU
// buffers can't hold them), uploads it, and renders the same scene twice:
// once brute-force, once with the BVH. Prints the speedup.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "metal-cpp/Foundation/Foundation.hpp"
#include "metal-cpp/Metal/Metal.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <simd/simd.h>

#define MAT_DIFFUSE 0
#define MAT_METAL   1
#define MAT_GLASS   2

struct Sphere {
    simd::float4 centerRadius;
    simd::float4 albedoFuzz;
    simd::float4 matIr;
};

// must match shader.metal exactly
struct BVHNode {
    simd::float4 boxMin;   // xyz = min, w = leftFirst
    simd::float4 boxMax;   // xyz = max, w = count (0 = interior node)
};

struct Camera {
    simd::float4 origin;
    simd::float4 lowerLeft;
    simd::float4 horizontal;
    simd::float4 vertical;
};

struct Params {
    uint width;
    uint height;
    uint samples;
    uint maxBounces;
    uint numSpheres;
    uint useBVH;
};

// ---------------------------------------------------------------------------
// BVH BUILDER
// Builds the tree, but writes it into a FLAT ARRAY instead of using pointers.
//   interior node: count = 0, left child at (myIdx+1), right child at leftFirst
//   leaf node:     count > 0, spheres [leftFirst, leftFirst+count)
// ---------------------------------------------------------------------------
std::vector<Sphere>  g_spheres;
std::vector<BVHNode> g_nodes;

void sphereBounds(const Sphere& s, simd::float3& lo, simd::float3& hi) {
    simd::float3 c = { s.centerRadius.x, s.centerRadius.y, s.centerRadius.z };
    float r = std::fabs(s.centerRadius.w);   // fabs: hollow spheres have negative radius
    lo = { c.x - r, c.y - r, c.z - r };
    hi = { c.x + r, c.y + r, c.z + r };
}

int buildBVH(int start, int end) {
    int myIdx = (int)g_nodes.size();
    g_nodes.push_back(BVHNode{});

    // bounding box over this range
    simd::float3 lo = {  1e30f,  1e30f,  1e30f };
    simd::float3 hi = { -1e30f, -1e30f, -1e30f };
    for (int i = start; i < end; i++) {
        simd::float3 slo, shi;
        sphereBounds(g_spheres[i], slo, shi);
        lo = simd::min(lo, slo);
        hi = simd::max(hi, shi);
    }

    int n = end - start;

    if (n <= 2) {
        // LEAF
        g_nodes[myIdx].boxMin = { lo.x, lo.y, lo.z, (float)start };
        g_nodes[myIdx].boxMax = { hi.x, hi.y, hi.z, (float)n };
        return myIdx;
    }

    // split along the longest axis, at the median
    simd::float3 extent = hi - lo;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > ((axis == 0) ? extent.x : extent.y)) axis = 2;

    std::sort(g_spheres.begin() + start, g_spheres.begin() + end,
        [axis](const Sphere& a, const Sphere& b) {
            float ca = (axis == 0) ? a.centerRadius.x : (axis == 1) ? a.centerRadius.y : a.centerRadius.z;
            float cb = (axis == 0) ? b.centerRadius.x : (axis == 1) ? b.centerRadius.y : b.centerRadius.z;
            return ca < cb;
        });

    int mid = start + n / 2;

    int leftIdx  = buildBVH(start, mid);   // always myIdx + 1
    int rightIdx = buildBVH(mid, end);
    (void)leftIdx;

    // INTERIOR: count = 0, and we store the RIGHT child's index
    g_nodes[myIdx].boxMin = { lo.x, lo.y, lo.z, (float)rightIdx };
    g_nodes[myIdx].boxMax = { hi.x, hi.y, hi.z, 0.0f };
    return myIdx;
}

// ---------------------------------------------------------------------------
double runRender(MTL::Device* device,
                 MTL::ComputePipelineState* pipeline,
                 MTL::CommandQueue* queue,
                 MTL::Buffer* pixelBuf,
                 MTL::Buffer* paramBuf,
                 MTL::Buffer* camBuf,
                 MTL::Buffer* sphereBuf,
                 MTL::Buffer* nodeBuf,
                 int width, int height)
{
    MTL::CommandBuffer* cmd = queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();

    enc->setComputePipelineState(pipeline);
    enc->setBuffer(pixelBuf,  0, 0);
    enc->setBuffer(paramBuf,  0, 1);
    enc->setBuffer(camBuf,    0, 2);
    enc->setBuffer(sphereBuf, 0, 3);
    enc->setBuffer(nodeBuf,   0, 4);

    enc->dispatchThreads(MTL::Size(width, height, 1), MTL::Size(8, 8, 1));
    enc->endEncoding();

    auto t0 = std::chrono::high_resolution_clock::now();
    cmd->commit();
    cmd->waitUntilCompleted();
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void savePPM(const char* name, MTL::Buffer* pixelBuf, int width, int height) {
    simd::float3* px = (simd::float3*)pixelBuf->contents();
    std::ofstream out(name);
    out << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height - 1; j >= 0; j--)
        for (int i = 0; i < width; i++) {
            simd::float3 c = px[j * width + i];
            auto b = [](float v){ if(v<0)v=0; if(v>1)v=1; return int(255.99f*v); };
            out << b(c.x) << " " << b(c.y) << " " << b(c.z) << "\n";
        }
}

int main() {
    int  width      = 800;
    int  height     = 450;
    uint samples    = 100;
    uint maxBounces = 25;
    float aspect = float(width) / float(height);

    // ---- BIG SCENE. A BVH only pays off when there are lots of objects. ----
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    auto rnd = [&](float a, float b){ return a + (b - a) * U(rng); };

    g_spheres.push_back({ {0.0f, -1000.0f, 0.0f, 1000.0f},
                          {0.5f, 0.5f, 0.5f, 0.0f},
                          {(float)MAT_DIFFUSE, 1.0f, 0, 0} });

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float cx = a + 0.9f * rnd(0, 1);
            float cz = b + 0.9f * rnd(0, 1);
            float pick = rnd(0, 1);
            if (pick < 0.75f) {
                g_spheres.push_back({ {cx, 0.2f, cz, 0.2f},
                                      {rnd(0,1)*rnd(0,1), rnd(0,1)*rnd(0,1), rnd(0,1)*rnd(0,1), 0.0f},
                                      {(float)MAT_DIFFUSE, 1.0f, 0, 0} });
            } else if (pick < 0.92f) {
                g_spheres.push_back({ {cx, 0.2f, cz, 0.2f},
                                      {rnd(0.5f,1), rnd(0.5f,1), rnd(0.5f,1), rnd(0.0f,0.4f)},
                                      {(float)MAT_METAL, 1.0f, 0, 0} });
            } else {
                g_spheres.push_back({ {cx, 0.2f, cz, 0.2f},
                                      {1, 1, 1, 0},
                                      {(float)MAT_GLASS, 1.5f, 0, 0} });
            }
        }
    }
    // three big hero spheres
    g_spheres.push_back({ {0, 1, 0, 1.0f},  {1,1,1,0},                {(float)MAT_GLASS,   1.5f, 0, 0} });
    g_spheres.push_back({ {-4, 1, 0, 1.0f}, {0.4f, 0.2f, 0.1f, 0.0f}, {(float)MAT_DIFFUSE, 1.0f, 0, 0} });
    g_spheres.push_back({ {4, 1, 0, 1.0f},  {0.7f, 0.6f, 0.5f, 0.0f}, {(float)MAT_METAL,   1.0f, 0, 0} });

    uint numSpheres = (uint)g_spheres.size();

    // ---- BUILD THE BVH ----
    auto bt0 = std::chrono::high_resolution_clock::now();
    buildBVH(0, (int)g_spheres.size());
    auto bt1 = std::chrono::high_resolution_clock::now();
    double buildMs = std::chrono::duration<double, std::milli>(bt1 - bt0).count();

    // ---- CAMERA ----
    float vh = 2.0f;
    float vw = aspect * vh;
    simd::float3 camPos = { 13.0f, 2.0f, 3.0f };
    simd::float3 lookAt = { 0.0f, 0.0f, 0.0f };

    simd::float3 w = simd::normalize(camPos - lookAt);
    simd::float3 up = { 0, 1, 0 };
    simd::float3 u = simd::normalize(simd::cross(up, w));
    simd::float3 v = simd::cross(w, u);

    float theta = 25.0f * 3.14159265f / 180.0f;
    float halfH = std::tan(theta / 2.0f);
    vh = 2.0f * halfH;
    vw = aspect * vh;

    simd::float3 horiz = u * vw;
    simd::float3 vert  = v * vh;
    simd::float3 ll    = camPos - horiz * 0.5f - vert * 0.5f - w;

    Camera cam;
    cam.origin     = { camPos.x, camPos.y, camPos.z, 0 };
    cam.horizontal = { horiz.x, horiz.y, horiz.z, 0 };
    cam.vertical   = { vert.x,  vert.y,  vert.z,  0 };
    cam.lowerLeft  = { ll.x,    ll.y,    ll.z,    0 };

    // ---- GPU SETUP ----
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) { std::cerr << "No GPU\n"; return 1; }

    std::cout << "GPU:        " << device->name()->utf8String() << "\n";
    std::cout << "Image:      " << width << "x" << height << "\n";
    std::cout << "Samples:    " << samples << " spp, " << maxBounces << " bounces\n";
    std::cout << "Scene:      " << numSpheres << " spheres\n";
    std::cout << "BVH nodes:  " << g_nodes.size() << "\n";
    std::cout << "BVH build:  " << std::fixed << std::setprecision(2) << buildMs << " ms (on CPU, once)\n\n";

    NS::Error* err = nullptr;
    NS::String* libPath = NS::String::string("shader.metallib", NS::UTF8StringEncoding);
    MTL::Library* library = device->newLibrary(libPath, &err);
    if (!library) { std::cerr << "Couldn't load shader.metallib\n"; return 1; }

    NS::String* fnName = NS::String::string("render", NS::UTF8StringEncoding);
    MTL::Function* fn = library->newFunction(fnName);
    MTL::ComputePipelineState* pipeline = device->newComputePipelineState(fn, &err);
    if (!pipeline) { std::cerr << "Pipeline failed\n"; return 1; }

    size_t pixelCount = (size_t)width * height;
    MTL::Buffer* pixelBuf  = device->newBuffer(pixelCount * sizeof(simd::float3), MTL::ResourceStorageModeShared);
    MTL::Buffer* camBuf    = device->newBuffer(&cam, sizeof(cam), MTL::ResourceStorageModeShared);
    MTL::Buffer* sphereBuf = device->newBuffer(g_spheres.data(),
                                               g_spheres.size() * sizeof(Sphere),
                                               MTL::ResourceStorageModeShared);
    MTL::Buffer* nodeBuf   = device->newBuffer(g_nodes.data(),
                                               g_nodes.size() * sizeof(BVHNode),
                                               MTL::ResourceStorageModeShared);

    MTL::CommandQueue* queue = device->newCommandQueue();

    Params params;
    params.width      = (uint)width;
    params.height     = (uint)height;
    params.samples    = samples;
    params.maxBounces = maxBounces;
    params.numSpheres = numSpheres;

    MTL::Buffer* paramBuf = device->newBuffer(sizeof(Params), MTL::ResourceStorageModeShared);

    // ---------- PASS 1: BRUTE FORCE ----------
    params.useBVH = 0;
    memcpy(paramBuf->contents(), &params, sizeof(params));
    std::cout << "Rendering BRUTE FORCE (tests every sphere)...\n";
    double msBrute = runRender(device, pipeline, queue, pixelBuf, paramBuf,
                               camBuf, sphereBuf, nodeBuf, width, height);
    std::cout << "  " << msBrute << " ms\n";
    savePPM("gpu_brute.ppm", pixelBuf, width, height);

    // ---------- PASS 2: BVH ----------
    params.useBVH = 1;
    memcpy(paramBuf->contents(), &params, sizeof(params));
    std::cout << "Rendering with BVH (walks the tree)...\n";
    double msBVH = runRender(device, pipeline, queue, pixelBuf, paramBuf,
                             camBuf, sphereBuf, nodeBuf, width, height);
    std::cout << "  " << msBVH << " ms\n";
    savePPM("gpu_bvh.ppm", pixelBuf, width, height);

    std::cout << "\n=========== RESULTS ===========\n";
    std::cout << "Brute force: " << msBrute << " ms\n";
    std::cout << "BVH:         " << msBVH   << " ms\n";
    std::cout << "Speedup:     " << (msBrute / msBVH) << "x\n";
    std::cout << "\nWrote gpu_brute.ppm and gpu_bvh.ppm (they should look identical)\n";

    return 0;
}