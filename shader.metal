#include <metal_stdlib>
using namespace metal;

#define MAT_DIFFUSE 0
#define MAT_METAL   1
#define MAT_GLASS   2

struct Sphere {
    float4 centerRadius;   // xyz = center, w = radius
    float4 albedoFuzz;     // xyz = albedo, w = fuzz
    float4 matIr;          // x = matType, y = index of refraction
};

// ---- THE FLATTENED BVH NODE ----
// Your CPU BVH held POINTERS to children. GPU buffers can't hold pointers,
// so the tree is flattened into an array and children become INDICES.
//   count == 0 -> interior: left child at (myIdx+1), right child at leftFirst
//   count  > 0 -> leaf:     spheres [leftFirst, leftFirst+count)
struct BVHNode {
    float4 boxMin;   // xyz = box min, w = leftFirst
    float4 boxMax;   // xyz = box max, w = count
};

struct Camera {
    float4 origin;
    float4 lowerLeft;
    float4 horizontal;
    float4 vertical;
};

struct Params {
    uint width;
    uint height;
    uint samples;
    uint maxBounces;
    uint numSpheres;
    uint useBVH;      // 1 = walk the tree, 0 = brute force
};

// ---- per-thread RNG ----
uint wangHash(uint seed) {
    seed = (seed ^ 61u) ^ (seed >> 16);
    seed *= 9u;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15);
    return seed;
}
float randFloat(thread uint& state) {
    state = state * 747796405u + 2891336453u;
    uint r = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    r = (r >> 22u) ^ r;
    return float(r) / 4294967295.0;
}
float3 randomUnitVector(thread uint& state) {
    for (int i = 0; i < 20; i++) {
        float3 p = float3(randFloat(state), randFloat(state), randFloat(state)) * 2.0 - 1.0;
        float d = dot(p, p);
        if (d < 1.0 && d > 1e-6) return normalize(p);
    }
    return float3(0, 1, 0);
}

struct Hit {
    bool   hit;
    float  t;
    float3 point;
    float3 normal;
    bool   frontFace;
    float3 albedo;
    float  fuzz;
    int    matType;
    float  ir;
};

bool hitSphere(Sphere s, float3 origin, float3 dir,
               float tMin, float tMax, thread Hit& rec)
{
    float3 center = s.centerRadius.xyz;
    float  radius = s.centerRadius.w;

    float3 oc = origin - center;
    float a  = dot(dir, dir);
    float hb = dot(oc, dir);
    float c  = dot(oc, oc) - radius * radius;
    float disc = hb*hb - a*c;
    if (disc < 0) return false;

    float sq = sqrt(disc);
    float root = (-hb - sq) / a;
    if (root < tMin || root > tMax) {
        root = (-hb + sq) / a;
        if (root < tMin || root > tMax) return false;
    }

    rec.hit   = true;
    rec.t     = root;
    rec.point = origin + dir * root;

    float3 outward = (rec.point - center) / radius;
    rec.frontFace = dot(dir, outward) < 0.0;
    rec.normal    = rec.frontFace ? outward : -outward;

    rec.albedo  = s.albedoFuzz.xyz;
    rec.fuzz    = s.albedoFuzz.w;
    rec.matType = int(s.matIr.x);
    rec.ir      = s.matIr.y;
    return true;
}

// slab test: does the ray pass through this box?
bool hitBox(float3 boxMin, float3 boxMax, float3 origin, float3 invDir,
            float tMin, float tMax)
{
    float3 t0 = (boxMin - origin) * invDir;
    float3 t1 = (boxMax - origin) * invDir;
    float3 tsmall = min(t0, t1);
    float3 tbig   = max(t0, t1);
    float tnear = max(max(tsmall.x, tsmall.y), max(tsmall.z, tMin));
    float tfar  = min(min(tbig.x,   tbig.y),   min(tbig.z,   tMax));
    return tnear <= tfar;
}

// ---- BVH TRAVERSAL WITHOUT RECURSION ----
// The CPU version called itself. GPUs have no real call stack, so we keep
// our OWN stack: a fixed array we push and pop by hand.
//
// This is also where WARP DIVERGENCE bites. GPU threads run in lockstep groups,
// but neighboring rays take DIFFERENT paths down the tree, so the hardware runs
// both branches and masks off the wrong results. The structure that made the CPU
// 15x faster partially fights the GPU's execution model. That's exactly why
// NVIDIA built dedicated RT cores.
bool hitBVH(device const BVHNode* nodes,
            device const Sphere*  spheres,
            float3 origin, float3 dir,
            float tMin, float tMax,
            thread Hit& best)
{
    float3 invDir = 1.0 / dir;

    uint stack[64];
    int  sp = 0;
    stack[sp++] = 0;     // push the root

    bool hitAny = false;
    float closest = tMax;

    while (sp > 0) {
        uint ni = stack[--sp];               // pop
        BVHNode node = nodes[ni];

        // miss the box -> skip this entire branch
        if (!hitBox(node.boxMin.xyz, node.boxMax.xyz, origin, invDir, tMin, closest))
            continue;

        int count     = int(node.boxMax.w);
        int leftFirst = int(node.boxMin.w);

        if (count > 0) {
            // LEAF: test the spheres here
            for (int i = 0; i < count; i++) {
                Hit tmp; tmp.hit = false;
                if (hitSphere(spheres[leftFirst + i], origin, dir, tMin, closest, tmp)) {
                    closest = tmp.t;
                    best    = tmp;
                    hitAny  = true;
                }
            }
        } else {
            // INTERIOR: push both children. Left is always at ni+1.
            if (sp < 62) {
                stack[sp++] = ni + 1;
                stack[sp++] = uint(leftFirst);
            }
        }
    }
    return hitAny;
}

// brute force, for the A/B comparison
bool hitBrute(device const Sphere* spheres, uint numSpheres,
              float3 origin, float3 dir, float tMin, float tMax,
              thread Hit& best)
{
    bool hitAny = false;
    float closest = tMax;
    for (uint i = 0; i < numSpheres; i++) {
        Hit tmp; tmp.hit = false;
        if (hitSphere(spheres[i], origin, dir, tMin, closest, tmp)) {
            closest = tmp.t;
            best    = tmp;
            hitAny  = true;
        }
    }
    return hitAny;
}

float reflectance(float cosine, float refIdx) {
    float r0 = (1.0 - refIdx) / (1.0 + refIdx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

float3 tracePath(float3 origin, float3 dir,
                 device const Sphere* spheres,
                 device const BVHNode* nodes,
                 constant Params& p,
                 thread uint& rngState)
{
    float3 throughput = float3(1.0, 1.0, 1.0);
    float3 color      = float3(0.0, 0.0, 0.0);

    for (uint bounce = 0; bounce < p.maxBounces; bounce++) {

        Hit rec; rec.hit = false;
        bool hit;

        if (p.useBVH == 1)
            hit = hitBVH(nodes, spheres, origin, dir, 0.001, 1e30, rec);
        else
            hit = hitBrute(spheres, p.numSpheres, origin, dir, 0.001, 1e30, rec);

        if (!hit) {
            float3 ud = normalize(dir);
            float t = 0.5 * (ud.y + 1.0);
            float3 sky = (1.0 - t) * float3(1.0, 1.0, 1.0) + t * float3(0.5, 0.7, 1.0);
            color += throughput * sky;
            break;
        }

        float3 scatterDir;
        float3 attenuation;

        if (rec.matType == MAT_DIFFUSE) {
            scatterDir = rec.normal + randomUnitVector(rngState);
            if (length(scatterDir) < 1e-6) scatterDir = rec.normal;
            attenuation = rec.albedo;
        }
        else if (rec.matType == MAT_METAL) {
            float3 reflected = reflect(normalize(dir), rec.normal);
            scatterDir = reflected + rec.fuzz * randomUnitVector(rngState);
            attenuation = rec.albedo;
            if (dot(scatterDir, rec.normal) <= 0.0) break;
        }
        else {
            attenuation = float3(1.0, 1.0, 1.0);
            float ratio = rec.frontFace ? (1.0 / rec.ir) : rec.ir;
            float3 ud = normalize(dir);
            float cosTheta = min(dot(-ud, rec.normal), 1.0);
            float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
            if (ratio * sinTheta > 1.0 || reflectance(cosTheta, ratio) > randFloat(rngState))
                scatterDir = reflect(ud, rec.normal);
            else
                scatterDir = refract(ud, rec.normal, ratio);
        }

        throughput *= attenuation;
        origin = rec.point;
        dir    = scatterDir;

        if (bounce > 3) {
            float q = max(throughput.x, max(throughput.y, throughput.z));
            if (randFloat(rngState) > q) break;
            throughput /= q;
        }
    }
    return color;
}

kernel void render(device float3*         output   [[buffer(0)]],
                   constant Params&       p        [[buffer(1)]],
                   constant Camera&       cam      [[buffer(2)]],
                   device const Sphere*   spheres  [[buffer(3)]],
                   device const BVHNode*  nodes    [[buffer(4)]],
                   uint2                  gid      [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;

    uint rngState = wangHash(gid.y * p.width + gid.x + 1u);
    float3 sum = float3(0.0, 0.0, 0.0);

    for (uint s = 0; s < p.samples; s++) {
        float u = (float(gid.x) + randFloat(rngState)) / float(p.width  - 1);
        float v = (float(gid.y) + randFloat(rngState)) / float(p.height - 1);

        float3 origin = cam.origin.xyz;
        float3 dir    = cam.lowerLeft.xyz
                      + u * cam.horizontal.xyz
                      + v * cam.vertical.xyz
                      - cam.origin.xyz;

        sum += tracePath(origin, dir, spheres, nodes, p, rngState);
    }

    float3 color = sum / float(p.samples);
    color = sqrt(color);
    output[gid.y * p.width + gid.x] = color;
}