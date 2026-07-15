// CPU baseline for the GPU comparison.
// SAME scene, SAME resolution, SAME sample count as the GPU version.
// Only the chip changes.
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <thread>
#include <atomic>
#include <chrono>
#include <iomanip>

thread_local std::mt19937 t_rng(std::random_device{}());
float randf(){
    static thread_local std::uniform_real_distribution<float> d(0.0f,1.0f);
    return d(t_rng);
}
float randf(float a, float b){ return a + (b-a)*randf(); }

struct V3 {
    float x,y,z;
    V3(){x=0;y=0;z=0;}
    V3(float a,float b,float c){x=a;y=b;z=c;}
    V3 operator+(const V3&v)const{return V3(x+v.x,y+v.y,z+v.z);}
    V3 operator-(const V3&v)const{return V3(x-v.x,y-v.y,z-v.z);}
    V3 operator*(float k)const{return V3(x*k,y*k,z*k);}
    V3 operator*(const V3&v)const{return V3(x*v.x,y*v.y,z*v.z);}
    V3 operator/(float k)const{return V3(x/k,y/k,z/k);}
    float lenSq()const{return x*x+y*y+z*z;}
    float len()const{return std::sqrt(lenSq());}
};
V3 operator*(float k, const V3&v){ return v*k; }
float dot(const V3&a,const V3&b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
V3 unit(const V3&v){ return v/v.len(); }
V3 reflect(const V3&v,const V3&n){ return v - 2*dot(v,n)*n; }
V3 refract(const V3&uv,const V3&n,float er){
    float ct = std::fmin(dot(uv*-1.0f,n),1.0f);
    V3 perp = er*(uv + ct*n);
    V3 par  = n * -std::sqrt(std::fabs(1.0f - perp.lenSq()));
    return perp + par;
}
float schlick(float c,float ri){ float r0=(1-ri)/(1+ri); r0=r0*r0; return r0+(1-r0)*std::pow(1-c,5); }
V3 randomUnit(){
    for(int i=0;i<20;i++){
        V3 p(randf(-1,1),randf(-1,1),randf(-1,1));
        float d = p.lenSq();
        if(d < 1.0f && d > 1e-6f) return unit(p);
    }
    return V3(0,1,0);
}

#define MAT_DIFFUSE 0
#define MAT_METAL   1
#define MAT_GLASS   2

struct Sphere {
    V3    center;
    float radius;
    V3    albedo;
    float fuzz;
    int   matType;
    float ir;
};

struct Hit {
    bool  hit;
    float t;
    V3    point, normal;
    bool  frontFace;
    V3    albedo;
    float fuzz;
    int   matType;
    float ir;
};

std::vector<Sphere> world;

bool hitSphere(const Sphere& s, const V3& o, const V3& d, float tMin, float tMax, Hit& rec){
    V3 oc = o - s.center;
    float a  = dot(d,d);
    float hb = dot(oc,d);
    float c  = dot(oc,oc) - s.radius*s.radius;
    float disc = hb*hb - a*c;
    if(disc < 0) return false;
    float sq = std::sqrt(disc);
    float root = (-hb - sq)/a;
    if(root < tMin || root > tMax){
        root = (-hb + sq)/a;
        if(root < tMin || root > tMax) return false;
    }
    rec.hit   = true;
    rec.t     = root;
    rec.point = o + d*root;
    V3 outward = (rec.point - s.center) / s.radius;
    rec.frontFace = dot(d, outward) < 0.0f;
    rec.normal    = rec.frontFace ? outward : outward*-1.0f;
    rec.albedo  = s.albedo;
    rec.fuzz    = s.fuzz;
    rec.matType = s.matType;
    rec.ir      = s.ir;
    return true;
}

// SAME iterative structure as the GPU, so the comparison is fair
V3 tracePath(V3 origin, V3 dir, int maxBounces){
    V3 throughput(1,1,1);
    V3 color(0,0,0);

    for(int bounce = 0; bounce < maxBounces; bounce++){
        Hit rec; rec.hit = false;
        float closest = 1e30f;

        for(const Sphere& s : world){
            Hit tmp; tmp.hit = false;
            if(hitSphere(s, origin, dir, 0.001f, closest, tmp)){
                closest = tmp.t;
                rec = tmp;
            }
        }

        if(!rec.hit){
            V3 ud = unit(dir);
            float t = 0.5f*(ud.y + 1.0f);
            V3 sky = V3(1,1,1)*(1.0f-t) + V3(0.5f,0.7f,1.0f)*t;
            color = color + throughput*sky;
            break;
        }

        V3 scatterDir, attenuation;

        if(rec.matType == MAT_DIFFUSE){
            scatterDir = rec.normal + randomUnit();
            if(scatterDir.len() < 1e-6f) scatterDir = rec.normal;
            attenuation = rec.albedo;
        }
        else if(rec.matType == MAT_METAL){
            V3 reflected = reflect(unit(dir), rec.normal);
            scatterDir = reflected + rec.fuzz*randomUnit();
            attenuation = rec.albedo;
            if(dot(scatterDir, rec.normal) <= 0.0f) break;
        }
        else { // GLASS
            attenuation = V3(1,1,1);
            float ratio = rec.frontFace ? (1.0f/rec.ir) : rec.ir;
            V3 ud = unit(dir);
            float cosTheta = std::fmin(dot(ud*-1.0f, rec.normal), 1.0f);
            float sinTheta = std::sqrt(1.0f - cosTheta*cosTheta);
            if(ratio*sinTheta > 1.0f || schlick(cosTheta, ratio) > randf())
                scatterDir = reflect(ud, rec.normal);
            else
                scatterDir = refract(ud, rec.normal, ratio);
        }

        throughput = throughput * attenuation;
        origin = rec.point;
        dir    = scatterDir;

        // Russian roulette (same as GPU)
        if(bounce > 3){
            float p = std::fmax(throughput.x, std::fmax(throughput.y, throughput.z));
            if(randf() > p) break;
            throughput = throughput / p;
        }
    }
    return color;
}

int main(){
    int  width      = 800;
    int  height     = 450;
    int  samples    = 200;
    int  maxBounces = 30;
    float aspect = float(width)/float(height);

    // ---- IDENTICAL SCENE to the GPU version ----
    world.push_back({ {0,-100.5f,-1}, 100.0f, {0.5f,0.5f,0.5f}, 0.0f, MAT_DIFFUSE, 1.0f });
    world.push_back({ {-1.05f,0,-1},  0.5f,   {1,1,1},          0.0f, MAT_GLASS,   1.5f });
    world.push_back({ {-1.05f,0,-1}, -0.45f,  {1,1,1},          0.0f, MAT_GLASS,   1.5f });
    world.push_back({ {0,0,-1},       0.5f,   {0.7f,0.25f,0.25f},0.0f,MAT_DIFFUSE, 1.0f });
    world.push_back({ {1.05f,0,-1},   0.5f,   {0.8f,0.7f,0.4f}, 0.05f,MAT_METAL,   1.0f });

    // ---- IDENTICAL CAMERA ----
    float vh = 2.0f;
    float vw = aspect * vh;
    float focal = 1.0f;
    V3 camPos(0.0f, 0.3f, 1.2f);
    V3 horizontal(vw, 0, 0);
    V3 vertical(0, vh, 0);
    V3 lowerLeft(camPos.x - vw*0.5f, camPos.y - vh*0.5f, camPos.z - focal);

    unsigned cores = std::thread::hardware_concurrency();
    if(cores == 0) cores = 4;

    std::cout << "CPU:     " << cores << " threads\n";
    std::cout << "Image:   " << width << "x" << height << "\n";
    std::cout << "Samples: " << samples << " spp, " << maxBounces << " bounces\n";
    std::cout << "Scene:   " << world.size() << " spheres\n\n";
    std::cout << "Rendering on CPU...\n";

    std::vector<V3> px(width*height);
    std::atomic<int> nextRow(0);

    auto t0 = std::chrono::high_resolution_clock::now();

    auto worker = [&](){
        while(true){
            int j = nextRow.fetch_add(1);
            if(j >= height) break;
            for(int i = 0; i < width; i++){
                V3 sum(0,0,0);
                for(int s = 0; s < samples; s++){
                    float u = (float(i) + randf()) / float(width - 1);
                    float v = (float(j) + randf()) / float(height - 1);
                    V3 dir = lowerLeft + u*horizontal + v*vertical - camPos;
                    sum = sum + tracePath(camPos, dir, maxBounces);
                }
                V3 c = sum / float(samples);
                px[j*width + i] = V3(std::sqrt(c.x), std::sqrt(c.y), std::sqrt(c.z));
            }
        }
    };

    std::vector<std::thread> threads;
    for(unsigned t = 0; t < cores; t++) threads.emplace_back(worker);
    for(auto& th : threads) th.join();

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CPU render: " << ms << " ms  (" << ms/1000.0 << " s)\n";

    std::ofstream out("cpu_path.ppm");
    out << "P3\n" << width << " " << height << "\n255\n";
    for(int j = height-1; j >= 0; j--)
        for(int i = 0; i < width; i++){
            V3 c = px[j*width + i];
            auto b = [](float v){ if(v<0)v=0; if(v>1)v=1; return int(255.99f*v); };
            out << b(c.x) << " " << b(c.y) << " " << b(c.z) << "\n";
        }
    std::cout << "Wrote cpu_path.ppm\n";
    return 0;
}