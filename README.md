# GPU Ray Tracer

I built a ray tracer on the CPU first (following Peter Shirley's Realistic Ray Tracing book), then wanted to see if I could get it running on the GPU using Metal. This is that.

![render](bvh_render.png)

I honestly thought porting it would be mostly copy-paste. It wasn't. A bunch of stuff I did on the CPU just doesn't work on a GPU, and figuring out why was the whole point.

## Things I had to change

**Recursion doesn't work.** My CPU version had a function that called itself every time a ray bounced. GPUs can't really do that, so I had to rewrite it as a loop that keeps track of the color as it goes instead of relying on the function stack.

**No classes or pointers.** On the CPU my scene was a bunch of objects with inheritance. The GPU only wants plain data, so everything became flat structs, and my BVH tree had to be flattened into an array where each node stores index numbers instead of pointers.

**Random numbers are different.** No std::mt19937. Each GPU thread makes its own random numbers from its pixel position, otherwise every pixel bounces the exact same way and the image looks broken.

## What I measured

All on an Apple M2 Pro, same scene both ways.

GPU vs my CPU version (12 threads):
- CPU: 652 ms
- GPU: 63 ms
- about 10x faster

BVH vs brute force on the GPU:
- Brute force: 2174 ms
- BVH: 302 ms
- about 7x faster

## The thing I thought was cool

My BVH gave about 15x speedup on the CPU but only 7x on the GPU. Same code basically. I read up on why, and it's because of something called warp divergence: GPU threads run in groups that all do the same thing at once, but rays going through the BVH take different paths, so the GPU ends up doing extra work. Apparently this is exactly why newer GPUs have special ray tracing hardware built in. Learning that was probably my favorite part.

## A bug that took me a while

My first GPU render was just a blank sky, no error message at all. Turns out a float3 is 16 bytes on the GPU but 12 on the CPU, so my structs didn't line up and the GPU was reading garbage for where my spheres were. Switching to float4 fixed it. Wild that it didn't crash, it just drew the wrong thing.

## Running it

macOS with the Metal toolchain. You need Apple's metal-cpp in the folder, then:

    ./build.sh && ./raytracer

cpu_bench.cpp is the CPU version I compared against.