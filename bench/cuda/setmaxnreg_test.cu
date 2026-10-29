// Does ptxas accept setmaxnreg for this target?
__global__ void __launch_bounds__(256, 1) k(int* o) {
  if (threadIdx.x < 128) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
  else asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  o[threadIdx.x] = threadIdx.x;
}
