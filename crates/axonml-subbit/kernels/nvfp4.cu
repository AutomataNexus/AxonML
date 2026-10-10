// axonml-subbit — Blackwell NVFP4 training-forward kernels (sm_120a ONLY; loaded lazily).
// nvfp4_quant_f32: dense f32 [R,K] -> e2m1 nibbles (2/byte, low-first) + per-16 ue4m3 block
//   scales (1 byte/block, byte b of the u32 scale word = k-block 4j+b) — BYTE-IDENTICAL to
//   the reference CPU quantizer (same IEEE divisions, same nearest/tie-to-lower rounding).
// nvfp4_gemm_sf2: C[M,ldc] += (A*sfa)(B*sfb)^T on the block-scaled FP4 tensor cores,
//   mma.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X — a block-scaled FP4 tensor-core kernel
//   (uint4-grouped scale loads, ~100 TOPS real-scale vs cuBLAS-tf32 ~26) + ldc/c0 so a
//   reconstruct-tile GEMM lands at a column offset of the full out buffer. Requires
//   M%128==0, Ntile%64==0, K%256==0 per split-K slice; C pre-zeroed (atomicAdd epilogue).
// Scale-factor lane layout (m67_fp4sf probe): SFA lane 4q->row q, 4q+1->row q+8; SFB lane
//   4c->col c; byte b->k-block b. Scales u32[rows][K/64] load with no shuffling.
#include <cuda_pipeline.h>

__device__ __forceinline__ float ue4m3_dec(unsigned char b){
    int e=(b>>3)&0xF; float m=(float)(b&7);
    return e==0 ? m*(1.f/512.f) : (1.f+m*0.125f)*exp2f((float)(e-7));
}
__device__ __forceinline__ unsigned char ue4m3_enc(float v){
    if(!(v>0.f)) return 0;
    int E; float fr=frexpf(v,&E);
    int e=E-1, ef; float m_real;
    if(e<-6){ m_real=v*512.f; ef=0; }
    else { m_real=(2.f*fr-1.f)*8.f; ef=e+7; }
    if(ef>15) return 0x7F;
    int m=(int)ceilf(m_real-0.5f);
    if(m==8){ if(ef>=15) return 0x7F; ef+=1; m=0; }
    return (unsigned char)((ef<<3)|m);
}
__device__ __forceinline__ unsigned char e2m1_enc(float v){
    unsigned char s=(v<0.f)?8:0; float a=fabsf(v);
    int l=(a>0.25f)+(a>0.75f)+(a>1.25f)+(a>1.75f)+(a>2.5f)+(a>3.5f)+(a>5.f);
    return l==0?0:(unsigned char)(s|l);
}

extern "C" __global__ void nvfp4_quant_f32(const float* __restrict__ W,
    unsigned char* __restrict__ P, unsigned char* __restrict__ S, long long nblocks){
    long long b=blockIdx.x*(long long)blockDim.x+threadIdx.x;
    if(b>=nblocks) return;
    const float* blk=W+b*16;
    float amax=0.f;
    #pragma unroll
    for(int i=0;i<16;i++){ float a=fabsf(blk[i]); if(a>amax)amax=a; }
    unsigned char se=ue4m3_enc(amax/6.0f);
    float sd=ue4m3_dec(se);
    S[b]=se;
    unsigned long long pk=0;
    if(sd>0.f){
        #pragma unroll
        for(int i=0;i<16;i++) pk|=(unsigned long long)e2m1_enc(blk[i]/sd)<<(4*i);
    }
    ((unsigned long long*)P)[b]=pk;
}

#define BM 128
#define BN 64
#define BK 64
#define BKB 32
#define PF(buf,koff) \
    __pipeline_memcpy_async(&As[buf][tid*16], &A[(size_t)(bm+tid/2)*(K/2)+(koff)+(tid&1)*16], 16); \
    __pipeline_memcpy_async(&Bs[buf][tid*8],  &B[(size_t)(bn+tid/4)*(K/2)+(koff)+(tid&3)*8], 8);
extern "C" __global__ void nvfp4_gemm_sf2(const unsigned char* __restrict__ A,
    const unsigned char* __restrict__ B, const unsigned* __restrict__ SFA,
    const unsigned* __restrict__ SFB, float* __restrict__ C, int M, int N, int K,
    int ldc, int c0){
    __shared__ unsigned char As[2][BM*BKB];
    __shared__ unsigned char Bs[2][BN*BKB];
    int tid=threadIdx.x, warp=tid>>5, t=tid&31, group=t&3, tid4=t>>2, wm=warp>>1, wn=warp&1;
    int bm=blockIdx.y*BM, bn=blockIdx.x*BN;
    int kper=K/gridDim.z, k0s=(blockIdx.z*kper)/2, nstep=kper/BK;
    int kw=K/64, ks0=(blockIdx.z*kper)/64;
    int ra=bm+wm*32+tid4+((t&3)==1?8:0), cbg=bn+wn*32+tid4;
    float acc[2][4][4];
    #pragma unroll
    for(int i=0;i<2;i++)for(int j=0;j<4;j++)for(int e=0;e<4;e++) acc[i][j][e]=0.f;
    PF(0,k0s); __pipeline_commit();
    for(int s0=0;s0<nstep;s0+=4){
        uint4 sa4[2], sb4[4];
        sa4[0]=*(const uint4*)&SFA[(size_t)ra*kw+ks0+s0];
        sa4[1]=*(const uint4*)&SFA[(size_t)(ra+16)*kw+ks0+s0];
        #pragma unroll
        for(int nj=0;nj<4;nj++) sb4[nj]=*(const uint4*)&SFB[(size_t)(cbg+nj*8)*kw+ks0+s0];
        #pragma unroll
        for(int si=0;si<4;si++){
            int s=s0+si; if(s>=nstep) break;
            int cur=s&1;
            if(s+1<nstep){ PF((s+1)&1,k0s+(s+1)*BKB); __pipeline_commit(); __pipeline_wait_prior(1); }
            else __pipeline_wait_prior(0);
            __syncthreads();
            unsigned a[2][4], b[4][2];
            unsigned sa[2]={((const unsigned*)&sa4[0])[si],((const unsigned*)&sa4[1])[si]};
            unsigned sb[4]={((const unsigned*)&sb4[0])[si],((const unsigned*)&sb4[1])[si],
                            ((const unsigned*)&sb4[2])[si],((const unsigned*)&sb4[3])[si]};
            #pragma unroll
            for(int mi=0;mi<2;mi++){ int rb=wm*32+mi*16;
                a[mi][0]=*(unsigned*)&As[cur][(rb+tid4)*BKB+group*4];
                a[mi][1]=*(unsigned*)&As[cur][(rb+tid4+8)*BKB+group*4];
                a[mi][2]=*(unsigned*)&As[cur][(rb+tid4)*BKB+group*4+16];
                a[mi][3]=*(unsigned*)&As[cur][(rb+tid4+8)*BKB+group*4+16]; }
            #pragma unroll
            for(int nj=0;nj<4;nj++){ int cbs=wn*32+nj*8;
                b[nj][0]=*(unsigned*)&Bs[cur][(cbs+tid4)*BKB+group*4];
                b[nj][1]=*(unsigned*)&Bs[cur][(cbs+tid4)*BKB+group*4+16]; }
            #pragma unroll
            for(int mi=0;mi<2;mi++)for(int nj=0;nj<4;nj++)
                asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                  "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
                  :"+f"(acc[mi][nj][0]),"+f"(acc[mi][nj][1]),"+f"(acc[mi][nj][2]),"+f"(acc[mi][nj][3])
                  :"r"(a[mi][0]),"r"(a[mi][1]),"r"(a[mi][2]),"r"(a[mi][3]),"r"(b[nj][0]),"r"(b[nj][1]),
                   "r"(sa[mi]),"r"(sb[nj]));
            __syncthreads();
        }
    }
    #pragma unroll
    for(int mi=0;mi<2;mi++)for(int nj=0;nj<4;nj++){
        int rbb=bm+wm*32+mi*16, cbb=c0+bn+wn*32+nj*8;
        atomicAdd(&C[(size_t)(rbb+tid4)*ldc+cbb+group*2+0],acc[mi][nj][0]);
        atomicAdd(&C[(size_t)(rbb+tid4)*ldc+cbb+group*2+1],acc[mi][nj][1]);
        atomicAdd(&C[(size_t)(rbb+tid4+8)*ldc+cbb+group*2+0],acc[mi][nj][2]);
        atomicAdd(&C[(size_t)(rbb+tid4+8)*ldc+cbb+group*2+1],acc[mi][nj][3]);
    }
}
__device__ __forceinline__ unsigned rhash(unsigned long long x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return (unsigned)x;
}
__device__ __forceinline__ unsigned char e2m1_enc_sr(float v, unsigned r){
    const float G[8] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
    unsigned char s = (v<0.f)?8:0; float a = fabsf(v);
    if(a >= 6.f) return s|7;
    int lo = (a>0.5f)+(a>1.f)+(a>1.5f)+(a>2.f)+(a>3.f)+(a>4.f);
    float g0=G[lo], g1=G[lo+1];
    float p = (a-g0)/(g1-g0);
    int up = ((float)(r&0xFFFFFF) * (1.f/16777216.f)) < p;
    int l = lo+up;
    return l==0?0:(unsigned char)(s|l);
}
extern "C" __global__ void nvfp4_quant_sr_f32(const float* __restrict__ W,
    unsigned char* __restrict__ P, unsigned char* __restrict__ S, long long nblocks,
    unsigned long long seed){
    long long b=blockIdx.x*(long long)blockDim.x+threadIdx.x;
    if(b>=nblocks) return;
    const float* blk=W+b*16;
    float amax=0.f;
    #pragma unroll
    for(int i=0;i<16;i++){ float a=fabsf(blk[i]); if(a>amax)amax=a; }
    unsigned char se=ue4m3_enc(amax/6.0f);
    float sd=ue4m3_dec(se);
    S[b]=se;
    unsigned long long pk=0;
    if(sd>0.f){
        #pragma unroll
        for(int i=0;i<16;i++)
            pk|=(unsigned long long)e2m1_enc_sr(blk[i]/sd, rhash(seed^(unsigned long long)(b*16+i)))<<(4*i);
    }
    ((unsigned long long*)P)[b]=pk;
}
