#include "histogram.hpp"
#include <vector>
#include <chrono>
#include <limits>
#include <cmath>
#include <new>
#include <omp.h>
#include <sched.h>
#ifdef LI_USE_BLAS
#include <cblas.h>
#endif
// Compile-time choice of the accelerated BF16 score implementation.
// Default: SVE2 svbfdot_f32. Build with scripts/build_cpu.sh --sme to select SME.
#ifdef LI_SCORE_USE_SME
#if defined(__ARM_FEATURE_SME)
#include <arm_sme.h>
#define HAVE_SME 1
#else
#error "LI_SCORE_USE_SME requires an SME-capable compiler and -march=...+sme"
#endif
#else
#define HAVE_SME 0
#endif
#if HAVE_SME
// Locally streaming: the kernel can be called from ordinary OpenMP workers and
// the compiler emits the required SMSTART/SMSTOP for the ZA tile.
#if defined(__clang__)
// Clang 16+ implements the ACLE SME keywords, not the attribute spellings.
#define LI_ARM_STREAMING __arm_locally_streaming
#define LI_ARM_NEW_ZA __arm_new("za")
#else
// GCC 14+ implements the SME attributes; GCC <= 13 has no usable SME support.
#define LI_ARM_STREAMING __attribute__((arm_locally_streaming))
#define LI_ARM_NEW_ZA __attribute__((arm_new("za")))
#endif
#endif

namespace {
float decode(uint16_t x,int bf) {
    uint32_t u;
    if(bf) u=uint32_t(x)<<16;
    else {
        uint32_t sign=uint32_t(x&0x8000)<<16,e=(x>>10)&31,m=x&1023;
        if(!e) {
            if(!m) u=sign;
            else {int z=0;while(!(m&1024)){m<<=1;++z;}u=sign|(uint32_t(113-z)<<23)|((m&1023)<<13);}
        } else if(e==31) u=sign|0x7f800000|(m<<13);
        else u=sign|((e+112)<<23)|(m<<13);
    }
    float f;std::memcpy(&f,&u,4);return f;
}
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point a,Clock::time_point b) {return std::chrono::duration<double,std::milli>(b-a).count();}
struct Row {int b,i,v;size_t index;};
struct Task {int b,q0,q1,start,end;};
struct alignas(64) Worker {
    uint32_t hist[256]{};
    size_t gt=0,eq=0,offg=0,offe=0;
    std::vector<uint32_t> keys;
    std::vector<uint8_t> bytes;
    std::vector<float> kv;
#if HAVE_SVE2
    // backend 2 (svebf16): interleaved q tile and key-duplicated tile (bf16 raw)
    std::vector<uint16_t> qbf, kdup;
#endif
#if HAVE_SME
    // backend 2 (sme): FP32 transposed query block and key block for SME MOPA
    std::vector<float> qT, kblk;
#endif
#ifdef LI_USE_BLAS
    std::vector<float> packed_key,dots;
#endif
    int cpu=-1,vl=0;
};
struct alignas(64) Group {uint32_t prefix=0;int remain=0,target=0,need=0;size_t ng=0;};
struct Workspace {
    int B,S1,S2,H,D,K,P,tile,qtile;
    int score_backend=0;
    std::vector<float> Q,W;
    std::vector<Row> rows;
    std::vector<Task> tasks;
    std::vector<Worker> workers;
    std::vector<Group> groups;
    int actual_threads=0,actual_groups=0;
    Workspace(int b,int s,int t,int h,int d,int k,int p,int tile_,int qt):
        B(b),S1(s),S2(t),H(h),D(d),K(k),P(p),tile(tile_),qtile(qt),
        Q(size_t(b)*s*h*d),W(size_t(b)*s*h),workers(p),groups(p) {
        rows.reserve(size_t(B)*S1);
        tasks.reserve(size_t(B)*((S1+qtile-1)/qtile)*((S2+tile-1)/tile));
        // A thread can own a whole row if enough rows exist. Allocate before parallel regions.
        for(auto& w:workers){w.keys.resize(S2);w.bytes.resize(S2);w.kv.resize(D);
#if HAVE_SVE2
            // qbf: qt queries x (H/8 groups) x (D/2 windows) x 16 bf16 = qt*H*D uint16.
            // kdup: tile keys x (D/2 windows) x 16 bf16 = tile*D*8 uint16.
            w.qbf.resize(size_t(qt)*h*d);w.kdup.resize(size_t(tile_)*d*8);
#endif
#if HAVE_SME
            // qT: D x H FP32 transposed query; kblk: D x NW FP32 key block.
            w.qT.resize(size_t(d)*h);w.kblk.resize(size_t(d)*8);
#endif
#ifdef LI_USE_BLAS
            w.packed_key.resize(size_t(tile)*D);w.dots.resize(size_t(qtile)*H*tile);
#endif
        }
    }
};
void histogram(Worker& w,size_t n,int byte,int hist) {
#if HAVE_SVE2
    if(hist){histogram_sve2(w.keys.data(),n,byte,w.hist,w.bytes.data());return;}
#else
    (void)hist;
#endif
    histogram_scalar(w.keys.data(),n,byte,w.hist);
}
void select_row(Worker& w,const Row& a,const Workspace& x,const float* score,int32_t* out,int sorted,int hist) {
    const int v=a.v,K=x.K;const size_t row=a.index;size_t n=v;
    if(v>K) {
        for(int j=0;j<v;++j)w.keys[j]=float_to_sortable_key(score[row*x.S2+j]);
        int rank=K;uint32_t prefix=0;
        for(int byte=3;byte>=0;--byte) {
            histogram(w,n,byte,hist);int target=0;
            for(int z=255;z>=0;--z){if(rank>int(w.hist[z]))rank-=w.hist[z];else{target=z;break;}}
            prefix|=uint32_t(target)<<(8*byte);
            if(byte){size_t next=0;for(size_t z=0;z<n;++z)if(extract_byte(w.keys[z],byte)==target)w.keys[next++]=w.keys[z];n=next;}
        }
        int next=0;
        for(int j=0;j<v;++j)if(float_to_sortable_key(score[row*x.S2+j])>prefix)out[row*K+next++]=j;
        for(int j=0;j<v&&next<K;++j)if(float_to_sortable_key(score[row*x.S2+j])==prefix)out[row*K+next++]=j;
    } else for(int j=0;j<v;++j)out[row*K+j]=j;
    if(sorted)std::sort(out+row*K,out+row*K+std::min(v,K),[&](int a,int b){float va=score[row*x.S2+a],vb=score[row*x.S2+b];return va!=vb?va>vb:a<b;});
}

#if HAVE_SVE2
// ---------------------------------------------------------------------------
// SVE2 BF16 score kernel (backend 2).
//
// score[b,i,j] = sum_h W[b,i,h] * relu( sum_d Q[b,i,h,d] * K[b,j,d] )
//
// Layout trick to use one svbfdot_f32 per (head-group, 4-dim window):
//   qbf[i][hg][w][16] : { q(i,h0,dw..dw+3), q(i,h1,dw..dw+3), q(i,h2,dw..dw+3), q(i,h3,dw..dw+3) }
//   kdup[j][w][16]    : { k(j,dw)x4, k(j,dw+1)x4, k(j,dw+2)x4, k(j,dw+3)x4 }
// svbfdot_f32(acc, qbf16, kdup16): lane h += q(h,dw..dw+3) . k(dw..dw+3)
// So D=128 -> 32 svbfdot per 4-head group (16 MAC/instr), acc lanes 0..3 = 4 head dots.
// Requires H%4==0 and D%4==0 (all target workloads satisfy this).
// ---------------------------------------------------------------------------

// score[b,i,j] = sum_h W[b,i,h] * relu( sum_d Q[b,i,h,d] * K[b,j,d] )
//
// Measured svbfdot_f32 on Kunpeng 950 (VL=256): lane i computes a 2-way dot
//   acc[i] += op2[2i]*op3[2i] + op2[2i+1]*op3[2i+1]
// using only elements [0..15] of each 32-element operand. So one call = 8 lanes x
// 2 MAC = 16 MAC. Layout: lane = one head, 2 dims per window step.
//   qbf[i][g][w][16] : lane L, dim c -> qbf[(g*nw+w)*16 + L*2 + c] = q[g*8+L, w*2+c]
//   kdup[j][w][16]   : lane L, dim c -> kdup[w*16 + L*2 + c] = k[w*2+c]  (window per lane)
// Requires H%8==0 and D%2==0 (checked by caller); D=128 -> 64 windows x H/8 groups.
// Multi-key row kernels: compute scores for NK keys sharing one query, so the
// per-window qv load is amortized across NK keys (big win when H is large: the
// hg==8 kernel previously re-read all of qbf for every key).
// kdup for NK keys must be laid out contiguously: kdup_multi + key*D*8.

// hg==2 (H==16): 4 keys at once. acc regs = 8, kv = 4, qv = 1 -> 13 live Z regs.
static inline void score_bf16_4row_hg2(const uint16_t* qbf, const uint16_t* kdup,
                                       const float* W, int H, int D, size_t row,
                                       float* out4) {
    const int nw = D / 2;
    const auto* qb = reinterpret_cast<const bfloat16_t*>(qbf);
    svfloat32_t a00=svdup_f32(0.f),a01=svdup_f32(0.f),a10=svdup_f32(0.f),a11=svdup_f32(0.f);
    svfloat32_t a20=svdup_f32(0.f),a21=svdup_f32(0.f),a30=svdup_f32(0.f),a31=svdup_f32(0.f);
    for (int w = 0; w < nw; ++w) {
        auto kv0=svld1_bf16(svptrue_b16(),reinterpret_cast<const bfloat16_t*>(kdup)+(0*D*8+w*16));
        auto kv1=svld1_bf16(svptrue_b16(),reinterpret_cast<const bfloat16_t*>(kdup)+(1*D*8+w*16));
        auto kv2=svld1_bf16(svptrue_b16(),reinterpret_cast<const bfloat16_t*>(kdup)+(2*D*8+w*16));
        auto kv3=svld1_bf16(svptrue_b16(),reinterpret_cast<const bfloat16_t*>(kdup)+(3*D*8+w*16));
        auto q0=svld1_bf16(svptrue_b16(),qb+(0*nw+w)*16);
        auto q1=svld1_bf16(svptrue_b16(),qb+(1*nw+w)*16);
        a00=svbfdot_f32(a00,q0,kv0); a01=svbfdot_f32(a01,q1,kv0);
        a10=svbfdot_f32(a10,q0,kv1); a11=svbfdot_f32(a11,q1,kv1);
        a20=svbfdot_f32(a20,q0,kv2); a21=svbfdot_f32(a21,q1,kv2);
        a30=svbfdot_f32(a30,q0,kv3); a31=svbfdot_f32(a31,q1,kv3);
    }
    const float* Wr = W + row * H;
    svfloat32_t r0 = svdup_f32(0.f), r1 = svdup_f32(0.f), r2 = svdup_f32(0.f), r3 = svdup_f32(0.f);
    r0 = svmla_f32_x(svptrue_b32(), r0, svmaxnm_f32_z(svptrue_b32(), a00, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 0));
    r0 = svmla_f32_x(svptrue_b32(), r0, svmaxnm_f32_z(svptrue_b32(), a01, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 8));
    r1 = svmla_f32_x(svptrue_b32(), r1, svmaxnm_f32_z(svptrue_b32(), a10, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 0));
    r1 = svmla_f32_x(svptrue_b32(), r1, svmaxnm_f32_z(svptrue_b32(), a11, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 8));
    r2 = svmla_f32_x(svptrue_b32(), r2, svmaxnm_f32_z(svptrue_b32(), a20, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 0));
    r2 = svmla_f32_x(svptrue_b32(), r2, svmaxnm_f32_z(svptrue_b32(), a21, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 8));
    r3 = svmla_f32_x(svptrue_b32(), r3, svmaxnm_f32_z(svptrue_b32(), a30, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 0));
    r3 = svmla_f32_x(svptrue_b32(), r3, svmaxnm_f32_z(svptrue_b32(), a31, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 8));
    out4[0] += svaddv_f32(svptrue_b32(), r0);
    out4[1] += svaddv_f32(svptrue_b32(), r1);
    out4[2] += svaddv_f32(svptrue_b32(), r2);
    out4[3] += svaddv_f32(svptrue_b32(), r3);
}

// hg==8 (H==64, decode_64k): 2 keys at once. acc = 16, kv = 2, qv = 8 -> 26 live Z regs.
static inline void score_bf16_2row_hg8(const uint16_t* qbf, const uint16_t* kdup,
                                       const float* W, int H, int D, size_t row,
                                       float* out2) {
    const int nw = D / 2;
    const auto* qb = reinterpret_cast<const bfloat16_t*>(qbf);
    svfloat32_t x00=svdup_f32(0.f),x01=svdup_f32(0.f),x02=svdup_f32(0.f),x03=svdup_f32(0.f);
    svfloat32_t x04=svdup_f32(0.f),x05=svdup_f32(0.f),x06=svdup_f32(0.f),x07=svdup_f32(0.f);
    svfloat32_t y00=svdup_f32(0.f),y01=svdup_f32(0.f),y02=svdup_f32(0.f),y03=svdup_f32(0.f);
    svfloat32_t y04=svdup_f32(0.f),y05=svdup_f32(0.f),y06=svdup_f32(0.f),y07=svdup_f32(0.f);
    for (int w = 0; w < nw; ++w) {
        auto kv0=svld1_bf16(svptrue_b16(),reinterpret_cast<const bfloat16_t*>(kdup)+(0*D*8+w*16));
        auto kv1=svld1_bf16(svptrue_b16(),reinterpret_cast<const bfloat16_t*>(kdup)+(1*D*8+w*16));
        auto q0=svld1_bf16(svptrue_b16(),qb+(0*nw+w)*16);
        auto q1=svld1_bf16(svptrue_b16(),qb+(1*nw+w)*16);
        auto q2=svld1_bf16(svptrue_b16(),qb+(2*nw+w)*16);
        auto q3=svld1_bf16(svptrue_b16(),qb+(3*nw+w)*16);
        auto q4=svld1_bf16(svptrue_b16(),qb+(4*nw+w)*16);
        auto q5=svld1_bf16(svptrue_b16(),qb+(5*nw+w)*16);
        auto q6=svld1_bf16(svptrue_b16(),qb+(6*nw+w)*16);
        auto q7=svld1_bf16(svptrue_b16(),qb+(7*nw+w)*16);
        x00=svbfdot_f32(x00,q0,kv0); x01=svbfdot_f32(x01,q1,kv0);
        x02=svbfdot_f32(x02,q2,kv0); x03=svbfdot_f32(x03,q3,kv0);
        x04=svbfdot_f32(x04,q4,kv0); x05=svbfdot_f32(x05,q5,kv0);
        x06=svbfdot_f32(x06,q6,kv0); x07=svbfdot_f32(x07,q7,kv0);
        y00=svbfdot_f32(y00,q0,kv1); y01=svbfdot_f32(y01,q1,kv1);
        y02=svbfdot_f32(y02,q2,kv1); y03=svbfdot_f32(y03,q3,kv1);
        y04=svbfdot_f32(y04,q4,kv1); y05=svbfdot_f32(y05,q5,kv1);
        y06=svbfdot_f32(y06,q6,kv1); y07=svbfdot_f32(y07,q7,kv1);
    }
    const float* Wr = W + row * H;
    svfloat32_t r0 = svdup_f32(0.f), r1 = svdup_f32(0.f);
    r0 = svmla_f32_x(svptrue_b32(), r0, svmaxnm_f32_z(svptrue_b32(), x00, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 0*8));
    r1 = svmla_f32_x(svptrue_b32(), r1, svmaxnm_f32_z(svptrue_b32(), y00, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 0*8));
    r0 = svmla_f32_x(svptrue_b32(), r0, svmaxnm_f32_z(svptrue_b32(), x01, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 1*8));
    r1 = svmla_f32_x(svptrue_b32(), r1, svmaxnm_f32_z(svptrue_b32(), y01, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 1*8));
    r0 = svmla_f32_x(svptrue_b32(), r0, svmaxnm_f32_z(svptrue_b32(), x02, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 2*8));
    r1 = svmla_f32_x(svptrue_b32(), r1, svmaxnm_f32_z(svptrue_b32(), y02, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 2*8));
    r0 = svmla_f32_x(svptrue_b32(), r0, svmaxnm_f32_z(svptrue_b32(), x03, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 3*8));
    r1 = svmla_f32_x(svptrue_b32(), r1, svmaxnm_f32_z(svptrue_b32(), y03, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 3*8));
    r0 = svmla_f32_x(svptrue_b32(), r0, svmaxnm_f32_z(svptrue_b32(), x04, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 4*8));
    r1 = svmla_f32_x(svptrue_b32(), r1, svmaxnm_f32_z(svptrue_b32(), y04, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 4*8));
    r0 = svmla_f32_x(svptrue_b32(), r0, svmaxnm_f32_z(svptrue_b32(), x05, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 5*8));
    r1 = svmla_f32_x(svptrue_b32(), r1, svmaxnm_f32_z(svptrue_b32(), y05, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 5*8));
    r0 = svmla_f32_x(svptrue_b32(), r0, svmaxnm_f32_z(svptrue_b32(), x06, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 6*8));
    r1 = svmla_f32_x(svptrue_b32(), r1, svmaxnm_f32_z(svptrue_b32(), y06, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 6*8));
    r0 = svmla_f32_x(svptrue_b32(), r0, svmaxnm_f32_z(svptrue_b32(), x07, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 7*8));
    r1 = svmla_f32_x(svptrue_b32(), r1, svmaxnm_f32_z(svptrue_b32(), y07, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + 7*8));
    out2[0] += svaddv_f32(svptrue_b32(), r0);
    out2[1] += svaddv_f32(svptrue_b32(), r1);
}


// ---------------------------------------------------------------------------
// Compact lane=dim-pair kernel (memory-bandwidth friendly).
//
// kdup_c[j][w][16]: element [2i] = k[j, 16w+2i], [2i+1] = k[j, 16w+2i+1]
//   -> lane i accumulates dim-pair i over all windows = full dot for that head.
//   No 8x broadcast redundancy: each key is exactly D bf16 = D*2 bytes.
// qbf_c[i][h][w][16]: element [2i'] = q[i,h,16w+2i'], [2i'+1]=q[i,h,16w+2i'+1]
//
// Kernel processes 1 key, looping heads NH=4 at a time with 2 accumulators per
// head (odd/even window interleave) to break the bfdot dependency chain. kv is
// shared across the 4 heads. Requires D%16==0 (D=128 -> nw16=8 windows).
// ---------------------------------------------------------------------------
static inline float score_bf16_compact(const uint16_t* qbf_c, const uint16_t* kdup_c,
                                       const float* W, int H, int D, size_t row) {
    const int nw16 = D / 16;  // 16-dim windows per head
    const auto* kd = reinterpret_cast<const bfloat16_t*>(kdup_c);
    float score = 0.f;
    for (int h = 0; h < H; h += 4) {
        const auto* qb0 = reinterpret_cast<const bfloat16_t*>(qbf_c + (size_t)(h+0)*nw16*16);
        const auto* qb1 = reinterpret_cast<const bfloat16_t*>(qbf_c + (size_t)(h+1)*nw16*16);
        const auto* qb2 = reinterpret_cast<const bfloat16_t*>(qbf_c + (size_t)(h+2)*nw16*16);
        const auto* qb3 = reinterpret_cast<const bfloat16_t*>(qbf_c + (size_t)(h+3)*nw16*16);
        svfloat32_t a00=svdup_f32(0.f),a01=svdup_f32(0.f),a10=svdup_f32(0.f),a11=svdup_f32(0.f);
        svfloat32_t a20=svdup_f32(0.f),a21=svdup_f32(0.f),a30=svdup_f32(0.f),a31=svdup_f32(0.f);
        for (int w = 0; w < nw16; w += 2) {
            auto kv0=svld1_bf16(svptrue_b16(),kd+(w+0)*16);
            auto kv1=svld1_bf16(svptrue_b16(),kd+(w+1)*16);
            auto q0_0=svld1_bf16(svptrue_b16(),qb0+(w+0)*16);
            auto q0_1=svld1_bf16(svptrue_b16(),qb0+(w+1)*16);
            auto q1_0=svld1_bf16(svptrue_b16(),qb1+(w+0)*16);
            auto q1_1=svld1_bf16(svptrue_b16(),qb1+(w+1)*16);
            auto q2_0=svld1_bf16(svptrue_b16(),qb2+(w+0)*16);
            auto q2_1=svld1_bf16(svptrue_b16(),qb2+(w+1)*16);
            auto q3_0=svld1_bf16(svptrue_b16(),qb3+(w+0)*16);
            auto q3_1=svld1_bf16(svptrue_b16(),qb3+(w+1)*16);
            a00=svbfdot_f32(a00,kv0,q0_0); a01=svbfdot_f32(a01,kv1,q0_1);
            a10=svbfdot_f32(a10,kv0,q1_0); a11=svbfdot_f32(a11,kv1,q1_1);
            a20=svbfdot_f32(a20,kv0,q2_0); a21=svbfdot_f32(a21,kv1,q2_1);
            a30=svbfdot_f32(a30,kv0,q3_0); a31=svbfdot_f32(a31,kv1,q3_1);
        }
        float d[8];
        svst1_f32(svptrue_b32(),d,a00); float dot0=0; for(int i=0;i<8;++i)dot0+=d[i];
        svst1_f32(svptrue_b32(),d,a01); for(int i=0;i<8;++i)dot0+=d[i];
        svst1_f32(svptrue_b32(),d,a10); float dot1=0; for(int i=0;i<8;++i)dot1+=d[i];
        svst1_f32(svptrue_b32(),d,a11); for(int i=0;i<8;++i)dot1+=d[i];
        svst1_f32(svptrue_b32(),d,a20); float dot2=0; for(int i=0;i<8;++i)dot2+=d[i];
        svst1_f32(svptrue_b32(),d,a21); for(int i=0;i<8;++i)dot2+=d[i];
        svst1_f32(svptrue_b32(),d,a30); float dot3=0; for(int i=0;i<8;++i)dot3+=d[i];
        svst1_f32(svptrue_b32(),d,a31); for(int i=0;i<8;++i)dot3+=d[i];
        const float* wh = W + row*H + h;
        score += wh[0]*(dot0>0.f?dot0:0.f) + wh[1]*(dot1>0.f?dot1:0.f)
               + wh[2]*(dot2>0.f?dot2:0.f) + wh[3]*(dot3>0.f?dot3:0.f);
    }
    return score;
}


// 4row_hg4: 4 keys x 4 head-groups (32 heads) per call. lane=head (8 heads/group).
// kdup[j][w][16]: [2L]=k[j,2w],[2L+1]=k[j,2w+1] broadcast; qbf[i][g][w][16]: [2L]=q[i,g*8+L,2w]
// per window: 4 kv + 4 qv = 8 ld, 16 bfdot -> 0.5 ld/bfdot. 16 acc (4x4) = 24 live Z regs.
// Requires D%2==0, H%8==0; caller splits H into groups-of-4 blocks (hg4 calls for H/32 blocks).
static inline void score_bf16_4row_hg4(const uint16_t* qbf, const uint16_t* kdup,
                                       const float* W, int H, int D, size_t row,
                                       int g0, float* out4) {
    const int nw = D / 2;
    const auto* qb = reinterpret_cast<const bfloat16_t*>(qbf);
    svfloat32_t a00=svdup_f32(0.f),a01=svdup_f32(0.f),a02=svdup_f32(0.f),a03=svdup_f32(0.f);
    svfloat32_t a10=svdup_f32(0.f),a11=svdup_f32(0.f),a12=svdup_f32(0.f),a13=svdup_f32(0.f);
    svfloat32_t a20=svdup_f32(0.f),a21=svdup_f32(0.f),a22=svdup_f32(0.f),a23=svdup_f32(0.f);
    svfloat32_t a30=svdup_f32(0.f),a31=svdup_f32(0.f),a32=svdup_f32(0.f),a33=svdup_f32(0.f);
    for (int w = 0; w < nw; ++w) {
        auto kv0=svld1_bf16(svptrue_b16(),reinterpret_cast<const bfloat16_t*>(kdup)+(0*D*8+w*16));
        auto kv1=svld1_bf16(svptrue_b16(),reinterpret_cast<const bfloat16_t*>(kdup)+(1*D*8+w*16));
        auto kv2=svld1_bf16(svptrue_b16(),reinterpret_cast<const bfloat16_t*>(kdup)+(2*D*8+w*16));
        auto kv3=svld1_bf16(svptrue_b16(),reinterpret_cast<const bfloat16_t*>(kdup)+(3*D*8+w*16));
        auto q0=svld1_bf16(svptrue_b16(),qb+(0*nw+w)*16);
        auto q1=svld1_bf16(svptrue_b16(),qb+(1*nw+w)*16);
        auto q2=svld1_bf16(svptrue_b16(),qb+(2*nw+w)*16);
        auto q3=svld1_bf16(svptrue_b16(),qb+(3*nw+w)*16);
        a00=svbfdot_f32(a00,q0,kv0); a01=svbfdot_f32(a01,q1,kv0);
        a02=svbfdot_f32(a02,q2,kv0); a03=svbfdot_f32(a03,q3,kv0);
        a10=svbfdot_f32(a10,q0,kv1); a11=svbfdot_f32(a11,q1,kv1);
        a12=svbfdot_f32(a12,q2,kv1); a13=svbfdot_f32(a13,q3,kv1);
        a20=svbfdot_f32(a20,q0,kv2); a21=svbfdot_f32(a21,q1,kv2);
        a22=svbfdot_f32(a22,q2,kv2); a23=svbfdot_f32(a23,q3,kv2);
        a30=svbfdot_f32(a30,q0,kv3); a31=svbfdot_f32(a31,q1,kv3);
        a32=svbfdot_f32(a32,q2,kv3); a33=svbfdot_f32(a33,q3,kv3);
    }
    // Vectorized W-relu epilogue: running 8-lane sums, one horizontal reduce per key.
    const float* Wr = W + row * H;
    svfloat32_t r0 = svdup_f32(0.f), r1 = svdup_f32(0.f), r2 = svdup_f32(0.f), r3 = svdup_f32(0.f);
    r0 = svmla_f32_x(svptrue_b32(), r0, svmaxnm_f32_z(svptrue_b32(), a00, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + (g0+0)*8));
    r0 = svmla_f32_x(svptrue_b32(), r0, svmaxnm_f32_z(svptrue_b32(), a01, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + (g0+1)*8));
    r0 = svmla_f32_x(svptrue_b32(), r0, svmaxnm_f32_z(svptrue_b32(), a02, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + (g0+2)*8));
    r0 = svmla_f32_x(svptrue_b32(), r0, svmaxnm_f32_z(svptrue_b32(), a03, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + (g0+3)*8));
    r1 = svmla_f32_x(svptrue_b32(), r1, svmaxnm_f32_z(svptrue_b32(), a10, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + (g0+0)*8));
    r1 = svmla_f32_x(svptrue_b32(), r1, svmaxnm_f32_z(svptrue_b32(), a11, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + (g0+1)*8));
    r1 = svmla_f32_x(svptrue_b32(), r1, svmaxnm_f32_z(svptrue_b32(), a12, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + (g0+2)*8));
    r1 = svmla_f32_x(svptrue_b32(), r1, svmaxnm_f32_z(svptrue_b32(), a13, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + (g0+3)*8));
    r2 = svmla_f32_x(svptrue_b32(), r2, svmaxnm_f32_z(svptrue_b32(), a20, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + (g0+0)*8));
    r2 = svmla_f32_x(svptrue_b32(), r2, svmaxnm_f32_z(svptrue_b32(), a21, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + (g0+1)*8));
    r2 = svmla_f32_x(svptrue_b32(), r2, svmaxnm_f32_z(svptrue_b32(), a22, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + (g0+2)*8));
    r2 = svmla_f32_x(svptrue_b32(), r2, svmaxnm_f32_z(svptrue_b32(), a23, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + (g0+3)*8));
    r3 = svmla_f32_x(svptrue_b32(), r3, svmaxnm_f32_z(svptrue_b32(), a30, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + (g0+0)*8));
    r3 = svmla_f32_x(svptrue_b32(), r3, svmaxnm_f32_z(svptrue_b32(), a31, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + (g0+1)*8));
    r3 = svmla_f32_x(svptrue_b32(), r3, svmaxnm_f32_z(svptrue_b32(), a32, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + (g0+2)*8));
    r3 = svmla_f32_x(svptrue_b32(), r3, svmaxnm_f32_z(svptrue_b32(), a33, svdup_f32(0.f)), svld1_f32(svptrue_b32(), Wr + (g0+3)*8));
    out4[0] += svaddv_f32(svptrue_b32(), r0);
    out4[1] += svaddv_f32(svptrue_b32(), r1);
    out4[2] += svaddv_f32(svptrue_b32(), r2);
    out4[3] += svaddv_f32(svptrue_b32(), r3);
}

// Single-key kernel for remaining shapes.
static inline float score_bf16_row(const uint16_t* qbf, const uint16_t* kdup,
                                   const float* W, int H, int D, size_t row) {
    const int hg = H / 8, nw = D / 2;
    const auto* kd = reinterpret_cast<const bfloat16_t*>(kdup);
    float score = 0.f;
    for (int g = 0; g < hg; ++g) {
        svfloat32_t acc = svdup_f32(0.f);
        for (int w = 0; w < nw; ++w) {
            auto kv = svld1_bf16(svptrue_b16(), kd + w * 16);
            auto qv = svld1_bf16(svptrue_b16(), reinterpret_cast<const bfloat16_t*>(qbf) + (size_t(g) * nw + w) * 16);
            acc = svbfdot_f32(acc, qv, kv);
        }
        float d[8]; svst1_f32(svptrue_b32(), d, acc);
        const int h = g * 8;
        for (int lane = 0; lane < 8; ++lane)
            score += W[row * H + h + lane] * std::max(d[lane], 0.f);
    }
    return score;
}
#endif

#if HAVE_SME
// ---------------------------------------------------------------------------
// SME BF16 score kernel (compile-time alternative to the SVE2 backend).
//
// Build with scripts/build_cpu.sh --sme (-DLI_SCORE_USE_SME,
// -march=armv9-a+sve2+bf16+sme). The runtime --score-backend remains
// simd/blas/svebf16/sme; the accelerated backend then executes this kernel.
//
// QK^T for one query row is expressed as an FP32 outer-product accumulation:
//   ZA[h][j] += Q[h][d] * K[j][d]        for every d
// using SME SV MOPA svmopa_za32_f32_m (ZA32 tile NW x NW, NW=svcntw()).
// BF16 inputs are widened to FP32 once (query transposed D x H, keys D x NW)
// so the SME dot is numerically identical to the FP32 reference used by the
// golden validation. For SVL=256 the tile is 8 heads x 8 keys; H must be a
// multiple of NW. Rows are reduced as relu(dot)*W over heads.
// ---------------------------------------------------------------------------
LI_ARM_STREAMING LI_ARM_NEW_ZA
static void score_sme_8keys(const float* qT,const float* kblk,const float* W,
                            int H,int D,size_t row,float* out8) {
    const svbool_t pg=svptrue_b32();
    const int nw=int(svcntw());
    for(int h0=0;h0<H;h0+=nw) {
        svzero_za();
        for(int d=0;d<D;++d) {
            svfloat32_t zn=svld1_f32(pg,qT+size_t(d)*H+h0);
            svfloat32_t zm=svld1_f32(pg,kblk+size_t(d)*nw);
            svmopa_za32_f32_m(0,pg,pg,zn,zm);
        }
        svfloat32_t acc=svdup_f32(0.f);
        const float* Wr=W+row*H+h0;
        for(int i=0;i<nw;++i) {
            svfloat32_t r=svread_hor_za32_f32_m(svundef_f32(),pg,0,uint32_t(i));
            r=svmaxnm_f32_z(pg,r,svdup_f32(0.f));
            // svmla is not streaming-compatible; use mul+add to stay defined.
            acc=svadd_f32_x(pg,acc,svmul_f32_x(pg,r,svdup_f32(Wr[i])));
        }
        svfloat32_t o=svld1_f32(pg,out8);
        svst1_f32(pg,out8,svadd_f32_x(pg,o,acc));
    }
}
#endif

}
extern "C" {
int li_abi_version(){return 3;}
int li_has_blas(){
#ifdef LI_USE_BLAS
    return 1;
#else
    return 0;
#endif
}
int li_set_score_backend(void* handle,int backend){
    if(!handle||backend<0||backend>2)return -1;
    if(backend==1&&!li_has_blas())return -7;
#if !HAVE_SVE2 && !HAVE_SME
    if(backend==2)return -8;   // accelerated BF16 backend unavailable in scalar build
#endif
    static_cast<Workspace*>(handle)->score_backend=backend;return 0;
}
int li_has_sve2(){return HAVE_SVE2;}
int li_has_sme(){return HAVE_SME;}
int li_vl(){
#if HAVE_SVE2
    return svcntb()*8;
#else
    return 0;
#endif
}
// Diagnostic entry used by test_native.py to exercise all bins and vector tails.
int li_histogram_check(const uint32_t* keys,int n,int byte,int hist,uint32_t* h,uint8_t* scratch) {
    if(n<0||byte<0||byte>3)return -1;
#if HAVE_SVE2
    if(hist){if(li_vl()!=256)return -2;histogram_sve2(keys,n,byte,h,scratch);return 0;}
#else
    (void)scratch;
    if(hist)return -4;
#endif
    histogram_scalar(keys,n,byte,h);return 0;
}
void* li_create(int B,int S1,int S2,int H,int D,int K,int P,int tile,int qtile) {
    if(B<1||S1<1||S2<1||H<1||D<1||K<1||P<1||tile<1||qtile<1)return nullptr;
    try{return new Workspace(B,S1,S2,H,D,K,P,tile,qtile);}catch(...){return nullptr;}
}
void li_destroy(void* x){delete static_cast<Workspace*>(x);}
int li_worker_info(void* x,int32_t* cpus,int32_t* vls,int capacity,int32_t* group_count) {
    auto& w=*static_cast<Workspace*>(x);
    int n=std::min(capacity,w.actual_threads);
    for(int i=0;i<n;++i){cpus[i]=w.workers[i].cpu;vls[i]=w.workers[i].vl;}
    *group_count=w.actual_groups;return w.actual_threads;
}
// q/w lengths fixed by workspace; keys contiguous BSND or physical PA_BSND N2=1.
// hist=1 requests SVE2 explicitly: fail instead of silently using scalar fallback.
int li_execute(void* handle,const uint16_t* q,const uint16_t* k,const uint16_t* w,
    const int32_t* table,const int32_t* lq,const int32_t* lk,
    int physical_tokens,int bs,int blocks_per_batch,int bf,int mode,int sorted,int hist,int topk_groups,
    int32_t* out,float* scores,double* timing) {
    if(!handle||!q||!k||!w||!lq||!lk||!out||!scores||!timing)return -1;
    auto& x=*static_cast<Workspace*>(handle);
    const int B=x.B,S1=x.S1,S2=x.S2,H=x.H,D=x.D,K=x.K,P=x.P;
    if(physical_tokens<1||bs<0||blocks_per_batch<0||(bf!=0&&bf!=1)||(mode!=0&&mode!=3)||
       (hist!=0&&hist!=1)||topk_groups<0||topk_groups>P)return -1;
#if !HAVE_SVE2
    if(hist)return -4;
#endif
    // Accelerated BF16 backend (SVE2 svbfdot or SME MOPA) needs H%8==0, D%2==0
    // and VL=256; otherwise degrade to simd instead of failing.
    int backend = x.score_backend;
#if HAVE_SVE2 || HAVE_SME
    if(backend==2 && (H%8 || D%2 || li_vl()!=256)) backend = 0;
#else
    if(backend==2) return -8;
#endif
    // Validate native entry too; never access an invalid PA table from unchecked callers.
    for(int b=0;b<B;++b) {
        if(lq[b]<0||lq[b]>S1||lk[b]<0||lk[b]>S2)return -1;
        if(bs) {
            if(!table||int64_t(blocks_per_batch)*bs<lk[b])return -1;
            for(int j=0;j<(lk[b]+bs-1)/bs;++j)
                if(table[size_t(b)*blocks_per_batch+j]<0||
                   int64_t(table[size_t(b)*blocks_per_batch+j])*bs+bs>physical_tokens)return -1;
        } else if(int64_t(B)*S2>physical_tokens)return -1;
    }
    const auto t0=Clock::now();
    x.rows.clear();x.tasks.clear();
    for(int b=0;b<B;++b) {
        for(int i=0;i<lq[b];++i) {
            int v=mode==3?std::max(0,lk[b]-lq[b]+i+1):lk[b];
            x.rows.push_back({b,i,v,size_t(b)*S1+i});
        }
        for(int i=0;i<lq[b];i+=x.qtile) {
            int last=std::min(i+x.qtile,lq[b]);
            int v=mode==3?std::max(0,lk[b]-lq[b]+last):lk[b];
            for(int start=0;start<v;start+=x.tile)
                x.tasks.push_back({b,i,last,start,std::min(start+x.tile,v)});
        }
    }
    Clock::time_point t1,t2,t3;
    int status=0,nt=0,G=0;
    #pragma omp parallel num_threads(P) shared(status,nt,G,t1,t2,t3)
    {
        const int tid=omp_get_thread_num();
        auto& work=x.workers[tid];
        work.cpu=sched_getcpu();work.vl=li_vl();
        #pragma omp single
        {nt=omp_get_num_threads();x.actual_threads=nt;
         G=std::max(1,std::min(nt,topk_groups?topk_groups:int(x.rows.size())));x.actual_groups=G;}
        if(hist&&work.vl!=256) {
            #pragma omp atomic write
            status=-2;
        }
        #pragma omp barrier
        if(!status) {
            if(backend!=2) {
                #pragma omp for schedule(static) nowait
                for(size_t j=0;j<x.Q.size();++j)x.Q[j]=decode(q[j],bf);
            }
            #pragma omp for schedule(static) nowait
            for(size_t j=0;j<x.W.size();++j)x.W[j]=decode(w[j],bf);
            #pragma omp for schedule(static) nowait
            for(size_t j=0;j<size_t(B)*S1*S2;++j)scores[j]=-std::numeric_limits<float>::infinity();
            #pragma omp for schedule(static)
            for(size_t j=0;j<size_t(B)*S1*K;++j)out[j]=-1;
            #pragma omp single
            t1=Clock::now();
            #pragma omp for schedule(static)
            for(size_t t=0;t<x.tasks.size();++t) {
                const auto a=x.tasks[t];
#ifdef LI_USE_BLAS
                if(backend==1) {
                    const int n=a.end-a.start,m=(a.q1-a.q0)*H;
                    for(int j=a.start;j<a.end;++j) {
                        size_t pos=bs?size_t(table[size_t(a.b)*blocks_per_batch+j/bs])*bs+j%bs:size_t(a.b)*S2+j;
                        for(int d=0;d<D;++d)work.packed_key[size_t(j-a.start)*D+d]=decode(k[pos*D+d],bf);
                    }
                    // One GEMM for all Query/head rows sharing this Key tile; no head-wise Key duplication.
                    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,n,D,1.f,
                        x.Q.data()+(size_t(a.b)*S1+a.q0)*H*D,D,work.packed_key.data(),D,0.f,work.dots.data(),n);
                    for(int i=a.q0;i<a.q1;++i)for(int j=a.start;j<a.end;++j) {
                        if(mode==3&&j>=lk[a.b]-lq[a.b]+i+1)continue;
                        size_t row=size_t(a.b)*S1+i;float value=0;
                        for(int h=0;h<H;++h)value+=x.W[row*H+h]*std::max(0.f,work.dots[(size_t(i-a.q0)*H+h)*n+j-a.start]);
                        if(!std::isfinite(value)){
                            #pragma omp atomic write
                            status=-6;
                        }
                        scores[row*S2+j]=value;
                    }
                    continue;
                }
#endif
#if HAVE_SME
                if(backend==2 && H%int(svcntw())==0) {
                    const int nw=int(svcntw());
                    for(int i=a.q0;i<a.q1;++i) {
                        const size_t row=size_t(a.b)*S1+i;
                        const uint16_t* qr=q+(size_t(a.b)*S1+i)*H*D;
                        float* qT=work.qT.data();
                        for(int h=0;h<H;++h) {
                            const uint16_t* src=qr+size_t(h)*D;
                            for(int d=0;d<D;++d)qT[size_t(d)*H+h]=decode(src[d],bf);
                        }
                        const int lim=mode==3?lk[a.b]-lq[a.b]+i+1:a.end;
                        for(int jg=a.start;jg<a.end;jg+=nw) {
                            const int jmax=std::min(jg+nw,a.end);
                            for(int r=0;r<nw;++r) {
                                const int j=jg+r;
                                const uint16_t* kp=nullptr;
                                if(j<a.end) {
                                    size_t pos=bs?size_t(table[size_t(a.b)*blocks_per_batch+j/bs])*bs+j%bs:size_t(a.b)*S2+j;
                                    kp=k+size_t(pos)*D;
                                }
                                for(int d=0;d<D;++d)
                                    work.kblk[size_t(d)*nw+r]=kp?decode(kp[d],bf):0.f;
                            }
                            float out8[8]={0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
                            score_sme_8keys(qT,work.kblk.data(),x.W.data(),H,D,row,out8);
                            for(int j=jg;j<jmax;++j) {
                                if(mode==3&&j>=lim)continue;
                                const float value=out8[j-jg];
                                if(!std::isfinite(value)) {
                                    #pragma omp atomic write
                                    status=-6;
                                }
                                scores[row*S2+j]=value;
                            }
                        }
                    }
                    continue;
                }
#endif
#if HAVE_SVE2
                if(backend==2) {
                    if(H%32==0||H==16) {
                        // --- lane=head path: 4row_hg4 (H%32==0) or 4row_hg2 (H==16) ---
                        const int nw=D/2, ng=H/8;   // H=64 -> nw=64, ng=8 (2 blocks of 4 groups)
// kdup4[j][w][16]: [2L]=k[j,2w],[2L+1]=k[j,2w+1] broadcast (D*8 bf16/key)
                        // SVE vectorized build: one 16-lane load per 16 dims, svtbl per
                        // window reproduces [k(2w),k(2w+1)]x8 in a single 32B store.
                        alignas(64) static const uint16_t kdup_idx[8][16] = {
                            {0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1},
                            {2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3},
                            {4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5},
                            {6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7},
                            {8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9},
                            {10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11},
                            {12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13},
                            {14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15}};
                        // --- fused main loop: build 4-key kdup in L1, then dot ---
                        // kdup4[j][w][16]: [2L]=k[j,2w],[2L+1]=k[j,2w+1] broadcast (D*8 bf16/key)
                        // Only 8KB (4 keys) is resident at a time instead of a whole tile's 256KB,
                        // so the dot pass reads kdup from L1 instead of L2/DRAM.
                        const auto iv0=svld1_u16(svptrue_b16(),kdup_idx[0]);
                        const auto iv1=svld1_u16(svptrue_b16(),kdup_idx[1]);
                        const auto iv2=svld1_u16(svptrue_b16(),kdup_idx[2]);
                        const auto iv3=svld1_u16(svptrue_b16(),kdup_idx[3]);
                        const auto iv4=svld1_u16(svptrue_b16(),kdup_idx[4]);
                        const auto iv5=svld1_u16(svptrue_b16(),kdup_idx[5]);
                        const auto iv6=svld1_u16(svptrue_b16(),kdup_idx[6]);
                        const auto iv7=svld1_u16(svptrue_b16(),kdup_idx[7]);
                        // qbf4[i][g][w][16]: [2L]=q[i,g*8+L,2w],[2L+1]=q[i,g*8+L,2w+1]
                        for(int i=a.q0;i<a.q1;++i) {
                            const uint16_t* qr=q+(size_t(a.b)*S1+i)*H*D;
                            uint16_t* qb=work.qbf.data()+size_t(i-a.q0)*H*D;
                            for(int g=0;g<ng;++g)
                                for(int w=0;w<nw;++w){
                                    const uint16_t* qs=qr+(size_t(g)*8*D+w*2);
                                    uint16_t* dst=qb+((size_t(g)*nw+w)*16);
                                    for(int L=0;L<8;++L){dst[L*2]=qs[L*D];dst[L*2+1]=qs[L*D+1];}
                                }
                        }
                        uint16_t* const kdb = work.kdup.data();
                        for(int jg=a.start;jg<a.end;jg+=4) {
                            const int jmax=std::min(jg+4,a.end);
                            for(int r=0;r<4;++r) {
                                const int j=std::min(jg+r,a.end-1);
                                size_t pos=bs?size_t(table[size_t(a.b)*blocks_per_batch+j/bs])*bs+j%bs:size_t(a.b)*S2+j;
                                const uint16_t* kp=k+size_t(pos)*D;
                                const uint16_t* kp16=reinterpret_cast<const uint16_t*>(kp);
                                uint16_t* kd=kdb+size_t(r)*D*8;
                                for(int t=0;t<D/16;++t) {
                                    const auto a16=svld1_u16(svptrue_b16(),kp16+t*16);
                                    uint16_t* dst=kd+t*16*8;
                                    svst1_u16(svptrue_b16(),dst+0*16,svtbl_u16(a16,iv0));
                                    svst1_u16(svptrue_b16(),dst+1*16,svtbl_u16(a16,iv1));
                                    svst1_u16(svptrue_b16(),dst+2*16,svtbl_u16(a16,iv2));
                                    svst1_u16(svptrue_b16(),dst+3*16,svtbl_u16(a16,iv3));
                                    svst1_u16(svptrue_b16(),dst+4*16,svtbl_u16(a16,iv4));
                                    svst1_u16(svptrue_b16(),dst+5*16,svtbl_u16(a16,iv5));
                                    svst1_u16(svptrue_b16(),dst+6*16,svtbl_u16(a16,iv6));
                                    svst1_u16(svptrue_b16(),dst+7*16,svtbl_u16(a16,iv7));
                                }
                            }
                            for(int i=a.q0;i<a.q1;++i) {
                                const size_t row=size_t(a.b)*S1+i;
                                const int lim=mode==3?lk[a.b]-lq[a.b]+i+1:a.end;
                                float out4[4]={0.f,0.f,0.f,0.f};
                                const uint16_t* qbase=work.qbf.data()+size_t(i-a.q0)*H*D;
                                if(H==16)
                                    score_bf16_4row_hg2(qbase,kdb,
                                                        x.W.data(),H,D,row,out4);
                                else
                                for(int g0=0;g0<ng;g0+=4)
                                    score_bf16_4row_hg4(qbase+size_t(g0)*D*8,
                                                        kdb,
                                                        x.W.data(),H,D,row,g0,out4);
                                for(int j=jg;j<jmax;++j){
                                    if(mode==3&&j>=lim)continue;
                                    const float value=out4[j-jg];
                                    if(!std::isfinite(value)) {
                                        #pragma omp atomic write
                                        status=-6;
                                    }
                                    scores[row*S2+j]=value;
                                }
                            }
                        }
                    } else {
                        // --- compact lane=dim-pair path (small H) ---
                        const int nw16=D/16;
                        for(int j=a.start;j<a.end;++j) {
                            size_t pos=bs?size_t(table[size_t(a.b)*blocks_per_batch+j/bs])*bs+j%bs:size_t(a.b)*S2+j;
                            uint16_t* kd=work.kdup.data()+size_t(j-a.start)*D;
                            const uint16_t* kp=k+size_t(pos)*D;
                            for(int d=0;d<D;++d)kd[d]=kp[d];
                        }
                        for(int i=a.q0;i<a.q1;++i) {
                            const uint16_t* qr=q+(size_t(a.b)*S1+i)*H*D;
                            uint16_t* qb=work.qbf.data()+size_t(i-a.q0)*H*nw16*16;
                            for(int h=0;h<H;++h)
                                for(int w=0;w<nw16;++w)
                                    for(int L=0;L<8;++L){
                                        qb[(size_t(h)*nw16+w)*16+2*L]=qr[size_t(h)*D+16*w+2*L];
                                        qb[(size_t(h)*nw16+w)*16+2*L+1]=qr[size_t(h)*D+16*w+2*L+1];
                                    }
                        }
                        for(int j=a.start;j<a.end;++j) {
                            const uint16_t* kd=work.kdup.data()+size_t(j-a.start)*D;
                            for(int i=a.q0;i<a.q1;++i) {
                                if(mode==3&&j>=lk[a.b]-lq[a.b]+i+1)continue;
                                const size_t row=size_t(a.b)*S1+i;
                                float value=score_bf16_compact(work.qbf.data()+size_t(i-a.q0)*H*nw16*16,
                                                               kd,x.W.data(),H,D,row);
                                if(!std::isfinite(value)) {
                                    #pragma omp atomic write
                                    status=-6;
                                }
                                scores[row*S2+j]=value;
                            }
                        }
                    }
                    continue;
                }
#endif
                for(int j=a.start;j<a.end;++j) {
                    size_t pos=bs?size_t(table[size_t(a.b)*blocks_per_batch+j/bs])*bs+j%bs:size_t(a.b)*S2+j;
                    for(int d=0;d<D;++d)work.kv[d]=decode(k[pos*D+d],bf);
                    // Key is reused over query tile and four heads; no shared score writes.
                    for(int i=a.q0;i<a.q1;++i) {
                        if(mode==3&&j>=lk[a.b]-lq[a.b]+i+1)continue;
                        size_t row=size_t(a.b)*S1+i;float score=0.0f;int h=0;
                        for(;h+3<H;h+=4) {
                            const float* p=x.Q.data()+(row*H+h)*D;
                            float s0=0,s1=0,s2=0,s3=0;
                            #pragma omp simd reduction(+:s0,s1,s2,s3)
                            for(int d=0;d<D;++d){float v=work.kv[d];s0+=p[d]*v;s1+=p[D+d]*v;s2+=p[2*D+d]*v;s3+=p[3*D+d]*v;}
                            score+=x.W[row*H+h]*std::max(s0,0.f);
                            score+=x.W[row*H+h+1]*std::max(s1,0.f);
                            score+=x.W[row*H+h+2]*std::max(s2,0.f);
                            score+=x.W[row*H+h+3]*std::max(s3,0.f);
                        }
                        for(;h<H;++h) {
                            const float* p=x.Q.data()+(row*H+h)*D;float dot=0;
                            #pragma omp simd reduction(+:dot)
                            for(int d=0;d<D;++d)dot+=p[d]*work.kv[d];
                            score+=x.W[row*H+h]*std::max(dot,0.f);
                        }
                        if(!std::isfinite(score)) {
                            #pragma omp atomic write
                            status=-6;
                        }
                        scores[row*S2+j]=score;
                    }
                }
            }
            #pragma omp single
            t2=Clock::now();
            if(!status) {
            if(G==nt) {
                // Enough independent groups: process rows without global round barriers.
                #pragma omp for schedule(dynamic,1)
                for(size_t r=0;r<x.rows.size();++r)select_row(work,x.rows[r],x,scores,out,sorted,hist);
            } else {
            // Balanced disjoint subteams. All subteams advance in lockstep rounds.
            // At nt>=rows: one group per row. At rows>=nt: one worker per group.
            const int g=int((int64_t(tid+1)*G-1)/nt);
            const int first=int(int64_t(g)*nt/G),last=int(int64_t(g+1)*nt/G);
            const int local=tid-first,count=last-first;
            auto& shared=x.groups[g];
            const size_t waves=(x.rows.size()+G-1)/G;
            for(size_t wave=0;wave<waves;++wave) {
                const size_t r=wave*G+g;const bool active=r<x.rows.size();
                const int v=active?x.rows[r].v:0;
                const size_t row=active?x.rows[r].index:0;
                const int begin=int(int64_t(v)*local/count),end=int(int64_t(v)*(local+1)/count);
                size_t n=0;
                // v<=K requires no selection, only optional final sorting.
                if(v>K) for(int j=begin;j<end;++j)work.keys[n++]=float_to_sortable_key(scores[row*S2+j]);
                if(!local){shared.prefix=0;shared.remain=std::min(v,K);}
                #pragma omp barrier
                for(int byte=3;byte>=0;--byte) {
                    if(v>K)histogram(work,n,byte,hist);
                    #pragma omp barrier
                    if(!local&&v>K) {
                        uint32_t sum[256]={};
                        for(int t=first;t<last;++t)for(int z=0;z<256;++z)sum[z]+=x.workers[t].hist[z];
                        int rank=shared.remain;
                        for(int z=255;z>=0;--z){if(rank>int(sum[z]))rank-=sum[z];else{shared.target=z;break;}}
                        shared.remain=rank;shared.prefix|=uint32_t(shared.target)<<(byte*8);
                    }
                    #pragma omp barrier
                    if(v>K&&byte) {
                        size_t next=0;for(size_t z=0;z<n;++z)if(extract_byte(work.keys[z],byte)==shared.target)work.keys[next++]=work.keys[z];n=next;
                    }
                    // Next round's pre-reduction barrier protects reads of shared.target.
                }
                work.gt=work.eq=0;
                if(v>K) for(int j=begin;j<end;++j){auto key=float_to_sortable_key(scores[row*S2+j]);work.gt+=key>shared.prefix;work.eq+=key==shared.prefix;}
                #pragma omp barrier
                if(!local&&v>K) {
                    size_t gt=0,eq=0;for(int t=first;t<last;++t){auto& a=x.workers[t];a.offg=gt;a.offe=eq;gt+=a.gt;eq+=a.eq;}
                    shared.ng=gt;shared.need=K-int(gt);
                }
                #pragma omp barrier
                if(v>K) {
                    size_t gt=work.offg,eq=work.offe;
                    for(int j=begin;j<end;++j){auto key=float_to_sortable_key(scores[row*S2+j]);
                        if(key>shared.prefix)out[row*K+gt++]=j;
                        else if(key==shared.prefix){if(eq<size_t(shared.need))out[row*K+shared.ng+eq]=j;++eq;}}
                } else for(int j=begin;j<end;++j)out[row*K+j]=j;
                #pragma omp barrier
                if(!local&&active&&sorted) {
                    std::sort(out+row*K,out+row*K+std::min(v,K),[&](int a,int b){
                        float va=scores[row*S2+a],vb=scores[row*S2+b];return va!=vb?va>vb:a<b;});
                }
                #pragma omp barrier
            }
            }
            }
            #pragma omp single
            t3=Clock::now();
        }
    }
    if(status)return status;
    if(nt!=P)return -5;
    timing[0]=elapsed(t0,t1);timing[1]=elapsed(t1,t2);timing[2]=elapsed(t2,t3);timing[3]=elapsed(t0,Clock::now());
    return 0;
}
}
