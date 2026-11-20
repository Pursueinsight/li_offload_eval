#pragma once
#include <cstdint>
#include <cstring>
#include <algorithm>
#if defined(__ARM_FEATURE_SVE2)
#include <arm_sve.h>
#define HAVE_SVE2 1
#else
#define HAVE_SVE2 0
#endif
static inline uint32_t float_to_sortable_key(float score) {
    // Treat -0 and +0 as the same score for stable index tie breaking.
    if (score == 0.0f) score = 0.0f;
    uint32_t bits; std::memcpy(&bits, &score, 4);
    return bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
}
static inline uint8_t extract_byte(uint32_t x, int byte) { return x >> (8*byte); }
static void histogram_scalar(const uint32_t* keys, size_t n, int byte, uint32_t* h) {
    std::fill(h,h+256,0u);
    for (size_t i=0;i<n;++i) ++h[extract_byte(keys[i],byte)];
}
#if HAVE_SVE2
// VL=256 only. Scratch capacity >= n. No allocation, no shared writes.
static void histogram_sve2(const uint32_t* keys, size_t n, int byte,
                           uint32_t* h, uint8_t* scratch) {
    if(!n){std::fill(h,h+256,0u);return;}
    for(size_t i=0;i<n;++i) scratch[i]=extract_byte(keys[i],byte);
    const auto pg=svptrue_b32();
    const auto lanes=svand_n_u8_x(svptrue_b8(),svindex_u8(0,1),15);
    // Loop over bin groups outside data chunks: keep widened counters in registers.
    for(int base=0;base<256;base+=16) {
        auto q=svadd_n_u8_x(svptrue_b8(),lanes,base);
        auto low=svdup_u32(0), high=svdup_u32(0);
        for(size_t offset=0;offset<n;offset+=32) {
            auto p=svwhilelt_b8(uint64_t(offset),uint64_t(n));
            auto counts=svhistseg_u8(q,svld1_u8(p,scratch+offset));
            auto a=svunpklo_u16(counts), b=svunpkhi_u16(counts);
            low=svadd_u32_x(pg,low,svadd_u32_x(pg,svunpklo_u32(a),svunpklo_u32(b)));
            high=svadd_u32_x(pg,high,svadd_u32_x(pg,svunpkhi_u32(a),svunpkhi_u32(b)));
        }
        svst1_u32(pg,h+base,low);svst1_u32(pg,h+base+8,high);
    }
    h[0]-=(32-n%32)%32; // predicate-zero tail contributed padding to bin zero
}
#endif
