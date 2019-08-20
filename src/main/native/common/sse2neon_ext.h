#ifndef SSE2NEON_EXT_H
#define SSE2NEON_EXT_H

#define ENABLE_CPP_VERSION 0

#if defined(__GNUC__) || defined(__clang__)
#	pragma push_macro("FORCE_INLINE")
#	pragma push_macro("ALIGN_STRUCT")
#	define FORCE_INLINE       static inline __attribute__((always_inline))
#	define ALIGN_STRUCT(x)    __attribute__((aligned(x)))
#else
#	error "Macro name collisions may happens with unknown compiler"
#	define FORCE_INLINE       static inline
#	define ALIGN_STRUCT(x)    __declspec(align(x))
#endif

#include <arm_neon.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>


typedef float32x4_t __m128;
typedef int8x8_t __m64;
typedef int32x4_t __m128i;

typedef int32x4x2_t __m256i;
typedef float64x2_t __m128d;
typedef float32x4x2_t __m256;
typedef float64x2x2_t __m256d;
typedef int64_t __int64;

typedef float32x4x4_t __m512;
typedef int32x4x4_t __m512i;
typedef float64x2x4_t __m512d;

typedef int8_t __mmask8;
typedef int16_t __mmask16;
typedef int32_t __int32;



//512 pairhmm ////////////////////////////////////////////////////////////////////////////////////////////////////////
// Add packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst.
FORCE_INLINE __m512d _mm512_add_pd(__m512d a, __m512d b)
{
    float64x2x4_t res;
    
    //float64x2_t vaddq_f64 (float64x2_t a, float64x2_t b)
    res.val[0] = vaddq_f64((float64x2_t)a.val[0], (float64x2_t)b.val[0]);
    res.val[1] = vaddq_f64((float64x2_t)a.val[1], (float64x2_t)b.val[1]);

    res.val[2] = vaddq_f64((float64x2_t)a.val[2], (float64x2_t)b.val[2]);
    res.val[3] = vaddq_f64((float64x2_t)a.val[3], (float64x2_t)b.val[3]);
    
    /*(res.val[0])[0] = (a.val[0])[0]+(b.val[0])[0];
    (res.val[0])[1] = (a.val[0])[1]+(b.val[0])[1];
    (res.val[1])[0] = (a.val[1])[0]+(b.val[1])[0];
    (res.val[1])[1] = (a.val[1])[1]+(b.val[1])[1];*/

    return (__m512d)res;
}


//Broadcast double-precision (64-bit) floating-point value a to all elements of dst.
//test--performance compare
FORCE_INLINE __m512d _mm512_set1_pd(double a)
{
    //return _mm256_set_pd(a, a, a, a);
    
    /*float64_t  ptr[] = {a, a};
     
    //float64x2x2_t vld2q_dup_f64 (float64_t const * ptr)
    return (__m256d)vld2q_dup_f64(ptr);*/
    
    float64x2x4_t res;

    (res.val[0])[0] = a;
    (res.val[0])[1] = a;
    (res.val[1])[0] = a;
    (res.val[1])[1] = a;

    (res.val[2])[0] = a;
    (res.val[2])[1] = a;
    (res.val[3])[0] = a;
    (res.val[3])[1] = a;

    return (__m512d)res;    
}


// Convert packed 32-bit integers in a to packed double-precision (64-bit) floating-point elements, and store the results in dst.
// test  Convert_Int32_To_FP64
FORCE_INLINE __m512d _mm512_cvtepi32_pd(__m256i a)
{
    float64x2x4_t res;
    
    (res.val[0])[0] = (float64_t)(int64_t)(((int32x4_t)a.val[0])[0]);
    (res.val[0])[1] = (float64_t)(int64_t)(((int32x4_t)a.val[0])[1]); 
    (res.val[1])[0] = (float64_t)(int64_t)(((int32x4_t)a.val[0])[2]);
    (res.val[1])[1] = (float64_t)(int64_t)(((int32x4_t)a.val[0])[3]);

    (res.val[2])[0] = (float64_t)(int64_t)(((int32x4_t)a.val[1])[0]);
    (res.val[2])[1] = (float64_t)(int64_t)(((int32x4_t)a.val[1])[1]);
    (res.val[3])[0] = (float64_t)(int64_t)(((int32x4_t)a.val[1])[2]);
    (res.val[3])[1] = (float64_t)(int64_t)(((int32x4_t)a.val[1])[3]);
    
    
    return (__m512d)res;
    
    /*float64x1_t tmp[4];
    
    //int32_t vgetq_lane_s32 (int32x4_t v, const int lane)
    tmp[0] = (float64x1_t)_mm_set_pi32(0, vgetq_lane_s32((int32x4_t)a , 0));
    tmp[1] = (float64x1_t)_mm_set_pi32(0, vgetq_lane_s32((int32x4_t)a , 1));
    tmp[2] = (float64x1_t)_mm_set_pi32(0, vgetq_lane_s32((int32x4_t)a , 2));
    tmp[3] = (float64x1_t)_mm_set_pi32(0, vgetq_lane_s32((int32x4_t)a , 3));
    
    return _mm256_set_pd(tmp[3], tmp[2], tmp[1], tmp[0]);*/
}


// Add packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
//test--overflow
FORCE_INLINE __m512 _mm512_add_ps (__m512 a, __m512 b) 
{
    float32x4x4_t res;
    
    //float32x4_t vaddq_f32 (float32x4_t a, float32x4_t b)
    res.val[0] = vaddq_f32((float32x4_t)a.val[0], (float32x4_t)b.val[0]);
    res.val[1] = vaddq_f32((float32x4_t)a.val[1], (float32x4_t)b.val[1]);

    res.val[2] = vaddq_f32((float32x4_t)a.val[2], (float32x4_t)b.val[2]);
    res.val[3] = vaddq_f32((float32x4_t)a.val[3], (float32x4_t)b.val[3]);

    return (__m512)res;
}

//Broadcast single-precision (32-bit) floating-point value a to all elements of dst.
FORCE_INLINE __m512 _mm512_set1_ps(float a)
{
    float  ptr[] = {a, a, a, a};
     
    //float32x4x2_t vld2q_dup_f32 (float32_t const * ptr)
    return (__m512)vld4q_dup_f32(ptr);
}


// Convert packed 32-bit integers in a to packed single-precision (32-bit) floating-point elements, and store the results in dst.
FORCE_INLINE __m512 _mm512_cvtepi32_ps(__m512i a)
{
    float32x4x4_t res;
    
    (res.val[0])[0] = (float)((int32x4_t)a.val[0])[0];
    (res.val[0])[1] = (float)((int32x4_t)a.val[0])[1];
    (res.val[0])[2] = (float)((int32x4_t)a.val[0])[2];
    (res.val[0])[3] = (float)((int32x4_t)a.val[0])[3];
    
    (res.val[1])[0] = (float)((int32x4_t)a.val[1])[0];
    (res.val[1])[1] = (float)((int32x4_t)a.val[1])[1];
    (res.val[1])[2] = (float)((int32x4_t)a.val[1])[2];
    (res.val[1])[3] = (float)((int32x4_t)a.val[1])[3];

    (res.val[2])[0] = (float)((int32x4_t)a.val[2])[0];
    (res.val[2])[1] = (float)((int32x4_t)a.val[2])[1];
    (res.val[2])[2] = (float)((int32x4_t)a.val[2])[2];
    (res.val[2])[3] = (float)((int32x4_t)a.val[2])[3];

    (res.val[3])[0] = (float)((int32x4_t)a.val[3])[0];
    (res.val[3])[1] = (float)((int32x4_t)a.val[3])[1];
    (res.val[3])[2] = (float)((int32x4_t)a.val[3])[2];
    (res.val[3])[3] = (float)((int32x4_t)a.val[3])[3];
    
    return (__m512)res;
        
    /*float32x4x2_t res;
    
    res.val[0] = (float32x4_t)(int32x4_t)a.val[0];
    res.val[1] = (float32x4_t)(int32x4_t)a.val[1];
    
    return (__m256)res;*/
    
    /*float tmp[8];
    float32x4x2_t res;
    
    (res.val[0])[0] = (float)vgetq_lane_s32((int32x4_t)a.val[0] , 0)
    
    
    
    //int32_t vgetq_lane_s32 (int32x4_t v, const int lane)
    tmp[0] = (float64x1_t)_mm_set_pi32(0, vgetq_lane_s32((int32x4_t)a , 0));
    tmp[1] = (float64x1_t)_mm_set_pi32(0, vgetq_lane_s32((int32x4_t)a , 1));
    tmp[2] = (float64x1_t)_mm_set_pi32(0, vgetq_lane_s32((int32x4_t)a , 2));
    tmp[3] = (float64x1_t)_mm_set_pi32(0, vgetq_lane_s32((int32x4_t)a , 3));
    
    return _mm256_set_pd(tmp[3], tmp[2], tmp[1], tmp[0]);*/
}

//Broadcast 32-bit integer a to all elements of dst.
FORCE_INLINE __m512i _mm512_set1_epi32(int a)
{
    int32x4x4_t res;

    (res.val[0])[0] = a;
    (res.val[0])[1] = a;
    (res.val[0])[2] = a;
    (res.val[0])[3] = a;

    (res.val[1])[0] = a;
    (res.val[1])[1] = a;
    (res.val[1])[2] = a;
    (res.val[1])[3] = a;

    (res.val[2])[0] = a;
    (res.val[2])[1] = a;
    (res.val[2])[2] = a;
    (res.val[2])[3] = a;

    (res.val[3])[0] = a;
    (res.val[3])[1] = a;
    (res.val[3])[2] = a;
    (res.val[3])[3] = a;

    return (__m512i)res;
}

// Set packed double-precision (64-bit) floating-point elements in dst with the supplied values.
FORCE_INLINE __m512d _mm512_set_pd(double e7, double e6, double e5, double e4, double e3, double e2, double e1, double e0)
{
    /*double ptr[] = {e0, e2, e1, e3};
    
    //float64x2x2_t vld2q_f64 (float64_t const * ptr)
    return (__m256d)vld2q_f64(ptr);*/

    float64x2x4_t res;

    (res.val[0])[0] = e0;
    (res.val[0])[1] = e1;
    (res.val[1])[0] = e2;
    (res.val[1])[1] = e3;

    (res.val[2])[0] = e4;
    (res.val[2])[1] = e5;
    (res.val[3])[0] = e6;
    (res.val[3])[1] = e7;

    return (__m512d)res;
}

// Divide packed double-precision (64-bit) floating-point elements in a by packed elements in b, and store the results in dst.
// test--divide 0
FORCE_INLINE __m512d _mm512_div_pd(__m512d a, __m512d b)
{
    float64x2x4_t res;

    // float64x2_t vdivq_f64 (float64x2_t a, float64x2_t b)
    res.val[0] = vdivq_f64((float64x2_t)a.val[0], (float64x2_t)b.val[0]);
    res.val[1] = vdivq_f64((float64x2_t)a.val[1], (float64x2_t)b.val[1]);

    res.val[2] = vdivq_f64((float64x2_t)a.val[2], (float64x2_t)b.val[2]);
    res.val[3] = vdivq_f64((float64x2_t)a.val[3], (float64x2_t)b.val[3]);

    return (__m512d)res;
}

// Subtract packed double-precision (64-bit) floating-point elements in b from packed double-precision (64-bit) floating-point elements in a, and store the results in dst. 
FORCE_INLINE __m512d _mm512_sub_pd(__m512d a, __m512d b)
{
    float64x2x4_t res;
    
    //float64x2_t vsubq_f64 (float64x2_t a, float64x2_t b)
    res.val[0] = vsubq_f64((float64x2_t)a.val[0], (float64x2_t)b.val[0]);
    res.val[1] = vsubq_f64((float64x2_t)a.val[1], (float64x2_t)b.val[1]);

    res.val[2] = vsubq_f64((float64x2_t)a.val[2], (float64x2_t)b.val[2]);
    res.val[3] = vsubq_f64((float64x2_t)a.val[3], (float64x2_t)b.val[3]);
    
    return (__m512d)res;
}

//Set packed single-precision (32-bit) floating-point elements in dst with the supplied values.
FORCE_INLINE __m512 _mm512_set_ps(float e15, float e14, float e13, float e12, float e11, float e10, float e9, float e8, float e7, float e6, float e5, float e4, float e3, float e2, float e1, float e0)
{
    float ptr[] = {e0, e4, e8, e12, e1, e5, e9, e13, e2, e6, e10, e14, e3, e7, e11, e15};
    
    return (__m512)vld4q_f32(ptr);
}

// Divide packed single-precision (32-bit) floating-point elements in a by packed elements in b, and store the results in dst.
// test x/0 overflow
FORCE_INLINE __m512 _mm512_div_ps(__m512 a, __m512 b)
{
    float32x4x4_t res;
    
    // float32x4_t vdivq_f32 (float32x4_t a, float32x4_t b)
    res.val[0] = vdivq_f32((float32x4_t)a.val[0], (float32x4_t)b.val[0]);
    res.val[1] = vdivq_f32((float32x4_t)a.val[1], (float32x4_t)b.val[1]);

    res.val[2] = vdivq_f32((float32x4_t)a.val[2], (float32x4_t)b.val[2]);
    res.val[3] = vdivq_f32((float32x4_t)a.val[3], (float32x4_t)b.val[3]);
    
    return (__m512)res;
}

// Subtract packed single-precision (32-bit) floating-point elements in b from packed single-precision (32-bit) floating-point elements in a, 
// and store the results in dst.
// test--underflow
FORCE_INLINE __m512 _mm512_sub_ps(__m512 a, __m512 b)
{
    float32x4x4_t res;
    
    // float32x4_t vsubq_f32 (float32x4_t a, float32x4_t b)
    res.val[0] = vsubq_f32((float32x4_t)a.val[0], (float32x4_t)b.val[0]);
    res.val[1] = vsubq_f32((float32x4_t)a.val[1], (float32x4_t)b.val[1]);

    res.val[2] = vsubq_f32((float32x4_t)a.val[2], (float32x4_t)b.val[2]);
    res.val[3] = vsubq_f32((float32x4_t)a.val[3], (float32x4_t)b.val[3]);
    
    return (__m512)res;
}


// Multiply packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst.
// test--overflow
FORCE_INLINE __m512d _mm512_mul_pd(__m512d a, __m512d b)
{
    float64x2x4_t res;
    
    //float64x2_t vmulq_f64 (float64x2_t a, float64x2_t b)
    res.val[0] = vmulq_f64((float64x2_t)a.val[0], (float64x2_t)b.val[0]);
    res.val[1] = vmulq_f64((float64x2_t)a.val[1], (float64x2_t)b.val[1]);

    res.val[2] = vmulq_f64((float64x2_t)a.val[2], (float64x2_t)b.val[2]);
    res.val[3] = vmulq_f64((float64x2_t)a.val[3], (float64x2_t)b.val[3]);
    
    return (__m512d)res;
}

// Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
// test--overflow
FORCE_INLINE __m512 _mm512_mul_ps(__m512 a, __m512 b)
{
    float32x4x4_t res;
    
    // float32x4_t vmulq_f32 (float32x4_t a, float32x4_t b)
    res.val[0] = vmulq_f32((float32x4_t)a.val[0], (float32x4_t)b.val[0]);
    res.val[1] = vmulq_f32((float32x4_t)a.val[1], (float32x4_t)b.val[1]);

    res.val[2] = vmulq_f32((float32x4_t)a.val[2], (float32x4_t)b.val[2]);
    res.val[3] = vmulq_f32((float32x4_t)a.val[3], (float32x4_t)b.val[3]);
    
    return (__m512)res;
}

// Compute the bitwise OR of packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
FORCE_INLINE __m512i _mm512_or_epi32(__m512i a, __m512i b)
{
    int32x4x4_t res;
    
    res.val[0] = vorrq_s32((int32x4_t)a.val[0], (int32x4_t)b.val[0]);
    res.val[1] = vorrq_s32((int32x4_t)a.val[1], (int32x4_t)b.val[1]);

    res.val[2] = vorrq_s32((int32x4_t)a.val[2], (int32x4_t)b.val[2]);
    res.val[3] = vorrq_s32((int32x4_t)a.val[3], (int32x4_t)b.val[3]);
    
    return (__m512i)res;
    
    /*
    int32x4x2_t res;
    
    res.val[0] = vorrq_s32((int32x4_t)a.val[0], (int32x4_t)b.val[0]);
    res.val[1] = vorrq_s32((int32x4_t)a.val[1], (int32x4_t)b.val[1]);
    
    return (__m256)res;
    */
}

// Compute the bitwise AND of packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
FORCE_INLINE __m512i _mm512_and_epi32(__m512i a, __m512i b)
{
    int32x4x4_t res;
    
    res.val[0] = vandq_s32((int32x4_t)a.val[0], (int32x4_t)b.val[0]);
    res.val[1] = vandq_s32((int32x4_t)a.val[1], (int32x4_t)b.val[1]);

    res.val[2] = vandq_s32((int32x4_t)a.val[2], (int32x4_t)b.val[2]);
    res.val[3] = vandq_s32((int32x4_t)a.val[3], (int32x4_t)b.val[3]);
    
    return (__m512i)res;
}

//Set packed 32-bit integers in dst with the supplied values.
FORCE_INLINE __m512i _mm512_set_epi32(int e15, int e14, int e13, int e12, int e11, int e10, int e9, int e8, int e7, int e6, int e5, int e4, int e3, int e2, int e1, int e0)
{
    int ptr[] = {e0, e4, e8, e12, e1, e5, e9, e13, e2, e6, e10, e14, e3, e7, e11, e15};
    
    return (__m512i)vld4q_s32(ptr);
}

//Return vector of type __m512d with all elements set to zero.
FORCE_INLINE __m512d _mm512_setzero_pd(void)
{
    /*float64_t  ptr[] = {0, 0};
     
    //float64x2x2_t vld2q_dup_f64 (float64_t const * ptr)
    return (__m256d)vld2q_dup_f64(ptr); 
    */
    float64x2x4_t res;

    (res.val[0])[0] = 0;
    (res.val[0])[1] = 0;
    (res.val[1])[0] = 0;
    (res.val[1])[1] = 0;

    (res.val[2])[0] = 0;
    (res.val[2])[1] = 0;
    (res.val[3])[0] = 0;
    (res.val[3])[1] = 0;

    return (__m512d)res;    
}

//Return vector of type __m512 with all elements set to zero.
FORCE_INLINE __m512 _mm512_setzero_ps(void)
{
    /*float  ptr[] = {0, 0, 0, 0};
     
    //float32x4x2_t vld2q_dup_f32 (float32_t const * ptr)
    return (__m256)vld2q_dup_f32(ptr);*/
     
    float32x4x4_t res;
    (res.val[0])[0] = 0;
    (res.val[0])[1] = 0;
    (res.val[0])[2] = 0;
    (res.val[0])[3] = 0;
    (res.val[1])[0] = 0;
    (res.val[1])[1] = 0;
    (res.val[1])[2] = 0;
    (res.val[1])[3] = 0;

    (res.val[2])[0] = 0;
    (res.val[2])[1] = 0;
    (res.val[2])[2] = 0;
    (res.val[2])[3] = 0;
    (res.val[3])[0] = 0;
    (res.val[3])[1] = 0;
    (res.val[3])[2] = 0;
    (res.val[3])[3] = 0;

    return (__m512)res; 
}



//new creat ###############################################################################################
//Broadcast 64-bit integer a to all elements of dst.
FORCE_INLINE __m512i _mm512_set1_epi64(__int64 a)
{
    int32x4x4_t res;

    res.val[0] = (int32x4_t)vdupq_n_s64((int64_t)a);
    res.val[1] = (int32x4_t)vdupq_n_s64((int64_t)a);
    res.val[2] = (int32x4_t)vdupq_n_s64((int64_t)a);
    res.val[3] = (int32x4_t)vdupq_n_s64((int64_t)a);

    return (__m512i)res;
}
 
 //Compute the bitwise OR of packed 64-bit integers in a and b, and store the resut in dst.
FORCE_INLINE __m512i _mm512_or_epi64(__m512i a, __m512i b)
{
    int32x4x4_t res;

    res.val[0] = (int32x4_t)vorrq_s64((int64x2_t)a.val[0], (int64x2_t)b.val[0]);
    res.val[1] = (int32x4_t)vorrq_s64((int64x2_t)a.val[1], (int64x2_t)b.val[1]);

    res.val[2] = (int32x4_t)vorrq_s64((int64x2_t)a.val[2], (int64x2_t)b.val[2]);
    res.val[3] = (int32x4_t)vorrq_s64((int64x2_t)a.val[3], (int64x2_t)b.val[3]);

    return (__m512i)res;
}

//Compute the bitwise AND of 512 bits (composed of packed 64-bit integers) in a and b, and store the results in dst.
FORCE_INLINE __m512i _mm512_and_epi64(__m512i a, __m512i b)
{
    int32x4x4_t res;

    res.val[0] = (int32x4_t)vandq_s64((int64x2_t)a.val[0], (int64x2_t)b.val[0]);
    res.val[1] = (int32x4_t)vandq_s64((int64x2_t)a.val[1], (int64x2_t)b.val[1]);

    res.val[2] = (int32x4_t)vandq_s64((int64x2_t)a.val[2], (int64x2_t)b.val[2]);
    res.val[3] = (int32x4_t)vandq_s64((int64x2_t)a.val[3], (int64x2_t)b.val[3]);

    return (__m512i)res;
}

//Set packed 64-bit integers in dst with the supplied values.
FORCE_INLINE __m512i _mm512_set_epi64(__int64 e7, __int64 e6, __int64 e5, __int64 e4, __int64 e3, __int64 e2, __int64 e1, __int64 e0)
{
    int32x4x4_t res;

    res.val[0] = (int32x4_t)vcombine_s64((int64x1_t)e0, (int64x1_t)e1);
    res.val[1] = (int32x4_t)vcombine_s64((int64x1_t)e2, (int64x1_t)e3);
    res.val[2] = (int32x4_t)vcombine_s64((int64x1_t)e4, (int64x1_t)e5);
    res.val[3] = (int32x4_t)vcombine_s64((int64x1_t)e6, (int64x1_t)e7);

    return (__m512i)res;
   
}

//  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
//Cast vector of type __m512d to type __m128d. This intrinsic is only used for compilation and does not generate any instructions, thus it has zero latency.
FORCE_INLINE __m128d _mm512_castpd512_pd128(__m512d a)
{
    float64x2_t res;

    res = (float64x2_t)a.val[0];

    return (__m128d)res;
}

//Cast vector of type __m512 to type __m128. This intrinsic is only used for compilation and does not generate any instructions, thus it has zero latency.
FORCE_INLINE __m128 _mm512_castps512_ps128(__m512 a)
{
    float32x4_t res;

    res = (float32x4_t)a.val[0];

    return (__m128)res;
}

//Cast vector of type __m128d to type __m512d; the upper 384 bits of the result are undefined. This intrinsic is only used for compilation and does not generate any instructions, thus it has zero latency.
FORCE_INLINE __m512d _mm512_castpd128_pd512(__m128d a)
{
    float64x2x4_t res;

    res.val[0] = (float64x2_t)a;

    return (__m512d)res;
}

//Cast vector of type __m128 to type __m512; the upper 384 bits of the result are undefined. This intrinsic is only used for compilation and does not generate any instructions, thus it has zero latency.
FORCE_INLINE __m512 _mm512_castps128_ps512(__m128 a)
{
    float32x4x4_t res;

    res.val[0] = (float32x4_t)a;

    return (__m512)res;
}

/*//Compute the bitwise XOR of packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst.
FORCE_INLINE __m512d _mm512_xor_pd(__m512d a, __m512d b)
{
    float64x2x4_t res;
    int64x2x4_t resint,aint,bint;

    aint.val[0] = vreinterpretq_s64_f64(a.val[0]);
    aint.val[1] = vreinterpretq_s64_f64(a.val[1]);
    aint.val[2] = vreinterpretq_s64_f64(a.val[2]);
    aint.val[3] = vreinterpretq_s64_f64(a.val[3]);

    bint.val[0] = vreinterpretq_s64_f64(b.val[0]);
    bint.val[1] = vreinterpretq_s64_f64(b.val[1]);
    bint.val[2] = vreinterpretq_s64_f64(b.val[2]);
    bint.val[3] = vreinterpretq_s64_f64(b.val[3]);

    resint.val[0] = veorq_s64((int64x2_t)aint.val[0], (int64x2_t)bint.val[0]);
    resint.val[1] = veorq_s64((int64x2_t)aint.val[1], (int64x2_t)bint.val[1]);

    resint.val[2] = veorq_s64((int64x2_t)aint.val[2], (int64x2_t)bint.val[2]);
    resint.val[3] = veorq_s64((int64x2_t)aint.val[3], (int64x2_t)bint.val[3]);

    res.val[0] = vreinterpretq_f64_s64(resint.val[0]);
    res.val[1] = vreinterpretq_f64_s64(resint.val[1]);
    res.val[2] = vreinterpretq_f64_s64(resint.val[2]);
    res.val[3] = vreinterpretq_f64_s64(resint.val[3]);

    return (__m512d)res;
}

//Compute the bitwise XOR of packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst.
FORCE_INLINE __m512 _mm512_xor_ps(__m512 a, __m512 b)
{
    float32x4x4_t res;
    int32x4x4_t resint,aint,bint;

    aint.val[0] = vreinterpretq_s32_f32(a.val[0]);
    aint.val[1] = vreinterpretq_s32_f32(a.val[1]);
    aint.val[2] = vreinterpretq_s32_f32(a.val[2]);
    aint.val[3] = vreinterpretq_s32_f32(a.val[3]);

    bint.val[0] = vreinterpretq_s32_f32(b.val[0]);
    bint.val[1] = vreinterpretq_s32_f32(b.val[1]);
    bint.val[2] = vreinterpretq_s32_f32(b.val[2]);
    bint.val[3] = vreinterpretq_s32_f32(b.val[3]);

    resint.val[0] = veorq_s32((int32x4_t)aint.val[0], (int32x4_t)bint.val[0]);
    resint.val[1] = veorq_s32((int32x4_t)aint.val[1], (int32x4_t)bint.val[1]);

    resint.val[2] = veorq_s32((int32x4_t)aint.val[2], (int32x4_t)bint.val[2]);
    resint.val[3] = veorq_s32((int32x4_t)aint.val[3], (int32x4_t)bint.val[3]);

    res.val[0] = vreinterpretq_f32_s32(resint.val[0]);
    res.val[1] = vreinterpretq_f32_s32(resint.val[1]);
    res.val[2] = vreinterpretq_f32_s32(resint.val[2]);
    res.val[3] = vreinterpretq_f32_s32(resint.val[3]);

    return (__m512)res;
}*/

//Compare packed double-precision (64-bit) floating-point elements in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
FORCE_INLINE __m512d _mm512_cmp_pd(__m512d a, __m512d b, const int imm8)
{
    float64x2x4_t res;
    
    if(imm8 > 31 || imm8 < 0){
                printf("%s:%d:%s:error: the last argument must be a 5-bit immediate\n", __FILE__, __LINE__, __FUNCTION__);
                exit(1);
        }
    
    switch(imm8)
    {
        case CMP_EQ_OQ:
        case CMP_EQ_OS:
            res.val[0] = (float64x2_t)vceqq_f64((float64x2_t)a.val[0], (float64x2_t)b.val[0]);
            res.val[1] = (float64x2_t)vceqq_f64((float64x2_t)a.val[1], (float64x2_t)b.val[1]);
            res.val[2] = (float64x2_t)vceqq_f64((float64x2_t)a.val[2], (float64x2_t)b.val[2]);
            res.val[3] = (float64x2_t)vceqq_f64((float64x2_t)a.val[3], (float64x2_t)b.val[3]);
            break;
        default:{
            res = a;
            break;
        }   
    }
    
    return (__m512d)res;
}

//Compare packed double-precision (64-bit) floating-point elements in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
FORCE_INLINE __m512 _mm512_cmp_ps (__m512 a, __m512 b, const int imm8)
{
    float32x4x4_t res;
    
    if(imm8 > 31 || imm8 < 0){
                printf("%s:%d:%s:error: the last argument must be a 5-bit immediate\n", __FILE__, __LINE__, __FUNCTION__);
                exit(1);
        }
    
    switch(imm8)
    {
        case CMP_EQ_OQ:
        case CMP_EQ_OS:
            res.val[0] = (float32x4_t)vceqq_f32((float32x4_t)a.val[0], (float32x4_t)b.val[0]);
            res.val[1] = (float32x4_t)vceqq_f32((float32x4_t)a.val[1], (float32x4_t)b.val[1]);
            res.val[2] = (float32x4_t)vceqq_f32((float32x4_t)a.val[2], (float32x4_t)b.val[2]);
            res.val[3] = (float32x4_t)vceqq_f32((float32x4_t)a.val[3], (float32x4_t)b.val[3]);
            break;
        default:{
            res = a;
            break;
        }
    }
    
    return (__m512)res;
}



//Shuffle 64-bit integers in a across lanes using the corresponding index in idx, and store the results in dst.
FORCE_INLINE __m512i _mm512_permutexvar_epi64(__m512i idx, __m512i a)
{
    int32x4x4_t res;

    int64_t e0, e1, e2, e3, e4, e5, e6, e7;
    e0 = vgetq_lane_s64((int64x2_t)idx.val[0], 0);
    e1 = vgetq_lane_s64((int64x2_t)idx.val[0], 1);
    e2 = vgetq_lane_s64((int64x2_t)idx.val[1], 0);
    e3 = vgetq_lane_s64((int64x2_t)idx.val[1], 1);
    e4 = vgetq_lane_s64((int64x2_t)idx.val[2], 0);
    e5 = vgetq_lane_s64((int64x2_t)idx.val[2], 1);
    e6 = vgetq_lane_s64((int64x2_t)idx.val[3], 0);
    e7 = vgetq_lane_s64((int64x2_t)idx.val[3], 1);

    e0 = (int64_t)(e0 & 0x0000000000000007);
    e1 = (int64_t)(e1 & 0x0000000000000007);
    e2 = (int64_t)(e2 & 0x0000000000000007);
    e3 = (int64_t)(e3 & 0x0000000000000007);
    e4 = (int64_t)(e4 & 0x0000000000000007);
    e5 = (int64_t)(e5 & 0x0000000000000007);
    e6 = (int64_t)(e6 & 0x0000000000000007);
    e7 = (int64_t)(e7 & 0x0000000000000007);

    switch(e0)
    {
        case 0:{
            (res.val[0])[0] = (a.val[0])[0];
            (res.val[0])[1] = (a.val[0])[1];
            break;
        }
        case 1:{
            (res.val[0])[0] = (a.val[0])[2];
            (res.val[0])[1] = (a.val[0])[3];
            break;
        }
        case 2:{
            (res.val[0])[0] = (a.val[1])[0];
            (res.val[0])[1] = (a.val[1])[1];
            break;
        }
        case 3:{
            (res.val[0])[0] = (a.val[1])[2];
            (res.val[0])[1] = (a.val[1])[3];
            break;
        }
        case 4:{
            (res.val[0])[0] = (a.val[2])[0];
            (res.val[0])[1] = (a.val[2])[1];
            break;
        }
        case 5:{
            (res.val[0])[0] = (a.val[2])[2];
            (res.val[0])[1] = (a.val[2])[3];
            break;
        }
        case 6:{
            (res.val[0])[0] = (a.val[3])[0];
            (res.val[0])[1] = (a.val[3])[1];
            break;
        }
        case 7:{
            (res.val[0])[0] = (a.val[3])[2];
            (res.val[0])[1] = (a.val[3])[3];
            break;
        }
        default:{
            (res.val[0])[0] = (a.val[4])[0];
            (res.val[0])[1] = (a.val[4])[0];
            break;
        }
    }

    switch(e1)
    {
        case 0:{
            (res.val[0])[2] = (a.val[0])[0];
            (res.val[0])[3] = (a.val[0])[1];
            break;
        }
        case 1:{
            (res.val[0])[2] = (a.val[0])[2];
            (res.val[0])[3] = (a.val[0])[3];
            break;
        }
        case 2:{
            (res.val[0])[2] = (a.val[1])[0];
            (res.val[0])[3] = (a.val[1])[1];
            break;
        }
        case 3:{
            (res.val[0])[2] = (a.val[1])[2];
            (res.val[0])[3] = (a.val[1])[3];
            break;
        }
        case 4:{
            (res.val[0])[2] = (a.val[2])[0];
            (res.val[0])[3] = (a.val[2])[1];
            break;
        }
        case 5:{
            (res.val[0])[2] = (a.val[2])[2];
            (res.val[0])[3] = (a.val[2])[3];
            break;
        }
        case 6:{
            (res.val[0])[2] = (a.val[3])[0];
            (res.val[0])[3] = (a.val[3])[1];
            break;
        }
        case 7:{
            (res.val[0])[2] = (a.val[3])[2];
            (res.val[0])[3] = (a.val[3])[3];
            break;
        }
        default:{
            (res.val[0])[2] = (a.val[4])[0];
            (res.val[0])[3] = (a.val[4])[0];
            break;
        }
    }

    switch(e2)
    {
        case 0:{
            (res.val[1])[0] = (a.val[0])[0];
            (res.val[1])[1] = (a.val[0])[1];
            break;
        }
        case 1:{
            (res.val[1])[0] = (a.val[0])[2];
            (res.val[1])[1] = (a.val[0])[3];
            break;
        }
        case 2:{
            (res.val[1])[0] = (a.val[1])[0];
            (res.val[1])[1] = (a.val[1])[1];
            break;
        }
        case 3:{
            (res.val[1])[0] = (a.val[1])[2];
            (res.val[1])[1] = (a.val[1])[3];
            break;
        }
        case 4:{
            (res.val[1])[0] = (a.val[2])[0];
            (res.val[1])[1] = (a.val[2])[1];
            break;
        }
        case 5:{
            (res.val[1])[0] = (a.val[2])[2];
            (res.val[1])[1] = (a.val[2])[3];
            break;
        }
        case 6:{
            (res.val[1])[0] = (a.val[3])[0];
            (res.val[1])[1] = (a.val[3])[1];
            break;
        }
        case 7:{
            (res.val[1])[0] = (a.val[3])[2];
            (res.val[1])[1] = (a.val[3])[3];
            break;
        }
        default:{
            (res.val[1])[0] = (a.val[4])[0];
            (res.val[1])[1] = (a.val[4])[0];
            break;
        }
    }

    switch(e3)
    {
        case 0:{
            (res.val[1])[2] = (a.val[0])[0];
            (res.val[1])[3] = (a.val[0])[1];
            break;
        }
        case 1:{
            (res.val[1])[2] = (a.val[0])[2];
            (res.val[1])[3] = (a.val[0])[3];
            break;
        }
        case 2:{
            (res.val[1])[2] = (a.val[1])[0];
            (res.val[1])[3] = (a.val[1])[1];
            break;
        }
        case 3:{
            (res.val[1])[2] = (a.val[1])[2];
            (res.val[1])[3] = (a.val[1])[3];
            break;
        }
        case 4:{
            (res.val[1])[2] = (a.val[2])[0];
            (res.val[1])[3] = (a.val[2])[1];
            break;
        }
        case 5:{
            (res.val[1])[2] = (a.val[2])[2];
            (res.val[1])[3] = (a.val[2])[3];
            break;
        }
        case 6:{
            (res.val[1])[2] = (a.val[3])[0];
            (res.val[1])[3] = (a.val[3])[1];
            break;
        }
        case 7:{
            (res.val[1])[2] = (a.val[3])[2];
            (res.val[1])[3] = (a.val[3])[3];
            break;
        }
        default:{
            (res.val[1])[2] = (a.val[4])[0];
            (res.val[1])[3] = (a.val[4])[0];
            break;
        }
    }

    switch(e4)
    {
        case 0:{
            (res.val[2])[0] = (a.val[0])[0];
            (res.val[2])[1] = (a.val[0])[1];
            break;
        }
        case 1:{
            (res.val[2])[0] = (a.val[0])[2];
            (res.val[2])[1] = (a.val[0])[3];
            break;
        }
        case 2:{
            (res.val[2])[0] = (a.val[1])[0];
            (res.val[2])[1] = (a.val[1])[1];
            break;
        }
        case 3:{
            (res.val[2])[0] = (a.val[1])[2];
            (res.val[2])[1] = (a.val[1])[3];
            break;
        }
        case 4:{
            (res.val[2])[0] = (a.val[2])[0];
            (res.val[2])[1] = (a.val[2])[1];
            break;
        }
        case 5:{
            (res.val[2])[0] = (a.val[2])[2];
            (res.val[2])[1] = (a.val[2])[3];
            break;
        }
        case 6:{
            (res.val[2])[0] = (a.val[3])[0];
            (res.val[2])[1] = (a.val[3])[1];
            break;
        }
        case 7:{
            (res.val[2])[0] = (a.val[3])[2];
            (res.val[2])[1] = (a.val[3])[3];
            break;
        }
        default:{
            (res.val[2])[0] = (a.val[4])[0];
            (res.val[2])[1] = (a.val[4])[0];
            break;
        }
    }

    switch(e5)
    {
        case 0:{
            (res.val[2])[2] = (a.val[0])[0];
            (res.val[2])[3] = (a.val[0])[1];
            break;
        }
        case 1:{
            (res.val[2])[2] = (a.val[0])[2];
            (res.val[2])[3] = (a.val[0])[3];
            break;
        }
        case 2:{
            (res.val[2])[2] = (a.val[1])[0];
            (res.val[2])[3] = (a.val[1])[1];
            break;
        }
        case 3:{
            (res.val[2])[2] = (a.val[1])[2];
            (res.val[2])[3] = (a.val[1])[3];
            break;
        }
        case 4:{
            (res.val[2])[2] = (a.val[2])[0];
            (res.val[2])[3] = (a.val[2])[1];
            break;
        }
        case 5:{
            (res.val[2])[2] = (a.val[2])[2];
            (res.val[2])[3] = (a.val[2])[3];
            break;
        }
        case 6:{
            (res.val[2])[2] = (a.val[3])[0];
            (res.val[2])[3] = (a.val[3])[1];
            break;
        }
        case 7:{
            (res.val[2])[2] = (a.val[3])[2];
            (res.val[2])[3] = (a.val[3])[3];
            break;
        }
        default:{
            (res.val[2])[2] = (a.val[4])[0];
            (res.val[2])[3] = (a.val[4])[0];
            break;
        }
    }

    switch(e6)
    {
        case 0:{
            (res.val[3])[0] = (a.val[0])[0];
            (res.val[3])[1] = (a.val[0])[1];
            break;
        }
        case 1:{
            (res.val[3])[0] = (a.val[0])[2];
            (res.val[3])[1] = (a.val[0])[3];
            break;
        }
        case 2:{
            (res.val[3])[0] = (a.val[1])[0];
            (res.val[3])[1] = (a.val[1])[1];
            break;
        }
        case 3:{
            (res.val[3])[0] = (a.val[1])[2];
            (res.val[3])[1] = (a.val[1])[3];
            break;
        }
        case 4:{
            (res.val[3])[0] = (a.val[2])[0];
            (res.val[3])[1] = (a.val[2])[1];
            break;
        }
        case 5:{
            (res.val[3])[0] = (a.val[2])[2];
            (res.val[3])[1] = (a.val[2])[3];
            break;
        }
        case 6:{
            (res.val[3])[0] = (a.val[3])[0];
            (res.val[3])[1] = (a.val[3])[1];
            break;
        }
        case 7:{
            (res.val[3])[0] = (a.val[3])[2];
            (res.val[3])[1] = (a.val[3])[3];
            break;
        }
        default:{
            (res.val[3])[0] = (a.val[4])[0];
            (res.val[3])[1] = (a.val[4])[0];
            break;
        }
    }

    switch(e7)
    {
        case 0:{
            (res.val[3])[2] = (a.val[0])[0];
            (res.val[3])[3] = (a.val[0])[1];
            break;
        }
        case 1:{
            (res.val[3])[2] = (a.val[0])[2];
            (res.val[3])[3] = (a.val[0])[3];
            break;
        }
        case 2:{
            (res.val[3])[2] = (a.val[1])[0];
            (res.val[3])[3] = (a.val[1])[1];
            break;
        }
        case 3:{
            (res.val[3])[2] = (a.val[1])[2];
            (res.val[3])[3] = (a.val[1])[3];
            break;
        }
        case 4:{
            (res.val[3])[2] = (a.val[2])[0];
            (res.val[3])[3] = (a.val[2])[1];
            break;
        }
        case 5:{
            (res.val[3])[2] = (a.val[2])[2];
            (res.val[3])[3] = (a.val[2])[3];
            break;
        }
        case 6:{
            (res.val[3])[2] = (a.val[3])[0];
            (res.val[3])[3] = (a.val[3])[1];
            break;
        }
        case 7:{
            (res.val[3])[2] = (a.val[3])[2];
            (res.val[3])[3] = (a.val[3])[3];
            break;
        }
        default:{
            (res.val[3])[2] = (a.val[4])[0];
            (res.val[3])[3] = (a.val[4])[0];
            break;
        }
    }

    
    /*int64x2x4_t b;
    int64x2x4_t c;


    (b.val[0])[0] = 7;
    (b.val[0])[1] = 7;
    (b.val[1])[0] = 7;
    (b.val[1])[1] = 7;

    (b.val[2])[0] = 7;
    (b.val[2])[1] = 7;
    (b.val[3])[0] = 7;
    (b.val[3])[1] = 7;*/

    //res = a;
    /*c.val[0] = vandq_s64((int64x2_t)idx.val[0], int64x2_t)b.val[0]);
    c.val[1] = vandq_s64((int64x2_t)idx.val[1], int64x2_t)b.val[1]);
    c.val[2] = vandq_s64((int64x2_t)idx.val[2], int64x2_t)b.val[2]);
    c.val[3] = vandq_s64((int64x2_t)idx.val[3], int64x2_t)b.val[3]);

    res.val[0] = a.val[]*/

  
    return (__m512i)res;
}

//Shuffle 32-bit integers in a across lanes using the corresponding index in idx, and store the results in dst.
FORCE_INLINE __m512i _mm512_permutexvar_epi32(__m512i idx, __m512i a)
{
    int32x4x4_t res;

    int32_t e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15;

    e0 = vgetq_lane_s32((int32x4_t)idx.val[0], 0);
    e1 = vgetq_lane_s32((int32x4_t)idx.val[0], 1);
    e2 = vgetq_lane_s32((int32x4_t)idx.val[0], 2);
    e3 = vgetq_lane_s32((int32x4_t)idx.val[0], 3);

    e4 = vgetq_lane_s32((int32x4_t)idx.val[1], 0);
    e5 = vgetq_lane_s32((int32x4_t)idx.val[1], 1);
    e6 = vgetq_lane_s32((int32x4_t)idx.val[1], 2);
    e7 = vgetq_lane_s32((int32x4_t)idx.val[1], 3);

    e8 = vgetq_lane_s32((int32x4_t)idx.val[2], 0);
    e9 = vgetq_lane_s32((int32x4_t)idx.val[2], 1);
    e10 = vgetq_lane_s32((int32x4_t)idx.val[2], 2);
    e11 = vgetq_lane_s32((int32x4_t)idx.val[2], 3);

    e12 = vgetq_lane_s32((int32x4_t)idx.val[3], 0);
    e13 = vgetq_lane_s32((int32x4_t)idx.val[3], 1);
    e14 = vgetq_lane_s32((int32x4_t)idx.val[3], 2);
    e15 = vgetq_lane_s32((int32x4_t)idx.val[3], 3);

    e0 = (int32_t)(e0 & 0x0000000F);
    e1 = (int32_t)(e1 & 0x0000000F);
    e2 = (int32_t)(e2 & 0x0000000F);
    e3 = (int32_t)(e3 & 0x0000000F);
    e4 = (int32_t)(e4 & 0x0000000F);
    e5 = (int32_t)(e5 & 0x0000000F);
    e6 = (int32_t)(e6 & 0x0000000F);
    e7 = (int32_t)(e7 & 0x0000000F);

    e8 = (int32_t)(e8 & 0x0000000F);
    e9 = (int32_t)(e9 & 0x0000000F);
    e10 = (int32_t)(e10 & 0x0000000F);
    e11 = (int32_t)(e11 & 0x0000000F);
    e12 = (int32_t)(e12 & 0x0000000F);
    e13 = (int32_t)(e13 & 0x0000000F);
    e14 = (int32_t)(e14 & 0x0000000F);
    e15 = (int32_t)(e15 & 0x0000000F);

    switch(e0)
    {
        case 0: {
            (res.val[0])[0] = (a.val[0])[0];
            break;
        }
        case 1: {
            (res.val[0])[0] = (a.val[0])[1];
            break;
        }
        case 2: {
            (res.val[0])[0] = (a.val[0])[2];
            break;
        }
        case 3: {
            (res.val[0])[0] = (a.val[0])[3];
            break;
        }
        case 4: {
            (res.val[0])[0] = (a.val[1])[0];
            break;
        }
        case 5: {
            (res.val[0])[0] = (a.val[1])[1];
            break;
        }
        case 6: {
            (res.val[0])[0] = (a.val[1])[2];
            break;
        }
        case 7: {
            (res.val[0])[0] = (a.val[1])[3];
            break;
        }
        case 8: {
            (res.val[0])[0] = (a.val[2])[0];
            break;
        }
        case 9: {
            (res.val[0])[0] = (a.val[2])[1];
            break;
        }
        case 10: {
            (res.val[0])[0] = (a.val[2])[2];
            break;
        }
        case 11: {
            (res.val[0])[0] = (a.val[2])[3];
            break;
        }
        case 12: {
            (res.val[0])[0] = (a.val[3])[0];
            break;
        }
        case 13: {
            (res.val[0])[0] = (a.val[3])[1];
            break;
        }
        case 14: {
            (res.val[0])[0] = (a.val[3])[2];
            break;
        }
        case 15: {
            (res.val[0])[0] = (a.val[3])[3];
            break;
        }
        default: {
            (res.val[0])[0] = (a.val[4])[0];
            break;
        }
    }

    switch(e1)
    {
        case 0: {
            (res.val[0])[1] = (a.val[0])[0];
            break;
        }
        case 1: {
            (res.val[0])[1] = (a.val[0])[1];
            break;
        }
        case 2: {
            (res.val[0])[1] = (a.val[0])[2];
            break;
        }
        case 3: {
            (res.val[0])[1] = (a.val[0])[3];
            break;
        }
        case 4: {
            (res.val[0])[1] = (a.val[1])[0];
            break;
        }
        case 5: {
            (res.val[0])[1] = (a.val[1])[1];
            break;
        }
        case 6: {
            (res.val[0])[1] = (a.val[1])[2];
            break;
        }
        case 7: {
            (res.val[0])[1] = (a.val[1])[3];
            break;
        }
        case 8: {
            (res.val[0])[1] = (a.val[2])[0];
            break;
        }
        case 9: {
            (res.val[0])[1] = (a.val[2])[1];
            break;
        }
        case 10: {
            (res.val[0])[1] = (a.val[2])[2];
            break;
        }
        case 11: {
            (res.val[0])[1] = (a.val[2])[3];
            break;
        }
        case 12: {
            (res.val[0])[1] = (a.val[3])[0];
            break;
        }
        case 13: {
            (res.val[0])[1] = (a.val[3])[1];
            break;
        }
        case 14: {
            (res.val[0])[1] = (a.val[3])[2];
            break;
        }
        case 15: {
            (res.val[0])[1] = (a.val[3])[3];
            break;
        }
        default: {
            (res.val[0])[1] = (a.val[4])[0];
            break;
        }
    }

    switch(e2)
    {
        case 0: {
            (res.val[0])[2] = (a.val[0])[0];
            break;
        }
        case 1: {
            (res.val[0])[2] = (a.val[0])[1];
            break;
        }
        case 2: {
            (res.val[0])[2] = (a.val[0])[2];
            break;
        }
        case 3: {
            (res.val[0])[2] = (a.val[0])[3];
            break;
        }
        case 4: {
            (res.val[0])[2] = (a.val[1])[0];
            break;
        }
        case 5: {
            (res.val[0])[2] = (a.val[1])[1];
            break;
        }
        case 6: {
            (res.val[0])[2] = (a.val[1])[2];
            break;
        }
        case 7: {
            (res.val[0])[2] = (a.val[1])[3];
            break;
        }
        case 8: {
            (res.val[0])[2] = (a.val[2])[0];
            break;
        }
        case 9: {
            (res.val[0])[2] = (a.val[2])[1];
            break;
        }
        case 10: {
            (res.val[0])[2] = (a.val[2])[2];
            break;
        }
        case 11: {
            (res.val[0])[2] = (a.val[2])[3];
            break;
        }
        case 12: {
            (res.val[0])[2] = (a.val[3])[0];
            break;
        }
        case 13: {
            (res.val[0])[2] = (a.val[3])[1];
            break;
        }
        case 14: {
            (res.val[0])[2] = (a.val[3])[2];
            break;
        }
        case 15: {
            (res.val[0])[2] = (a.val[3])[3];
            break;
        }
        default: {
            (res.val[0])[2] = (a.val[4])[0];
            break;
        }
    }

    switch(e3)
    {
        case 0: {
            (res.val[0])[3] = (a.val[0])[0];
            break;
        }
        case 1: {
            (res.val[0])[3] = (a.val[0])[1];
            break;
        }
        case 2: {
            (res.val[0])[3] = (a.val[0])[2];
            break;
        }
        case 3: {
            (res.val[0])[3] = (a.val[0])[3];
            break;
        }
        case 4: {
            (res.val[0])[3] = (a.val[1])[0];
            break;
        }
        case 5: {
            (res.val[0])[3] = (a.val[1])[1];
            break;
        }
        case 6: {
            (res.val[0])[3] = (a.val[1])[2];
            break;
        }
        case 7: {
            (res.val[0])[3] = (a.val[1])[3];
            break;
        }
        case 8: {
            (res.val[0])[3] = (a.val[2])[0];
            break;
        }
        case 9: {
            (res.val[0])[3] = (a.val[2])[1];
            break;
        }
        case 10: {
            (res.val[0])[3] = (a.val[2])[2];
            break;
        }
        case 11: {
            (res.val[0])[3] = (a.val[2])[3];
            break;
        }
        case 12: {
            (res.val[0])[3] = (a.val[3])[0];
            break;
        }
        case 13: {
            (res.val[0])[3] = (a.val[3])[1];
            break;
        }
        case 14: {
            (res.val[0])[3] = (a.val[3])[2];
            break;
        }
        case 15: {
            (res.val[0])[3] = (a.val[3])[3];
            break;
        }
        default: {
            (res.val[0])[3] = (a.val[4])[0];
            break;
        }
    }

    switch(e4)
    {
        case 0: {
            (res.val[1])[0] = (a.val[0])[0];
            break;
        }
        case 1: {
            (res.val[1])[0] = (a.val[0])[1];
            break;
        }
        case 2: {
            (res.val[1])[0] = (a.val[0])[2];
            break;
        }
        case 3: {
            (res.val[1])[0] = (a.val[0])[3];
            break;
        }
        case 4: {
            (res.val[1])[0] = (a.val[1])[0];
            break;
        }
        case 5: {
            (res.val[1])[0] = (a.val[1])[1];
            break;
        }
        case 6: {
            (res.val[1])[0] = (a.val[1])[2];
            break;
        }
        case 7: {
            (res.val[1])[0] = (a.val[1])[3];
            break;
        }
        case 8: {
            (res.val[1])[0] = (a.val[2])[0];
            break;
        }
        case 9: {
            (res.val[1])[0] = (a.val[2])[1];
            break;
        }
        case 10: {
            (res.val[1])[0] = (a.val[2])[2];
            break;
        }
        case 11: {
            (res.val[1])[0] = (a.val[2])[3];
            break;
        }
        case 12: {
            (res.val[1])[0] = (a.val[3])[0];
            break;
        }
        case 13: {
            (res.val[1])[0] = (a.val[3])[1];
            break;
        }
        case 14: {
            (res.val[1])[0] = (a.val[3])[2];
            break;
        }
        case 15: {
            (res.val[1])[0] = (a.val[3])[3];
            break;
        }
        default: {
            (res.val[1])[0] = (a.val[4])[0];
            break;
        }
    }

    switch(e5)
    {
        case 0: {
            (res.val[1])[1] = (a.val[0])[0];
            break;
        }
        case 1: {
            (res.val[1])[1] = (a.val[0])[1];
            break;
        }
        case 2: {
            (res.val[1])[1] = (a.val[0])[2];
            break;
        }
        case 3: {
            (res.val[1])[1] = (a.val[0])[3];
            break;
        }
        case 4: {
            (res.val[1])[1] = (a.val[1])[0];
            break;
        }
        case 5: {
            (res.val[1])[1] = (a.val[1])[1];
            break;
        }
        case 6: {
            (res.val[1])[1] = (a.val[1])[2];
            break;
        }
        case 7: {
            (res.val[1])[1] = (a.val[1])[3];
            break;
        }
        case 8: {
            (res.val[1])[1] = (a.val[2])[0];
            break;
        }
        case 9: {
            (res.val[1])[1] = (a.val[2])[1];
            break;
        }
        case 10: {
            (res.val[1])[1] = (a.val[2])[2];
            break;
        }
        case 11: {
            (res.val[1])[1] = (a.val[2])[3];
            break;
        }
        case 12: {
            (res.val[1])[1] = (a.val[3])[0];
            break;
        }
        case 13: {
            (res.val[1])[1] = (a.val[3])[1];
            break;
        }
        case 14: {
            (res.val[1])[1] = (a.val[3])[2];
            break;
        }
        case 15: {
            (res.val[1])[1] = (a.val[3])[3];
            break;
        }
        default: {
            (res.val[1])[1] = (a.val[4])[0];
            break;
        }
    }

    switch(e6)
    {
        case 0: {
            (res.val[1])[2] = (a.val[0])[0];
            break;
        }
        case 1: {
            (res.val[1])[2] = (a.val[0])[1];
            break;
        }
        case 2: {
            (res.val[1])[2] = (a.val[0])[2];
            break;
        }
        case 3: {
            (res.val[1])[2] = (a.val[0])[3];
            break;
        }
        case 4: {
            (res.val[1])[2] = (a.val[1])[0];
            break;
        }
        case 5: {
            (res.val[1])[2] = (a.val[1])[1];
            break;
        }
        case 6: {
            (res.val[1])[2] = (a.val[1])[2];
            break;
        }
        case 7: {
            (res.val[1])[2] = (a.val[1])[3];
            break;
        }
        case 8: {
            (res.val[1])[2] = (a.val[2])[0];
            break;
        }
        case 9: {
            (res.val[1])[2] = (a.val[2])[1];
            break;
        }
        case 10: {
            (res.val[1])[2] = (a.val[2])[2];
            break;
        }
        case 11: {
            (res.val[1])[2] = (a.val[2])[3];
            break;
        }
        case 12: {
            (res.val[1])[2] = (a.val[3])[0];
            break;
        }
        case 13: {
            (res.val[1])[2] = (a.val[3])[1];
            break;
        }
        case 14: {
            (res.val[1])[2] = (a.val[3])[2];
            break;
        }
        case 15: {
            (res.val[1])[2] = (a.val[3])[3];
            break;
        }
        default: {
            (res.val[1])[2] = (a.val[4])[0];
            break;
        }
    }

    switch(e7)
    {
        case 0: {
            (res.val[1])[3] = (a.val[0])[0];
            break;
        }
        case 1: {
            (res.val[1])[3] = (a.val[0])[1];
            break;
        }
        case 2: {
            (res.val[1])[3] = (a.val[0])[2];
            break;
        }
        case 3: {
            (res.val[1])[3] = (a.val[0])[3];
            break;
        }
        case 4: {
            (res.val[1])[3] = (a.val[1])[0];
            break;
        }
        case 5: {
            (res.val[1])[3] = (a.val[1])[1];
            break;
        }
        case 6: {
            (res.val[1])[3] = (a.val[1])[2];
            break;
        }
        case 7: {
            (res.val[1])[3] = (a.val[1])[3];
            break;
        }
        case 8: {
            (res.val[1])[3] = (a.val[2])[0];
            break;
        }
        case 9: {
            (res.val[1])[3] = (a.val[2])[1];
            break;
        }
        case 10: {
            (res.val[1])[3] = (a.val[2])[2];
            break;
        }
        case 11: {
            (res.val[1])[3] = (a.val[2])[3];
            break;
        }
        case 12: {
            (res.val[1])[3] = (a.val[3])[0];
            break;
        }
        case 13: {
            (res.val[1])[3] = (a.val[3])[1];
            break;
        }
        case 14: {
            (res.val[1])[3] = (a.val[3])[2];
            break;
        }
        case 15: {
            (res.val[1])[3] = (a.val[3])[3];
            break;
        }
        default: {
            (res.val[1])[3] = (a.val[4])[0];
            break;
        }
    }

    switch(e8)
    {
        case 0: {
            (res.val[2])[0] = (a.val[0])[0];
            break;
        }
        case 1: {
            (res.val[2])[0] = (a.val[0])[1];
            break;
        }
        case 2: {
            (res.val[2])[0] = (a.val[0])[2];
            break;
        }
        case 3: {
            (res.val[2])[0] = (a.val[0])[3];
            break;
        }
        case 4: {
            (res.val[2])[0] = (a.val[1])[0];
            break;
        }
        case 5: {
            (res.val[2])[0] = (a.val[1])[1];
            break;
        }
        case 6: {
            (res.val[2])[0] = (a.val[1])[2];
            break;
        }
        case 7: {
            (res.val[2])[0] = (a.val[1])[3];
            break;
        }
        case 8: {
            (res.val[2])[0] = (a.val[2])[0];
            break;
        }
        case 9: {
            (res.val[2])[0] = (a.val[2])[1];
            break;
        }
        case 10: {
            (res.val[2])[0] = (a.val[2])[2];
            break;
        }
        case 11: {
            (res.val[2])[0] = (a.val[2])[3];
            break;
        }
        case 12: {
            (res.val[2])[0] = (a.val[3])[0];
            break;
        }
        case 13: {
            (res.val[2])[0] = (a.val[3])[1];
            break;
        }
        case 14: {
            (res.val[2])[0] = (a.val[3])[2];
            break;
        }
        case 15: {
            (res.val[2])[0] = (a.val[3])[3];
            break;
        }
        default: {
            (res.val[2])[0] = (a.val[4])[0];
            break;
        }
    }

    switch(e9)
    {
        case 0: {
            (res.val[2])[1] = (a.val[0])[0];
            break;
        }
        case 1: {
            (res.val[2])[1] = (a.val[0])[1];
            break;
        }
        case 2: {
            (res.val[2])[1] = (a.val[0])[2];
            break;
        }
        case 3: {
            (res.val[2])[1] = (a.val[0])[3];
            break;
        }
        case 4: {
            (res.val[2])[1] = (a.val[1])[0];
            break;
        }
        case 5: {
            (res.val[2])[1] = (a.val[1])[1];
            break;
        }
        case 6: {
            (res.val[2])[1] = (a.val[1])[2];
            break;
        }
        case 7: {
            (res.val[2])[1] = (a.val[1])[3];
            break;
        }
        case 8: {
            (res.val[2])[1] = (a.val[2])[0];
            break;
        }
        case 9: {
            (res.val[2])[1] = (a.val[2])[1];
            break;
        }
        case 10: {
            (res.val[2])[1] = (a.val[2])[2];
            break;
        }
        case 11: {
            (res.val[2])[1] = (a.val[2])[3];
            break;
        }
        case 12: {
            (res.val[2])[1] = (a.val[3])[0];
            break;
        }
        case 13: {
            (res.val[2])[1] = (a.val[3])[1];
            break;
        }
        case 14: {
            (res.val[2])[1] = (a.val[3])[2];
            break;
        }
        case 15: {
            (res.val[2])[1] = (a.val[3])[3];
            break;
        }
        default: {
            (res.val[2])[1] = (a.val[4])[0];
            break;
        }
    }

    switch(e10)
    {
        case 0: {
            (res.val[2])[2] = (a.val[0])[0];
            break;
        }
        case 1: {
            (res.val[2])[2] = (a.val[0])[1];
            break;
        }
        case 2: {
            (res.val[2])[2] = (a.val[0])[2];
            break;
        }
        case 3: {
            (res.val[2])[2] = (a.val[0])[3];
            break;
        }
        case 4: {
            (res.val[2])[2] = (a.val[1])[0];
            break;
        }
        case 5: {
            (res.val[2])[2] = (a.val[1])[1];
            break;
        }
        case 6: {
            (res.val[2])[2] = (a.val[1])[2];
            break;
        }
        case 7: {
            (res.val[2])[2] = (a.val[1])[3];
            break;
        }
        case 8: {
            (res.val[2])[2] = (a.val[2])[0];
            break;
        }
        case 9: {
            (res.val[2])[2] = (a.val[2])[1];
            break;
        }
        case 10: {
            (res.val[2])[2] = (a.val[2])[2];
            break;
        }
        case 11: {
            (res.val[2])[2] = (a.val[2])[3];
            break;
        }
        case 12: {
            (res.val[2])[2] = (a.val[3])[0];
            break;
        }
        case 13: {
            (res.val[2])[2] = (a.val[3])[1];
            break;
        }
        case 14: {
            (res.val[2])[2] = (a.val[3])[2];
            break;
        }
        case 15: {
            (res.val[2])[2] = (a.val[3])[3];
            break;
        }
        default: {
            (res.val[2])[2] = (a.val[4])[0];
            break;
        }
    }

    switch(e11)
    {
        case 0: {
            (res.val[2])[3] = (a.val[0])[0];
            break;
        }
        case 1: {
            (res.val[2])[3] = (a.val[0])[1];
            break;
        }
        case 2: {
            (res.val[2])[3] = (a.val[0])[2];
            break;
        }
        case 3: {
            (res.val[2])[3] = (a.val[0])[3];
            break;
        }
        case 4: {
            (res.val[2])[3] = (a.val[1])[0];
            break;
        }
        case 5: {
            (res.val[2])[3] = (a.val[1])[1];
            break;
        }
        case 6: {
            (res.val[2])[3] = (a.val[1])[2];
            break;
        }
        case 7: {
            (res.val[2])[3] = (a.val[1])[3];
            break;
        }
        case 8: {
            (res.val[2])[3] = (a.val[2])[0];
            break;
        }
        case 9: {
            (res.val[2])[3] = (a.val[2])[1];
            break;
        }
        case 10: {
            (res.val[2])[3] = (a.val[2])[2];
            break;
        }
        case 11: {
            (res.val[2])[3] = (a.val[2])[3];
            break;
        }
        case 12: {
            (res.val[2])[3] = (a.val[3])[0];
            break;
        }
        case 13: {
            (res.val[2])[3] = (a.val[3])[1];
            break;
        }
        case 14: {
            (res.val[2])[3] = (a.val[3])[2];
            break;
        }
        case 15: {
            (res.val[2])[3] = (a.val[3])[3];
            break;
        }
        default: {
            (res.val[2])[3] = (a.val[4])[0];
            break;
        }
    }

    switch(e12)
    {
        case 0: {
            (res.val[3])[0] = (a.val[0])[0];
            break;
        }
        case 1: {
            (res.val[3])[0] = (a.val[0])[1];
            break;
        }
        case 2: {
            (res.val[3])[0] = (a.val[0])[2];
            break;
        }
        case 3: {
            (res.val[3])[0] = (a.val[0])[3];
            break;
        }
        case 4: {
            (res.val[3])[0] = (a.val[1])[0];
            break;
        }
        case 5: {
            (res.val[3])[0] = (a.val[1])[1];
            break;
        }
        case 6: {
            (res.val[3])[0] = (a.val[1])[2];
            break;
        }
        case 7: {
            (res.val[3])[0] = (a.val[1])[3];
            break;
        }
        case 8: {
            (res.val[3])[0] = (a.val[2])[0];
            break;
        }
        case 9: {
            (res.val[3])[0] = (a.val[2])[1];
            break;
        }
        case 10: {
            (res.val[3])[0] = (a.val[2])[2];
            break;
        }
        case 11: {
            (res.val[3])[0] = (a.val[2])[3];
            break;
        }
        case 12: {
            (res.val[3])[0] = (a.val[3])[0];
            break;
        }
        case 13: {
            (res.val[3])[0] = (a.val[3])[1];
            break;
        }
        case 14: {
            (res.val[3])[0] = (a.val[3])[2];
            break;
        }
        case 15: {
            (res.val[3])[0] = (a.val[3])[3];
            break;
        }
        default: {
            (res.val[3])[0] = (a.val[4])[0];
            break;
        }
    }

    switch(e13)
    {
        case 0: {
            (res.val[3])[1] = (a.val[0])[0];
            break;
        }
        case 1: {
            (res.val[3])[1] = (a.val[0])[1];
            break;
        }
        case 2: {
            (res.val[3])[1] = (a.val[0])[2];
            break;
        }
        case 3: {
            (res.val[3])[1] = (a.val[0])[3];
            break;
        }
        case 4: {
            (res.val[3])[1] = (a.val[1])[0];
            break;
        }
        case 5: {
            (res.val[3])[1] = (a.val[1])[1];
            break;
        }
        case 6: {
            (res.val[3])[1] = (a.val[1])[2];
            break;
        }
        case 7: {
            (res.val[3])[1] = (a.val[1])[3];
            break;
        }
        case 8: {
            (res.val[3])[1] = (a.val[2])[0];
            break;
        }
        case 9: {
            (res.val[3])[1] = (a.val[2])[1];
            break;
        }
        case 10: {
            (res.val[3])[1] = (a.val[2])[2];
            break;
        }
        case 11: {
            (res.val[3])[1] = (a.val[2])[3];
            break;
        }
        case 12: {
            (res.val[3])[1] = (a.val[3])[0];
            break;
        }
        case 13: {
            (res.val[3])[1] = (a.val[3])[1];
            break;
        }
        case 14: {
            (res.val[3])[1] = (a.val[3])[2];
            break;
        }
        case 15: {
            (res.val[3])[1] = (a.val[3])[3];
            break;
        }
        default: {
            (res.val[3])[1] = (a.val[4])[0];
            break;
        }
    }

    switch(e14)
    {
        case 0: {
            (res.val[3])[2] = (a.val[0])[0];
            break;
        }
        case 1: {
            (res.val[3])[2] = (a.val[0])[1];
            break;
        }
        case 2: {
            (res.val[3])[2] = (a.val[0])[2];
            break;
        }
        case 3: {
            (res.val[3])[2] = (a.val[0])[3];
            break;
        }
        case 4: {
            (res.val[3])[2] = (a.val[1])[0];
            break;
        }
        case 5: {
            (res.val[3])[2] = (a.val[1])[1];
            break;
        }
        case 6: {
            (res.val[3])[2] = (a.val[1])[2];
            break;
        }
        case 7: {
            (res.val[3])[2] = (a.val[1])[3];
            break;
        }
        case 8: {
            (res.val[3])[2] = (a.val[2])[0];
            break;
        }
        case 9: {
            (res.val[3])[2] = (a.val[2])[1];
            break;
        }
        case 10: {
            (res.val[3])[2] = (a.val[2])[2];
            break;
        }
        case 11: {
            (res.val[3])[2] = (a.val[2])[3];
            break;
        }
        case 12: {
            (res.val[3])[2] = (a.val[3])[0];
            break;
        }
        case 13: {
            (res.val[3])[2] = (a.val[3])[1];
            break;
        }
        case 14: {
            (res.val[3])[2] = (a.val[3])[2];
            break;
        }
        case 15: {
            (res.val[3])[2] = (a.val[3])[3];
            break;
        }
        default: {
            (res.val[3])[2] = (a.val[4])[0];
            break;
        }
    }    
