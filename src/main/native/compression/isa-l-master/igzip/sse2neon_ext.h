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

    switch(e15)
    {
        case 0: {
            (res.val[3])[3] = (a.val[0])[0];
            break;
        }
        case 1: {
            (res.val[3])[3] = (a.val[0])[1];
            break;
        }
        case 2: {
            (res.val[3])[3] = (a.val[0])[2];
            break;
        }
        case 3: {
            (res.val[3])[3] = (a.val[0])[3];
            break;
        }
        case 4: {
            (res.val[3])[3] = (a.val[1])[0];
            break;
        }
        case 5: {
            (res.val[3])[3] = (a.val[1])[1];
            break;
        }
        case 6: {
            (res.val[3])[3] = (a.val[1])[2];
            break;
        }
        case 7: {
            (res.val[3])[3] = (a.val[1])[3];
            break;
        }
        case 8: {
            (res.val[3])[3] = (a.val[2])[0];
            break;
        }
        case 9: {
            (res.val[3])[3] = (a.val[2])[1];
            break;
        }
        case 10: {
            (res.val[3])[3] = (a.val[2])[2];
            break;
        }
        case 11: {
            (res.val[3])[3] = (a.val[2])[3];
            break;
        }
        case 12: {
            (res.val[3])[3] = (a.val[3])[0];
            break;
        }
        case 13: {
            (res.val[3])[3] = (a.val[3])[1];
            break;
        }
        case 14: {
            (res.val[3])[3] = (a.val[3])[2];
            break;
        }
        case 15: {
            (res.val[3])[3] = (a.val[3])[3];
            break;
        }
        default: {
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

//Copy a to dst, and insert the 64-bit integer i into dst at the location specified by index.
FORCE_INLINE __m256i _mm256_insert_epi64(__m256i a, __int64 i, const int index)
{
    int32x4x2_t res;

    switch(index)
    {
        case 0:{
            res.val[0] = (int32x4_t)vsetq_lane_s64((__int64)i,(int64x2_t)a.val[0], 0);
            res.val[1] = (int32x4_t)a.val[1];
            break;
        }
            
       case 1:{
            res.val[0] = (int32x4_t)vsetq_lane_s64((__int64)i,(int64x2_t)a.val[0], 1);
            res.val[1] = (int32x4_t)a.val[1];
            break;
        }

        case 2:{
            res.val[0] = (int32x4_t)a.val[0];
            res.val[1] = (int32x4_t)vsetq_lane_s64((__int64)i,(int64x2_t)a.val[1], 0);
            break;
        }

        case 3:{
            res.val[0] = (int32x4_t)a.val[0];
            res.val[1] = (int32x4_t)vsetq_lane_s64((__int64)i,(int64x2_t)a.val[1], 1);
            break;
        }

        default :{
            res.val[0] = (int32x4_t)a.val[0];
            res.val[1] = (int32x4_t)a.val[1];
            break;
        }
    }

    //res = (int64_t);
  
    return (__m256i)res;
}

//Copy a to dst, and insert the 32-bit integer i into dst at the location specified by index.
FORCE_INLINE __m256i _mm256_insert_epi32(__m256i a, __int32 i, const int index)
{
    int32x4x2_t res;

    switch(index)
    {
        case 0:{
            res.val[0] = (int32x4_t)vsetq_lane_s32((__int32)i,(int32x4_t)a.val[0], 0);
            res.val[1] = (int32x4_t)a.val[1];
            break;
        }
            
       case 1:{
            res.val[0] = (int32x4_t)vsetq_lane_s32((__int32)i,(int32x4_t)a.val[0], 1);
            res.val[1] = (int32x4_t)a.val[1];
            break;
        }

        case 2:{
            res.val[0] = (int32x4_t)vsetq_lane_s32((__int32)i,(int32x4_t)a.val[0], 2);
            res.val[1] = (int32x4_t)a.val[1];
            break;
        }

        case 3:{
            res.val[0] = (int32x4_t)vsetq_lane_s32((__int32)i,(int32x4_t)a.val[0], 3);
            res.val[1] = (int32x4_t)a.val[1];
            break;
        }

        case 4:{
            res.val[0] = (int32x4_t)a.val[0];
            res.val[1] = (int32x4_t)vsetq_lane_s32((__int32)i,(int32x4_t)a.val[1], 0);
            break;
        }

        case 5:{
            res.val[0] = (int32x4_t)a.val[0];
            res.val[1] = (int32x4_t)vsetq_lane_s32((__int32)i,(int32x4_t)a.val[1], 1);
            break;
        }

        case 6:{
            res.val[0] = (int32x4_t)a.val[0];
            res.val[1] = (int32x4_t)vsetq_lane_s32((__int32)i,(int32x4_t)a.val[1], 2);
            break;
        }

        case 7:{
            res.val[0] = (int32x4_t)a.val[0];
            res.val[1] = (int32x4_t)vsetq_lane_s32((__int32)i,(int32x4_t)a.val[1], 3);
            break;
        }

        default :{
            res.val[0] = (int32x4_t)a.val[0];
            res.val[1] = (int32x4_t)a.val[1];
            break;
        }
    }
  
    return (__m256i)res;
}

//Extract a 64-bit integer from a, selected with index, and store the result in dst.
FORCE_INLINE __int64 _mm256_extract_epi64(__m256i a, const int index)
{
    //int64_t res;

    switch(index)
    {
        case 0:
            return vgetq_lane_s64((int64x2_t)a.val[0], 0);
        case 1:
            return vgetq_lane_s64((int64x2_t)a.val[0], 1);
        case 2:
            return vgetq_lane_s64((int64x2_t)a.val[1], 0);
        case 3:
            return vgetq_lane_s64((int64x2_t)a.val[1], 1);
        default:
            return vgetq_lane_s64((int64x2_t)a.val[2], 0);
    }

    //res = (int64_t);
  
    //return (__int64)res;
}

//Extract a 32-bit integer from a, selected with index, and store the result in dst.
FORCE_INLINE __int32 _mm256_extract_epi32(__m256i a, const int index)
{
    //int32_t res;
    switch(index)
    {
        case 0:
            return vgetq_lane_s32((int32x4_t)a.val[0], 0);
        case 1:
            return vgetq_lane_s32((int32x4_t)a.val[0], 1);
        case 2:
            return vgetq_lane_s32((int32x4_t)a.val[0], 2);
        case 3:
            return vgetq_lane_s32((int32x4_t)a.val[0], 3);
        case 4:
            return vgetq_lane_s32((int32x4_t)a.val[1], 0);
        case 5:
            return vgetq_lane_s32((int32x4_t)a.val[1], 1);
        case 6:
            return vgetq_lane_s32((int32x4_t)a.val[1], 2);
        case 7:
            return vgetq_lane_s32((int32x4_t)a.val[1], 3);

        default:
            return vgetq_lane_s32((int32x4_t)a.val[2], 0);
    }

    //res = 0;
  
    //return (__int32)res;
}

//Shift packed 64-bit integers in a left by imm8 while shifting in zeros, and store the results in dst.
FORCE_INLINE __m256i _mm256_slli_epi64(__m256i a, const int imm8)
{
    int32x4x2_t res;

    //res = a;
        // int64x2_t vdupq_n_s64 (int64_t value)
    //if(imm8 > 63 || imm8 < 0){
        //res.val[0] = vdupq_n_s32((int32_t)0);
        //res.val[1] = vdupq_n_s32((int32_t)0);
        //return (__m256i)res;
    //}
        
    //res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], imm8);
    //res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], imm8);
    // int64x2_t vshlq_n_s64 (int64x2_t a, const int n)
    
    switch(imm8)
    {
        case 0:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 0);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 0);
        case 1:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 1);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 1);
        case 2:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 2);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 2);
        case 3:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 3);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 3);
        case 4:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 4);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 4);
        case 5:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 5);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 5);
        case 6:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 6);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 6);
        case 7:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 7);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 7);
        case 8:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 8);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 8);
        case 9:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 9);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 9);
        case 10:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 10);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 10);
        case 11:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 11);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 11);
        case 12:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 12);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 12);
        case 13:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 13);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 13);
        case 14:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 14);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 14);
        case 15:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 15);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 15);
        case 16:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 16);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 16);
        case 17:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 17);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 17);
        case 18:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 18);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 18);
        case 19:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 19);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 19);
        case 20:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 20);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 20);
        case 21:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 21);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 21);
        case 22:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 22);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 22);
        case 23:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 23);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 23);
        case 24:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 24);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 24);
        case 25:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 25);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 25);
        case 26:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 26);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 26);
        case 27:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 27);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 27);
        case 28:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 28);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 28);
        case 29:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 29);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 29);
        case 30:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 30);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 30);
        case 31:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 31);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 31);
        case 32:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 32);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 32);
        case 33:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 33);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 33);
        case 34:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 34);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 34);
        case 35:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 35);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 35);
        case 36:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 36);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 36);
        case 37:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 37);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 37);
        case 38:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 38);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 38);
        case 39:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 39);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 39);
        case 40:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 40);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 40);
        case 41:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 41);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 41);
        case 42:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 42);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 42);
        case 43:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 43);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 43);
        case 44:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 44);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 44);
        case 45:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 45);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 45);
        case 46:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 46);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 46);
        case 47:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 47);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 47);
        case 48:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 48);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 48);
        case 49:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 49);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 49);
        case 50:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 50);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 50);
        case 51:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 51);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 51);
        case 52:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 52);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 52);
        case 53:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 53);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 53);
        case 54:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 54);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 54);
        case 55:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 55);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 55);
        case 56:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 56);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 56);
        case 57:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 57);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 57);
        case 58:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 58);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 58);
        case 59:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 59);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 59);
        case 60:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 60);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 60);
        case 61:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 61);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 61);
        case 62:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 62);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 62);
        case 63:
            res.val[0] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[0], 63);
            res.val[1] = (int32x4_t)vshlq_n_s64((int64x2_t)a.val[1], 63);
        default:
            res.val[0] = vdupq_n_s32((int32_t)0);
            res.val[1] = vdupq_n_s32((int32_t)0);
    }
    return (__m256i)res;
}

//Shift packed 32-bit integers in a left by imm8 while shifting in zeros, and store the results in dst.
FORCE_INLINE __m256i _mm256_slli_epi32(__m256i a, const int imm8)
{
    int32x4x2_t res;

    //res = a;
    //if(imm8 > 31 || imm8 < 0){
        //res.val[0] = vdupq_n_s32((int32_t)0);
        //res.val[1] = vdupq_n_s32((int32_t)0);
        //return (__m256i)res;
    //}
        
    //res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], imm8);
    //res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], imm8);

    switch(imm8)
    {
        case 0:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 0);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 0);
        case 1:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 1);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 1);
        case 2:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 2);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 2);
        case 3:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 3);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 3);
        case 4:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 4);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 4);
        case 5:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 5);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 5);
        case 6:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 6);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 6);
        case 7:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 7);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 7);
        case 8:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 8);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 8);
        case 9:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 9);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 9);
        case 10:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 10);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 10);
        case 11:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 11);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 11);
        case 12:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 12);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 12);
        case 13:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 13);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 13);
        case 14:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 14);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 14);
        case 15:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 15);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 15);
        case 16:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 16);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 16);
        case 17:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 17);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 17);
        case 18:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 18);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 18);
        case 19:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 19);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 19);
        case 20:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 20);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 20);
        case 21:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 21);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 21);
        case 22:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 22);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 22);
        case 23:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 23);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 23);
        case 24:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 24);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 24);
        case 25:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 25);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 25);
        case 26:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 26);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 26);
        case 27:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 27);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 27);
        case 28:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 28);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 28);
        case 29:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 29);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 29);
        case 30:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 30);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 30);
        case 31:
            res.val[0] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[0], 31);
            res.val[1] = (int32x4_t)vshlq_n_s32((int32x4_t)a.val[1], 31);
        default:
            res.val[0] = vdupq_n_s32((int32_t)0);
            res.val[1] = vdupq_n_s32((int32_t)0);
    }
  
    return (__m256i)res;
}

//Shift 128-bit lanes in a left by imm8 bytes while shifting in zeros, and store the results in dst.
/*FORCE_INLINE __m256i _mm256_slli_si256(__m256i a, const int imm8)
{
    int32x4x2_t res;

    res.val[0] = (int32x4_t)a.val[0]; 
    res.val[1] = (int32x4_t)a.val[1];
  
    return (__m256i)res;
}

//Shift 128-bit lanes in a left by imm8 bytes while shifting in zeros, and store the results in dst.
FORCE_INLINE __m512i _mm512_bslli_epi128(__m512i a, int imm8)
{
    int32x4x4_t res;

    res.val[0] = (int32x4_t)a.val[0]; 
    res.val[1] = (int32x4_t)a.val[1];
    res.val[2] = (int32x4_t)a.val[2]; 
    res.val[3] = (int32x4_t)a.val[3];
  
    return (__m512i)res;
}

//Shift 128-bit lanes in a right by imm8 bytes while shifting in zeros, and store the results in dst.
FORCE_INLINE __m512i _mm512_bsrli_epi128(__m512i a, int imm8)
{
    int32x4x4_t res;

    res.val[0] = (int32x4_t)a.val[0]; 
    res.val[1] = (int32x4_t)a.val[1];
    res.val[2] = (int32x4_t)a.val[2]; 
    res.val[3] = (int32x4_t)a.val[3];
  
    return (__m512i)res;
}*/

//Copy a to dst, then insert 256 bits (composed of 8 packed 32-bit integers) from b into dst at the location specified by imm8.
FORCE_INLINE __m512i _mm512_inserti32x8(__m512i a, __m256i b, int imm8)
{
    int32x4x4_t res;

    res.val[0] = (int32x4_t)a.val[0]; 
    res.val[1] = (int32x4_t)a.val[1];
    res.val[2] = (int32x4_t)a.val[2]; 
    res.val[3] = (int32x4_t)a.val[3];


    switch(imm8)
    {
        case 0: {
            res.val[0] = (int32x4_t)b.val[0]; 
            res.val[1] = (int32x4_t)b.val[1];
            break;
        }
        case 1: {
            res.val[2] = (int32x4_t)b.val[0]; 
            res.val[3] = (int32x4_t)b.val[1];
            break;
        }
        default: {
            res.val[2] = (int32x4_t)b.val[2]; 
            res.val[3] = (int32x4_t)b.val[2];
            break;
        }
    }
  

    //res = a;
  
    return (__m512i)res;
}

//Copy a to dst, then insert 256 bits (composed of 4 packed 64-bit integers) from b into dst at the location specified by imm8.
FORCE_INLINE __m512i _mm512_inserti64x4(__m512i a, __m256i b, int imm8)
{
    //int32x4x4_t res;
    int32x4x4_t res;

    res.val[0] = (int32x4_t)a.val[0]; 
    res.val[1] = (int32x4_t)a.val[1];
    res.val[2] = (int32x4_t)a.val[2]; 
    res.val[3] = (int32x4_t)a.val[3];

    /*if (imm8 == 0){
        res.val[0] = (float64x2_t)a.val[0]; 
        res.val[1] = (float64x2_t)a.val[1];
    }

    if (imm8 == 1){
        res.val[2] = (float64x2_t)a.val[2]; 
        res.val[3] = (float64x2_t)a.val[3];
    }*/
    switch(imm8)
    {
        case 0: {
            res.val[0] = (int32x4_t)b.val[0]; 
            res.val[1] = (int32x4_t)b.val[1];
            break;
        }
        case 1: {
            res.val[2] = (int32x4_t)b.val[0]; 
            res.val[3] = (int32x4_t)b.val[1];
            break;
        }
        default: {
            res.val[2] = (int32x4_t)b.val[2]; 
            res.val[3] = (int32x4_t)b.val[2];
            break;
        }
    }
  
    return (__m512i)res;

    //res = a;
  
    //return (__m512i)res;
}

//Extract 256 bits (composed of 8 packed single-precision (32-bit) floating-point elements) from a, selected with imm8, and store the result in dst.
FORCE_INLINE __m256 _mm512_extractf32x8_ps(__m512 a, int imm8)
{
    float32x4x2_t res;

    switch(imm8)
    {
        case 0: {
            res.val[0] = (float32x4_t)a.val[0]; 
            res.val[1] = (float32x4_t)a.val[1];
            break;
        }
        case 1: {
            res.val[0] = (float32x4_t)a.val[2]; 
            res.val[1] = (float32x4_t)a.val[3];
            break;
        }
        default: {
            res.val[0] = (float32x4_t)a.val[4]; 
            res.val[1] = (float32x4_t)a.val[4];
            break;
        }
    }

  
    return (__m256)res;
}


//Extract 256 bits (composed of 4 packed double-precision (64-bit) floating-point elements) from a, selected with imm8, and store the result in dst.
FORCE_INLINE __m256d _mm512_extractf64x4_pd(__m512d a, int imm8)
{
    float64x2x2_t res;
    /*if (imm8 == 0){
        res.val[0] = (float64x2_t)a.val[0]; 
        res.val[1] = (float64x2_t)a.val[1];
    }

    if (imm8 == 1){
        res.val[2] = (float64x2_t)a.val[2]; 
        res.val[3] = (float64x2_t)a.val[3];
    }*/
    switch(imm8)
    {
        case 0: {
            res.val[0] = (float64x2_t)a.val[0]; 
            res.val[1] = (float64x2_t)a.val[1];
            break;
        }
        case 1: {
            res.val[0] = (float64x2_t)a.val[2]; 
            res.val[1] = (float64x2_t)a.val[3];
            break;
        }
        default: {
            res.val[0] = (float64x2_t)a.val[4]; 
            res.val[1] = (float64x2_t)a.val[4];
            break;
        }
    }

  
    return (__m256d)res;
}

//Copy a to dst, then insert 256 bits (composed of 4 packed double-precision (64-bit) floating-point elements) from b into dst at the location specified by imm8.
FORCE_INLINE __m512d _mm512_insertf64x4(__m512d a, __m256d b, int imm8)
{
    float64x2x4_t res;

    res.val[0] = (float64x2_t)a.val[0]; 
    res.val[1] = (float64x2_t)a.val[1];
    res.val[2] = (float64x2_t)a.val[2]; 
    res.val[3] = (float64x2_t)a.val[3];

    switch(imm8)
    {
        case 0: {
            res.val[0] = (float64x2_t)b.val[0]; 
            res.val[1] = (float64x2_t)b.val[1];
            break;
        }
        case 1: {
            res.val[2] = (float64x2_t)b.val[0]; 
            res.val[3] = (float64x2_t)b.val[1];
            break;
        }
        default: {
            res.val[2] = (float64x2_t)b.val[2]; 
            res.val[3] = (float64x2_t)b.val[2];
            break;
        }
    }
  
    return (__m512d)res;
} 

//Copy a to dst, then insert 256 bits (composed of 4 packed double-precision (32-bit) floating-point elements) from b into dst at the location specified by imm8.
FORCE_INLINE __m512 _mm512_insertf32x8(__m512 a, __m256 b, int imm8)
{
    float32x4x4_t res;

    res.val[0] = (float32x4_t)a.val[0]; 
    res.val[1] = (float32x4_t)a.val[1];
    res.val[2] = (float32x4_t)a.val[2]; 
    res.val[3] = (float32x4_t)a.val[3];

    switch(imm8)
    {
        case 0: {
            res.val[0] = (float32x4_t)b.val[0]; 
            res.val[1] = (float32x4_t)b.val[1];
            break;
        }
        case 1: {
            res.val[2] = (float32x4_t)b.val[0]; 
            res.val[3] = (float32x4_t)b.val[1];
            break;
        }
        default: {
            res.val[2] = (float32x4_t)b.val[2]; 
            res.val[3] = (float32x4_t)b.val[2];
            break;
        }
    }
  
    return (__m512)res;
} 

// mask %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//Blend packed double-precision (64-bit) floating-point elements from a and b using control mask k, and store the results in dst.
FORCE_INLINE __m512d _mm512_mask_blend_pd(__mmask8 k, __m512d a, __m512d b)
{
    float64x2x4_t res;
    //res = a;
    (res.val[0])[0] = (k & 0x01) ? ((float64x2_t)b.val[0])[0] : ((float64x2_t)a.val[0])[0];
    (res.val[0])[1] = (k & 0x02) ? ((float64x2_t)b.val[0])[1] : ((float64x2_t)a.val[0])[1];
    (res.val[1])[0] = (k & 0x04) ? ((float64x2_t)b.val[1])[0] : ((float64x2_t)a.val[1])[0];
    (res.val[1])[1] = (k & 0x08) ? ((float64x2_t)b.val[1])[1] : ((float64x2_t)a.val[1])[1];

    (res.val[2])[0] = (k & 0x10) ? ((float64x2_t)b.val[2])[0] : ((float64x2_t)a.val[2])[0];
    (res.val[2])[1] = (k & 0x20) ? ((float64x2_t)b.val[2])[1] : ((float64x2_t)a.val[2])[1];
    (res.val[3])[0] = (k & 0x40) ? ((float64x2_t)b.val[3])[0] : ((float64x2_t)a.val[3])[0];
    (res.val[3])[1] = (k & 0x80) ? ((float64x2_t)b.val[3])[1] : ((float64x2_t)a.val[3])[1];
    /*if ((k & 0x01) == 0) (res.val[0])[0] = (a.val[0])[0];
    else (res.val[0])[0] = (b.val[0])[0];

    if ((k & 0x02) == 0) (res.val[0])[1] = (a.val[0])[1];
    else (res.val[0])[1] = (b.val[0])[1];

    if ((k & 0x04) == 0) (res.val[1])[0] = (a.val[1])[0];
    else (res.val[1])[0] = (b.val[1])[0];

    if ((k & 0x08) == 0) (res.val[1])[1] = (a.val[1])[1];
    else (res.val[1])[1] = (b.val[1])[1];

    if ((k & 0x10) == 0) (res.val[2])[0] = (a.val[2])[0];
    else (res.val[2])[0] = (b.val[2])[0];

    if ((k & 0x20) == 0) (res.val[2])[1] = (a.val[2])[1];
    else (res.val[2])[1] = (b.val[2])[1];

    if ((k & 0x40) == 0) (res.val[3])[0] = (a.val[3])[0];
    else (res.val[3])[0] = (b.val[3])[0];

    if ((k & 0x80) == 0) (res.val[3])[1] = (a.val[3])[1];
    else (res.val[3])[1] = (b.val[3])[1];*/


    return (__m512d)res;
}

//Blend packed double-precision (32-bit) floating-point elements from a and b using control mask k, and store the results in dst.
FORCE_INLINE __m512 _mm512_mask_blend_ps(__mmask16 k, __m512 a, __m512 b)
{
    float32x4x4_t res;
    //res = a;

    (res.val[0])[0] = (k & 0x0001) ? ((float32x4_t)b.val[0])[0] : ((float32x4_t)a.val[0])[0];
    (res.val[0])[1] = (k & 0x0002) ? ((float32x4_t)b.val[0])[1] : ((float32x4_t)a.val[0])[1];
    (res.val[0])[2] = (k & 0x0004) ? ((float32x4_t)b.val[0])[2] : ((float32x4_t)a.val[0])[2];
    (res.val[0])[3] = (k & 0x0008) ? ((float32x4_t)b.val[0])[3] : ((float32x4_t)a.val[0])[3];

    (res.val[1])[0] = (k & 0x0010) ? ((float32x4_t)b.val[1])[0] : ((float32x4_t)a.val[1])[0];
    (res.val[1])[1] = (k & 0x0020) ? ((float32x4_t)b.val[1])[1] : ((float32x4_t)a.val[1])[1];
    (res.val[1])[2] = (k & 0x0040) ? ((float32x4_t)b.val[1])[2] : ((float32x4_t)a.val[1])[2];
    (res.val[1])[3] = (k & 0x0080) ? ((float32x4_t)b.val[1])[3] : ((float32x4_t)a.val[1])[3];

    (res.val[2])[0] = (k & 0x0100) ? ((float32x4_t)b.val[2])[0] : ((float32x4_t)a.val[2])[0];
    (res.val[2])[1] = (k & 0x0200) ? ((float32x4_t)b.val[2])[1] : ((float32x4_t)a.val[2])[1];
    (res.val[2])[2] = (k & 0x0400) ? ((float32x4_t)b.val[2])[2] : ((float32x4_t)a.val[2])[2];
    (res.val[2])[3] = (k & 0x0800) ? ((float32x4_t)b.val[2])[3] : ((float32x4_t)a.val[2])[3];

    (res.val[3])[0] = (k & 0x1000) ? ((float32x4_t)b.val[3])[0] : ((float32x4_t)a.val[3])[0];
    (res.val[3])[1] = (k & 0x2000) ? ((float32x4_t)b.val[3])[1] : ((float32x4_t)a.val[3])[1];
    (res.val[3])[2] = (k & 0x4000) ? ((float32x4_t)b.val[3])[2] : ((float32x4_t)a.val[3])[2];
    (res.val[3])[3] = (k & 0x8000) ? ((float32x4_t)b.val[3])[3] : ((float32x4_t)a.val[3])[3];


    /*if ((k & 0x0001) == 0) (res.val[0])[0] = (a.val[0])[0];
    else (res.val[0])[0] = (b.val[0])[0];

    if ((k & 0x0002) == 0) (res.val[0])[1] = (a.val[0])[1];
    else (res.val[0])[1] = (b.val[0])[1];

    if ((k & 0x0004) == 0) (res.val[0])[2] = (a.val[0])[2];
    else (res.val[0])[2] = (b.val[0])[2];

    if ((k & 0x0008) == 0) (res.val[0])[3] = (a.val[0])[3];
    else (res.val[0])[3] = (b.val[0])[3];


    if ((k & 0x0010) == 0) (res.val[1])[0] = (a.val[1])[0];
    else (res.val[1])[0] = (b.val[1])[0];

    if ((k & 0x0020) == 0) (res.val[1])[1] = (a.val[1])[1];
    else (res.val[1])[1] = (b.val[1])[1];

    if ((k & 0x0040) == 0) (res.val[1])[2] = (a.val[1])[2];
    else (res.val[1])[2] = (b.val[1])[2];

    if ((k & 0x0080) == 0) (res.val[1])[3] = (a.val[1])[3];
    else (res.val[1])[3] = (b.val[1])[3];


    if ((k & 0x0100) == 0) (res.val[2])[0] = (a.val[2])[0];
    else (res.val[2])[0] = (b.val[2])[0];

    if ((k & 0x0200) == 0) (res.val[2])[1] = (a.val[2])[1];
    else (res.val[2])[1] = (b.val[2])[1];

    if ((k & 0x0400) == 0) (res.val[2])[2] = (a.val[2])[2];
    else (res.val[2])[2] = (b.val[2])[2];

    if ((k & 0x0800) == 0) (res.val[2])[3] = (a.val[2])[3];
    else (res.val[2])[3] = (b.val[2])[3];


    if ((k & 0x1000) == 0) (res.val[3])[0] = (a.val[3])[0];
    else (res.val[3])[0] = (b.val[3])[0];

    if ((k & 0x2000) == 0) (res.val[3])[1] = (a.val[3])[1];
    else (res.val[3])[1] = (b.val[3])[1];

    if ((k & 0x4000) == 0) (res.val[3])[2] = (a.val[3])[2];
    else (res.val[3])[2] = (b.val[3])[2];

    if ((k & 0x8000) == 0) (res.val[3])[3] = (a.val[3])[3];
    else (res.val[3])[3] = (b.val[3])[3];*/


    return (__m512)res;
}

//Compute the bitwise AND of packed 64-bit integers in a and b, producing intermediate 64-bit values, and set the corresponding bit in result mask k if the intermediate value is non-zero.
FORCE_INLINE __mmask8 _mm512_test_epi64_mask(__m512i a, __m512i b)
{
    int8_t res = 0x00;

    int64x2x4_t resand;
    //res = 0;
    resand.val[0] = vandq_s64((int64x2_t)a.val[0], (int64x2_t)b.val[0]);
    resand.val[1] = vandq_s64((int64x2_t)a.val[1], (int64x2_t)b.val[1]);
    resand.val[2] = vandq_s64((int64x2_t)a.val[2], (int64x2_t)b.val[2]);
    resand.val[3] = vandq_s64((int64x2_t)a.val[3], (int64x2_t)b.val[3]);

    if (vgetq_lane_s64(resand.val[0], 0) != 0) res = res | 0x01;

    if (vgetq_lane_s64(resand.val[0], 1) != 0) res = res | 0x02;

    if (vgetq_lane_s64(resand.val[1], 0) != 0) res = res | 0x04;

    if (vgetq_lane_s64(resand.val[1], 1) != 0) res = res | 0x08;

    if (vgetq_lane_s64(resand.val[2], 0) != 0) res = res | 0x10;

    if (vgetq_lane_s64(resand.val[2], 1) != 0) res = res | 0x20;

    if (vgetq_lane_s64(resand.val[3], 0) != 0) res = res | 0x40;

    if (vgetq_lane_s64(resand.val[3], 1) != 0) res = res | 0x80;

    return (__mmask8)res;
}

//Compute the bitwise AND of packed 64-bit integers in a and b, producing intermediate 32-bit values, and set the corresponding bit in result mask k if the intermediate value is non-zero.
FORCE_INLINE __mmask16 _mm512_test_epi32_mask(__m512i a, __m512i b)
{
    int16_t res = 0x0000;
    int32x4x4_t resand;
 
    resand.val[0] = vandq_s32((int32x4_t)a.val[0], (int32x4_t)b.val[0]);
    resand.val[1] = vandq_s32((int32x4_t)a.val[1], (int32x4_t)b.val[1]);
    resand.val[2] = vandq_s32((int32x4_t)a.val[2], (int32x4_t)b.val[2]);
    resand.val[3] = vandq_s32((int32x4_t)a.val[3], (int32x4_t)b.val[3]);

    if (vgetq_lane_s32(resand.val[0], 0) != 0) res = res | 0x0001;

    if (vgetq_lane_s32(resand.val[0], 1) != 0) res = res | 0x0002;

    if (vgetq_lane_s32(resand.val[0], 2) != 0) res = res | 0x0004;

    if (vgetq_lane_s32(resand.val[0], 3) != 0) res = res | 0x0008;

    if (vgetq_lane_s32(resand.val[1], 0) != 0) res = res | 0x0010;

    if (vgetq_lane_s32(resand.val[1], 1) != 0) res = res | 0x0020;

    if (vgetq_lane_s32(resand.val[1], 2) != 0) res = res | 0x0040;

    if (vgetq_lane_s32(resand.val[1], 3) != 0) res = res | 0x0080;

    if (vgetq_lane_s32(resand.val[2], 0) != 0) res = res | 0x0100;

    if (vgetq_lane_s32(resand.val[2], 1) != 0) res = res | 0x0200;

    if (vgetq_lane_s32(resand.val[2], 2) != 0) res = res | 0x0400;

    if (vgetq_lane_s32(resand.val[2], 3) != 0) res = res | 0x0800;

    if (vgetq_lane_s32(resand.val[3], 0) != 0) res = res | 0x1000;

    if (vgetq_lane_s32(resand.val[3], 1) != 0) res = res | 0x2000;

    if (vgetq_lane_s32(resand.val[3], 2) != 0) res = res | 0x4000;

    if (vgetq_lane_s32(resand.val[3], 3) != 0) res = res | 0x8000;

    return (__mmask16)res;
}

// smithwaterman ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//256////////////////
//Allocate size bytes of memory, aligned to the alignment specified in align, and return a pointer to the allocated memory.
#ifndef _MM_MALLOC_H_INCLUDED
#define _MM_MALLOC_H_INCLUDED

#ifndef __cplusplus
extern int posix_memalign (void **, size_t, size_t);
#else
extern "C" int posix_memalign (void **, size_t, size_t) throw ();
#endif

FORCE_INLINE void * _mm_malloc(size_t size, size_t alignment)
{
    void *ptr;
    if (alignment == 1)
        return malloc (size);
    if (alignment == 2 || (sizeof (void *) == 8 && alignment == 4))
        alignment = sizeof (void *);
    if (posix_memalign (&ptr, alignment, size) == 0)
        return ptr;
    else
        return NULL;

    /*//PowerPC64 ELF V2 ABI requires quadword alignment.  
    size_t vec_align = sizeof (__vector float);
    //Linux GLIBC malloc alignment is at least 2 X ptr size.  
    size_t malloc_align = (sizeof (void *) + sizeof (void *));
    void *ptr;

    if (alignment == malloc_align && alignment == vec_align)
        return malloc (size);
    if (alignment < vec_align)
        alignment = vec_align;
    if (posix_memalign (&ptr, alignment, size) == 0)
        return ptr;
    else
        return NULL;*/

}

FORCE_INLINE void _mm_free(void * ptr)
{
    free (ptr);
}

#endif

//Store 256-bits of integer data from a into memory using a non-temporal memory hint.
FORCE_INLINE void _mm256_stream_si256(__m256i *mem_addr, __m256i a)
{
    //*mem_addr = a;
    //vst2q_s32((int32_t*) mem_addr, (int32x4x2_t)a);
    vst1q_s32((int32_t*) mem_addr, a.val[0]);
    vst1q_s32(((int32_t*) mem_addr) + 4, a.val[1]);
}

//Convert packed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst.
FORCE_INLINE __m256i _mm256_packs_epi32(__m256i a, __m256i b)
{
    int32x4x2_t res;

    res.val[0] = (int32x4_t)vcombine_s16(vqmovn_s32(a.val[0]), vqmovn_s32(b.val[0]));
    res.val[1] = (int32x4_t)vcombine_s16(vqmovn_s32(a.val[1]), vqmovn_s32(b.val[1]));

    return (__m256i)res;
}

FORCE_INLINE __m128i SELECT4(__m256i a, __m256i b, int imm8)
{
    int32x4_t res;
    int imm8_3;
    imm8_3 = imm8 & 0x00000003;

    switch(imm8_3)
    {
        case 0:{
            res = a.val[0];
            break;
        }
        case 1:{
            res = a.val[1];
            break;
        }
        case 2:{
            res = b.val[0];
            break;
        }
        case 3:{
            res = b.val[1];
            break;
        }
        default:{
            res = a.val[2];
            break;
        }
    }
    if ((imm8 & 0x00000008) != 0) res = vdupq_n_s32(0);

    return (__m128i)res;
}


//Shuffle 128-bits (composed of integer data) selected by imm8 from a and b, and store the results in dst.
FORCE_INLINE __m256i _mm256_permute2f128_si256(__m256i a, __m256i b, int imm8)
{
    int32x4x2_t res;
    int imm8_4;
    imm8_4 = imm8 >> 4;

    res.val[0] = SELECT4(a, b, imm8);
    res.val[1] = SELECT4(a, b, imm8_4);

    return (__m256i)res;
}

// Blend packed single-precision (32-bit) floating-point elements from a and b using mask, and store the results in dst.
FORCE_INLINE __m256i _mm256_blendv_epi32(__m256i a, __m256i b, __m256i mask)
{
    int tmp = 0x80000000;
    int32x4x2_t res;
    
    (res.val[0])[0] = (((int32x4_t)mask.val[0])[0] & tmp) ? ((int32x4_t)b.val[0])[0] : ((int32x4_t)a.val[0])[0];
    (res.val[0])[1] = (((int32x4_t)mask.val[0])[1] & tmp) ? ((int32x4_t)b.val[0])[1] : ((int32x4_t)a.val[0])[1];
    (res.val[0])[2] = (((int32x4_t)mask.val[0])[2] & tmp) ? ((int32x4_t)b.val[0])[2] : ((int32x4_t)a.val[0])[2];
    (res.val[0])[3] = (((int32x4_t)mask.val[0])[3] & tmp) ? ((int32x4_t)b.val[0])[3] : ((int32x4_t)a.val[0])[3];
    
    (res.val[1])[0] = (((int32x4_t)mask.val[1])[0] & tmp) ? ((int32x4_t)b.val[1])[0] : ((int32x4_t)a.val[1])[0];
    (res.val[1])[1] = (((int32x4_t)mask.val[1])[1] & tmp) ? ((int32x4_t)b.val[1])[1] : ((int32x4_t)a.val[1])[1];
    (res.val[1])[2] = (((int32x4_t)mask.val[1])[2] & tmp) ? ((int32x4_t)b.val[1])[2] : ((int32x4_t)a.val[1])[2];
    (res.val[1])[3] = (((int32x4_t)mask.val[1])[3] & tmp) ? ((int32x4_t)b.val[1])[3] : ((int32x4_t)a.val[1])[3];

    return (__m256i)res;
}

//Compute the bitwise AND of 256 bits (representing integer data) in a and b, and store the result in dst.
FORCE_INLINE __m256i _mm256_and_si256(__m256i a, __m256i b)
{
    int32x4x2_t res;

    res.val[0] = vandq_s32((int32x4_t)a.val[0], (int32x4_t)b.val[0]);
    res.val[1] = vandq_s32((int32x4_t)a.val[1], (int32x4_t)b.val[1]);
   
    return (__m256i)res;
}

//Compute the bitwise OR of 256 bits (representing integer data) in a and b, and store the result in dst.
FORCE_INLINE __m256i _mm256_or_si256(__m256i a, __m256i b)
{
    int32x4x2_t res;

    res.val[0] = vorrq_s32((int32x4_t)a.val[0], (int32x4_t)b.val[0]);
    res.val[1] = vorrq_s32((int32x4_t)a.val[1], (int32x4_t)b.val[1]);
   
    return (__m256i)res;
}

//Compare packed 32-bit integers in a and b for equality, and store the results in dst.
FORCE_INLINE __m256i _mm256_cmpeq_epi32(__m256i a, __m256i b)
{
    int32x4x2_t res;

    res.val[0] = (int32x4_t)vceqq_s32((int32x4_t)a.val[0], (int32x4_t)b.val[0]);
    res.val[1] = (int32x4_t)vceqq_s32((int32x4_t)a.val[1], (int32x4_t)b.val[1]);

    return(__m256i)res;
}

//Store 256-bits of integer data from a into memory. mem_addr does not need to be aligned on any particular boundary.
FORCE_INLINE void _mm256_storeu_si256(__m256i *mem_addr, __m256i a)
{
    //vst2q_s32((int32_t*) mem_addr, (int32x4x2_t)a);
    vst1q_s32((int32_t*) mem_addr, a.val[0]);
    vst1q_s32(((int32_t*) mem_addr) + 4, a.val[1]);
}

//Compute the bitwise NOT of 256 bits (representing integer data) in a and then AND with b, and store the result in dst.
FORCE_INLINE __m256i _mm256_andnot_si256(__m256i a, __m256i b)
{
    int32x4x2_t res;

    res.val[0] = vbicq_s32((int32x4_t)b.val[0], (int32x4_t)a.val[0]);
    res.val[1] = vbicq_s32((int32x4_t)b.val[1], (int32x4_t)a.val[1]);
   
    return (__m256i)res;
}

//Compare packed 32-bit integers in a and b for greater-than, and store the results in dst.
FORCE_INLINE __m256i _mm256_cmpgt_epi32(__m256i a, __m256i b)
{
    int32x4x2_t res;

    res.val[0] = (int32x4_t)vcgtq_s32((int32x4_t)a.val[0], (int32x4_t)b.val[0]);
    res.val[1] = (int32x4_t)vcgtq_s32((int32x4_t)a.val[1], (int32x4_t)b.val[1]);

    return(__m256i)res;
}

//Compare packed 32-bit integers in a and b, and store packed maximum values in dst.
FORCE_INLINE __m256i _mm256_max_epi32(__m256i a, __m256i b)
{
    int32x4x2_t res;

    res.val[0] = vmaxq_s32((int32x4_t)a.val[0], (int32x4_t)b.val[0]);
    res.val[1] = vmaxq_s32((int32x4_t)a.val[1], (int32x4_t)b.val[1]);

    return(__m256i)res;
}

//Add packed 32-bit integers in a and b, and store the results in dst.
FORCE_INLINE __m256i _mm256_add_epi32(__m256i a, __m256i b)
{
    int32x4x2_t res;

    res.val[0] = vaddq_s32((int32x4_t)a.val[0], (int32x4_t)b.val[0]);
    res.val[1] = vaddq_s32((int32x4_t)a.val[1], (int32x4_t)b.val[1]);

    return(__m256i)res;
}

//Load 256-bits of integer data from memory into dst. mem_addr does not need to be aligned on any particular boundary.
FORCE_INLINE __m256i _mm256_loadu_si256(const __m256i *mem_addr)
{
    int32x4x2_t res;

    res.val[0] = vld1q_s32((int32_t const*)mem_addr);
    res.val[1] = vld1q_s32((int32_t const*)mem_addr + 4);

    return (__m256i)res;
    //return (__m256i)vld2q_s32((int32_t*)mem_addr);
}

//Store 256-bits of integer data from a into memory. mem_addr must be aligned on a 32-byte boundary or a general-protection exception may be generated.
FORCE_INLINE void _mm256_store_si256(__m256i *mem_addr, __m256i a)
{
    //vst2q_s32((int32_t*) mem_addr, (int32x4x2_t)a);
    vst1q_s32((int32_t*) mem_addr, a.val[0]);
    vst1q_s32(((int32_t*) mem_addr) + 4, a.val[1]);
}

//Return vector of type __m256i with all elements set to zero.
FORCE_INLINE __m256i _mm256_setzero_si256(void)
{
    int32x4x2_t res;
    
    res.val[0] = vdupq_n_s32(0);
    res.val[1] = vdupq_n_s32(0);
   
    return (__m256i)res;
}


//512 ########################################################################################################################################
//Store 512-bits of integer data from a into memory using a non-temporal memory hint. mem_addr must be aligned on a 64-byte boundary or a general-protection exception may be generated.
FORCE_INLINE void _mm512_stream_si512(__m512i *mem_addr, __m512i a)
{
    //*mem_addr = a;
    //vst4q_s32((int32_t*) mem_addr, (int32x4x4_t)a);
    vst1q_s32((int32_t*) mem_addr, a.val[0]);
    vst1q_s32(((int32_t*) mem_addr) + 4, a.val[1]);
    vst1q_s32(((int32_t*) mem_addr) + 8, a.val[2]);
    vst1q_s32(((int32_t*) mem_addr) + 12, a.val[3]);
}

//Convert packed 32-bit integers from a and b to packed 16-bit integers using signed saturation, and store the results in dst.
FORCE_INLINE __m512i _mm512_packs_epi32(__m512i a, __m512i b)
{
    int32x4x4_t res;

    res.val[0] = (int32x4_t)vcombine_s16(vqmovn_s32(a.val[0]), vqmovn_s32(b.val[0]));
    res.val[1] = (int32x4_t)vcombine_s16(vqmovn_s32(a.val[1]), vqmovn_s32(b.val[1]));

    res.val[2] = (int32x4_t)vcombine_s16(vqmovn_s32(a.val[2]), vqmovn_s32(b.val[2]));
    res.val[3] = (int32x4_t)vcombine_s16(vqmovn_s32(a.val[3]), vqmovn_s32(b.val[3]));

    return (__m512i)res;
}

//Compute the bitwise AND of 512 bits (representing integer data) in a and b, and store the result in dst.
FORCE_INLINE __m512i _mm512_and_si512(__m512i a, __m512i b)
{
    int32x4x4_t res;

    res.val[0] = vandq_s32((int32x4_t)a.val[0], (int32x4_t)b.val[0]);
    res.val[1] = vandq_s32((int32x4_t)a.val[1], (int32x4_t)b.val[1]);

    res.val[2] = vandq_s32((int32x4_t)a.val[2], (int32x4_t)b.val[2]);
    res.val[3] = vandq_s32((int32x4_t)a.val[3], (int32x4_t)b.val[3]);
   
    return (__m512i)res;
}

//Compute the bitwise OR of 512 bits (representing integer data) in a and b, and store the result in dst.
FORCE_INLINE __m512i _mm512_or_si512(__m512i a, __m512i b)
{
    int32x4x4_t res;

    res.val[0] = vorrq_s32((int32x4_t)a.val[0], (int32x4_t)b.val[0]);
    res.val[1] = vorrq_s32((int32x4_t)a.val[1], (int32x4_t)b.val[1]);

    res.val[2] = vorrq_s32((int32x4_t)a.val[2], (int32x4_t)b.val[2]);
    res.val[3] = vorrq_s32((int32x4_t)a.val[3], (int32x4_t)b.val[3]);
   
    return (__m512i)res;
}

//Store 256-bits of integer data from a into memory. mem_addr does not need to be aligned on any particular boundary.
FORCE_INLINE void _mm512_storeu_si512(void *mem_addr, __m512i a)
{
    //vst4q_s32((int32_t*) mem_addr, (int32x4x4_t)a);
    vst1q_s32((int32_t*) mem_addr, a.val[0]);
    vst1q_s32(((int32_t*) mem_addr) + 4, a.val[1]);
    vst1q_s32(((int32_t*) mem_addr) + 8, a.val[2]);
    vst1q_s32(((int32_t*) mem_addr) + 12, a.val[3]);
}

//Compute the bitwise NOT of 512 bits (representing integer data) in a and then AND with b, and store the result in dst.
FORCE_INLINE __m512i _mm512_andnot_si512(__m512i a, __m512i b)
{
    int32x4x4_t res;

    res.val[0] = vbicq_s32((int32x4_t)b.val[0], (int32x4_t)a.val[0]);
    res.val[1] = vbicq_s32((int32x4_t)b.val[1], (int32x4_t)a.val[1]);

    res.val[2] = vbicq_s32((int32x4_t)b.val[2], (int32x4_t)a.val[2]);
    res.val[3] = vbicq_s32((int32x4_t)b.val[3], (int32x4_t)a.val[3]);
   
    return (__m512i)res;
}

//Compare packed 32-bit integers in a and b, and store packed maximum values in dst.
FORCE_INLINE __m512i _mm512_max_epi32(__m512i a, __m512i b)
{
    int32x4x4_t res;

    res.val[0] = vmaxq_s32((int32x4_t)a.val[0], (int32x4_t)b.val[0]);
    res.val[1] = vmaxq_s32((int32x4_t)a.val[1], (int32x4_t)b.val[1]);

    res.val[2] = vmaxq_s32((int32x4_t)a.val[2], (int32x4_t)b.val[2]);
    res.val[3] = vmaxq_s32((int32x4_t)a.val[3], (int32x4_t)b.val[3]);

    return(__m512i)res;
}

//Add packed 32-bit integers in a and b, and store the results in dst.
FORCE_INLINE __m512i _mm512_add_epi32(__m512i a, __m512i b)
{
    int32x4x4_t res;

    res.val[0] = vaddq_s32((int32x4_t)a.val[0], (int32x4_t)b.val[0]);
    res.val[1] = vaddq_s32((int32x4_t)a.val[1], (int32x4_t)b.val[1]);

    res.val[2] = vaddq_s32((int32x4_t)a.val[2], (int32x4_t)b.val[2]);
    res.val[3] = vaddq_s32((int32x4_t)a.val[3], (int32x4_t)b.val[3]);

    return(__m512i)res;
}

//Load 512-bits of integer data from memory into dst. mem_addr does not need to be aligned on any particular boundary.
FORCE_INLINE __m512i _mm512_loadu_si512(const __m512i *mem_addr)
{
    //return (__m512i)vld4q_s32((int32_t*)mem_addr);
    int32x4x4_t res;

    res.val[0] = vld1q_s32((int32_t const*)mem_addr);
    res.val[1] = vld1q_s32((int32_t const*)mem_addr + 4);
    res.val[2] = vld1q_s32((int32_t const*)mem_addr + 8);
    res.val[3] = vld1q_s32((int32_t const*)mem_addr + 12);

    return (__m512i)res;
}

//Store 512-bits of integer data from a into memory. mem_addr must be aligned on a 64-byte boundary or a general-protection exception may be generated.
FORCE_INLINE void _mm512_store_si512(void *mem_addr, __m512i a)
{
    //vst4q_s32((int32_t*) mem_addr, (int32x4x4_t)a);
    vst1q_s32((int32_t*) mem_addr, a.val[0]);
    vst1q_s32(((int32_t*) mem_addr) + 4, a.val[1]);
    vst1q_s32(((int32_t*) mem_addr) + 8, a.val[2]);
    vst1q_s32(((int32_t*) mem_addr) + 12, a.val[3]);
}

//Return vector of type __m512i with all elements set to zero.
FORCE_INLINE __m512i _mm512_setzero_si512(void)
{
    int32x4x4_t res;
    
    res.val[0] = vdupq_n_s32(0);
    res.val[1] = vdupq_n_s32(0);

    res.val[2] = vdupq_n_s32(0);
    res.val[3] = vdupq_n_s32(0);
   
    return (__m512i)res;
}

//Set each packed 32-bit integer in dst to all ones or all zeros based on the value of the corresponding bit in k.
FORCE_INLINE __m512i _mm512_movm_epi32(__mmask16 k)
{
    int32x4x4_t res;

    (res.val[0])[0] = ((k & 0x0001) == 0) ? 0 : 0xffffffff;
    (res.val[0])[1] = ((k & 0x0002) == 0) ? 0 : 0xffffffff;
    (res.val[0])[2] = ((k & 0x0004) == 0) ? 0 : 0xffffffff;
    (res.val[0])[3] = ((k & 0x0008) == 0) ? 0 : 0xffffffff;

    (res.val[1])[0] = ((k & 0x0010) == 0) ? 0 : 0xffffffff;
    (res.val[1])[1] = ((k & 0x0020) == 0) ? 0 : 0xffffffff;
    (res.val[1])[2] = ((k & 0x0040) == 0) ? 0 : 0xffffffff;
    (res.val[1])[3] = ((k & 0x0080) == 0) ? 0 : 0xffffffff;

    (res.val[2])[0] = ((k & 0x0100) == 0) ? 0 : 0xffffffff;
    (res.val[2])[1] = ((k & 0x0200) == 0) ? 0 : 0xffffffff;
    (res.val[2])[2] = ((k & 0x0400) == 0) ? 0 : 0xffffffff;
    (res.val[2])[3] = ((k & 0x0800) == 0) ? 0 : 0xffffffff;

    (res.val[3])[0] = ((k & 0x1000) == 0) ? 0 : 0xffffffff;
    (res.val[3])[1] = ((k & 0x2000) == 0) ? 0 : 0xffffffff;
    (res.val[3])[2] = ((k & 0x4000) == 0) ? 0 : 0xffffffff;
    (res.val[3])[3] = ((k & 0x8000) == 0) ? 0 : 0xffffffff;
   
    return (__m512i)res;
}

//Compare packed 32-bit integers in a and b for greater-than, and store the results in mask vector k.
FORCE_INLINE __mmask16 _mm512_cmpgt_epi32_mask(__m512i a, __m512i b)
{
    __mmask16 res;
    res = 0x0000;

    if (((int32x4_t)a.val[0])[0] > ((int32x4_t)b.val[0])[0]) res = res | 0x0001;
    if (((int32x4_t)a.val[0])[1] > ((int32x4_t)b.val[0])[1]) res = res | 0x0002;
    if (((int32x4_t)a.val[0])[2] > ((int32x4_t)b.val[0])[2]) res = res | 0x0004;
    if (((int32x4_t)a.val[0])[3] > ((int32x4_t)b.val[0])[3]) res = res | 0x0008;

    if (((int32x4_t)a.val[1])[0] > ((int32x4_t)b.val[1])[0]) res = res | 0x0010;
    if (((int32x4_t)a.val[1])[1] > ((int32x4_t)b.val[1])[1]) res = res | 0x0020;
    if (((int32x4_t)a.val[1])[2] > ((int32x4_t)b.val[1])[2]) res = res | 0x0040;
    if (((int32x4_t)a.val[1])[3] > ((int32x4_t)b.val[1])[3]) res = res | 0x0080;

    if (((int32x4_t)a.val[2])[0] > ((int32x4_t)b.val[2])[0]) res = res | 0x0100;
    if (((int32x4_t)a.val[2])[1] > ((int32x4_t)b.val[2])[1]) res = res | 0x0200;
    if (((int32x4_t)a.val[2])[2] > ((int32x4_t)b.val[2])[2]) res = res | 0x0400;
    if (((int32x4_t)a.val[2])[3] > ((int32x4_t)b.val[2])[3]) res = res | 0x0800;

    if (((int32x4_t)a.val[3])[0] > ((int32x4_t)b.val[3])[0]) res = res | 0x1000;
    if (((int32x4_t)a.val[3])[1] > ((int32x4_t)b.val[3])[1]) res = res | 0x2000;
    if (((int32x4_t)a.val[3])[2] > ((int32x4_t)b.val[3])[2]) res = res | 0x4000;
    if (((int32x4_t)a.val[3])[3] > ((int32x4_t)b.val[3])[3]) res = res | 0x8000;


    return (__mmask16)res;
}

//Compare packed 32-bit integers in a and b for equality, and store the results in mask vector k.
FORCE_INLINE __mmask16 _mm512_cmpeq_epi32_mask(__m512i a, __m512i b)
{
    __mmask16 res;
    res = 0x0000;

    if (((int32x4_t)a.val[0])[0] == ((int32x4_t)b.val[0])[0]) res = res | 0x0001;
    if (((int32x4_t)a.val[0])[1] == ((int32x4_t)b.val[0])[1]) res = res | 0x0002;
    if (((int32x4_t)a.val[0])[2] == ((int32x4_t)b.val[0])[2]) res = res | 0x0004;
    if (((int32x4_t)a.val[0])[3] == ((int32x4_t)b.val[0])[3]) res = res | 0x0008;

    if (((int32x4_t)a.val[1])[0] == ((int32x4_t)b.val[1])[0]) res = res | 0x0010;
    if (((int32x4_t)a.val[1])[1] == ((int32x4_t)b.val[1])[1]) res = res | 0x0020;
    if (((int32x4_t)a.val[1])[2] == ((int32x4_t)b.val[1])[2]) res = res | 0x0040;
    if (((int32x4_t)a.val[1])[3] == ((int32x4_t)b.val[1])[3]) res = res | 0x0080;

    if (((int32x4_t)a.val[2])[0] == ((int32x4_t)b.val[2])[0]) res = res | 0x0100;
    if (((int32x4_t)a.val[2])[1] == ((int32x4_t)b.val[2])[1]) res = res | 0x0200;
    if (((int32x4_t)a.val[2])[2] == ((int32x4_t)b.val[2])[2]) res = res | 0x0400;
    if (((int32x4_t)a.val[2])[3] == ((int32x4_t)b.val[2])[3]) res = res | 0x0800;

    if (((int32x4_t)a.val[3])[0] == ((int32x4_t)b.val[3])[0]) res = res | 0x1000;
    if (((int32x4_t)a.val[3])[1] == ((int32x4_t)b.val[3])[1]) res = res | 0x2000;
    if (((int32x4_t)a.val[3])[2] == ((int32x4_t)b.val[3])[2]) res = res | 0x4000;
    if (((int32x4_t)a.val[3])[3] == ((int32x4_t)b.val[3])[3]) res = res | 0x8000;


    return (__mmask16)res;
}

//Blend packed 32-bit integers from a and b using control mask k, and store the results in dst.
FORCE_INLINE __m512i _mm512_mask_blend_epi32(__mmask16 k, __m512i a, __m512i b)
{
    int32x4x4_t res;

    (res.val[0])[0] = (k & 0x0001) ? ((int32x4_t)b.val[0])[0] : ((int32x4_t)a.val[0])[0];
    (res.val[0])[1] = (k & 0x0002) ? ((int32x4_t)b.val[0])[1] : ((int32x4_t)a.val[0])[1];
    (res.val[0])[2] = (k & 0x0004) ? ((int32x4_t)b.val[0])[2] : ((int32x4_t)a.val[0])[2];
    (res.val[0])[3] = (k & 0x0008) ? ((int32x4_t)b.val[0])[3] : ((int32x4_t)a.val[0])[3];

    (res.val[1])[0] = (k & 0x0010) ? ((int32x4_t)b.val[1])[0] : ((int32x4_t)a.val[1])[0];
    (res.val[1])[1] = (k & 0x0020) ? ((int32x4_t)b.val[1])[1] : ((int32x4_t)a.val[1])[1];
    (res.val[1])[2] = (k & 0x0040) ? ((int32x4_t)b.val[1])[2] : ((int32x4_t)a.val[1])[2];
    (res.val[1])[3] = (k & 0x0080) ? ((int32x4_t)b.val[1])[3] : ((int32x4_t)a.val[1])[3];

    (res.val[2])[0] = (k & 0x0100) ? ((int32x4_t)b.val[2])[0] : ((int32x4_t)a.val[2])[0];
    (res.val[2])[1] = (k & 0x0200) ? ((int32x4_t)b.val[2])[1] : ((int32x4_t)a.val[2])[1];
    (res.val[2])[2] = (k & 0x0400) ? ((int32x4_t)b.val[2])[2] : ((int32x4_t)a.val[2])[2];
    (res.val[2])[3] = (k & 0x0800) ? ((int32x4_t)b.val[2])[3] : ((int32x4_t)a.val[2])[3];

    (res.val[3])[0] = (k & 0x1000) ? ((int32x4_t)b.val[3])[0] : ((int32x4_t)a.val[3])[0];
    (res.val[3])[1] = (k & 0x2000) ? ((int32x4_t)b.val[3])[1] : ((int32x4_t)a.val[3])[1];
    (res.val[3])[2] = (k & 0x4000) ? ((int32x4_t)b.val[3])[2] : ((int32x4_t)a.val[3])[2];
    (res.val[3])[3] = (k & 0x8000) ? ((int32x4_t)b.val[3])[3] : ((int32x4_t)a.val[3])[3];

    return (__m512i)res;
}

//Shuffle 32-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst.
FORCE_INLINE __m512i _mm512_permutex2var_epi32(__m512i a, __m512i idx, __m512i b)
{
    int32x4x4_t res;

    //res = a;
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
