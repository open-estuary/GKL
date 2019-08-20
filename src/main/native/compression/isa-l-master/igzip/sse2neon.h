#ifndef SSE2NEON_H
#define SSE2NEON_H

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
//#include <arm_acle.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
//#include "pch.h"
//#include "misc.h"

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


FORCE_INLINE __m256d _mm256_set_pd(double e3, double e2, double e1, double e0);
FORCE_INLINE __m256 _mm256_set_ps(float e7, float e6, float e5, float e4, float e3, float e2, float e1, float e0);

//--------------------------------------------------------------------------------------------------first phase----------------------------------------------------------------------------------------------
//Add packed 8-bit integers in a and b using saturation, and store the results in dst
FORCE_INLINE __m64 _mm_adds_pi8(__m64 a, __m64 b)
{
	//int8x8_t vadd_s8 (int8x8_t a, int8x8_t b)
    return (__m64)vadd_s8((int8x8_t)a, (int8x8_t)b);
}

//Set packed 32-bit integers in dst with the supplied values.
FORCE_INLINE __m128i _mm_set_epi32 (int e3, int e2, int e1, int e0)
{
	int ptr[] = {e0, e1, e2, e3};
	
	//int32x4_t   vld1q_s32(__transfersize(4) int32_t const * ptr);
	return (__m128i)vld1q_s32(ptr);
}

//Set packed 32-bit integers in dst with the supplied values.
FORCE_INLINE __m64 _mm_set_pi32 (int e1, int e0)
{
	//int32x2_t   vld1_s32(__transfersize(2) int32_t const * ptr);
	int ptr[] = {e0, e1};
	
	return (__m64)vld1_s32(ptr);
}

//Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst.
FORCE_INLINE __m64 _mm_adds_pu8(__m64 a, __m64 b)
{
	//uint8x8_t vqadd_u8 (uint8x8_t a, uint8x8_t b)
    return (__m64)vqadd_u8((uint8x8_t)a, (uint8x8_t)b);
}

//Compute the bitwise AND of 64 bits (representing integer data) in a and b, and store the result in dst.
FORCE_INLINE __m64 _mm_and_si64(__m64 a, __m64 b)
{
	//int64x1_t vand_s64 (int64x1_t a, int64x1_t b)
	return (__m64)vand_s64((int64x1_t)a, (int64x1_t)b);
}

//Compare packed 16-bit integers in a and b for equality, and store the results in dst.
FORCE_INLINE __m128i _mm_cmpeq_epi16(__m128i a, __m128i b)
{
	//uint16x8_t vceqq_s16 (int16x8_t a, int16x8_t b)
	return(__m128i)vceqq_s16((int16x8_t)a, (int16x8_t)b);
}

//Compare packed 8-bit integers in a and b for equality, and store the results in dst.
FORCE_INLINE __m64 _mm_cmpeq_pi8(__m64 a, __m64 b)
{
	//uint8x8_t vceq_s8 (int8x8_t a, int8x8_t b)
	return (__m64)vceq_s8((int8x8_t)a, (int8x8_t)b);
}

//Copy 32-bit integer a to the lower elements of dst, and zero the upper element of dst.
FORCE_INLINE __m64 _mm_cvtsi32_si64(int a)
{
	int32x2_t res = vdup_n_s32(0);
	
	//int32x2_t vset_lane_s32 (int32_t a, int32x2_t v, const int lane)
	return (__m64)vset_lane_s32((int32_t)a, res, 0);
}

//Empty the MMX state, which marks the x87 FPU registers as available for use by x87 instructions. 
//This instruction must be used at the end of all MMX technology procedures.
FORCE_INLINE void _mm_empty (void)
{
}

//Copy a to dst, and insert the 16-bit integer i into dst at the location specified by imm8.
FORCE_INLINE __m128i _mm_insert_epi16(__m128i a, int i, const int imm8)
{
	//int16x8_t vsetq_lane_s16 (int16_t a, int16x8_t v,__constrange(0,7) int lane)
	//return (__m128i)vsetq_lane_s16((int16_t)i,(int16x8_t)a,imm8);
    switch(imm8)
    {
        //不需要break，匹配后直接return
        case 0:
            return (__m128i)vsetq_lane_s16((int16_t)i, (int16x8_t)a,0);
        case 1:
            return (__m128i)vsetq_lane_s16((int16_t)i, (int16x8_t)a,1);
        case 2:
            return (__m128i)vsetq_lane_s16((int16_t)i, (int16x8_t)a,2);
        case 3:
            return (__m128i)vsetq_lane_s16((int16_t)i, (int16x8_t)a,3);
        case 4:
            return (__m128i)vsetq_lane_s16((int16_t)i, (int16x8_t)a,4);
        case 5:
            return (__m128i)vsetq_lane_s16((int16_t)i, (int16x8_t)a,5);
        case 6:
            return (__m128i)vsetq_lane_s16((int16_t)i, (int16x8_t)a,6);
        case 7:
            return (__m128i)vsetq_lane_s16((int16_t)i, (int16x8_t)a,7);
		default:
			return (__m128i)vsetq_lane_s16((int16_t)i, (int16x8_t)a,0);
    }
}

//Create mask from the most significant bit of each 8-bit element in a, and store the result in dst.
FORCE_INLINE int _mm_movemask_pi8(__m64 a)
{
	uint8x8_t res = (uint8x8_t)a;
	int8_t xr[] = {-7, -6, -5, -4, -3, -2, -1, 0}; 
	//const int8_t __attribute__((aligned(16))) xr[8] = { -7, -6, -5, -4, -3, -2, -1, 0 }; 
	uint8x8_t mask_and = vdup_n_u8(0x80); //uint8x8_t vdup_n_u8(uint8_t)
	int8x8_t mask_shift = vld1_s8(xr);    //int8x8_t vld1_s8(int8_t const* ptr)
	
	res = vand_u8(res, mask_and);         //uint8x8_t vand_u8(uint8x8_t,uint8x8_t)
	res = vshl_u8(res, mask_shift);       //uint8x8_t vshl_u8(uint8x8_t,int8x8_t) negative-shift right

	res = vpadd_u8(res, res);             //uint8x8_t vpadd_u8(uint8x8_t,uint8x8_t)  ADDP  add adjacent vector elements 
	res = vpadd_u8(res, res);
	res = vpadd_u8(res, res);

	return (res[0] & 0xFF);
}


FORCE_INLINE int _mm_movemask_epi16(__m128i a)
{
	uint16x8_t res = (uint16x8_t)a;
	int16_t xr[] = {-15, -14, -13, -12, -11, -10, -9, -8}; 
	//const int8_t __attribute__((aligned(16))) xr[8] = { -15, -14, -13, -12, -11, -10, -9, -8}; 
	uint16x8_t mask_and = vdupq_n_u16(0x8000); //uint16x8_t vdupq_n_u16 (uint16_t value)
	int16x8_t mask_shift = vld1q_s16(xr);      //int16x8_t vld1q_s16 (int16_t const * ptr)
	
	res = vandq_u16(res, mask_and);            //uint16x8_t vandq_u16 (uint16x8_t a, uint16x8_t b)
	res = vshlq_u16(res, mask_shift);          //uint16x8_t vshlq_u16 (uint16x8_t a, int16x8_t b) negative-shift right

	res = vpaddq_u16(res, res);                 //uint16x8_t vpaddq_u16 (uint16x8_t a, uint16x8_t b)  ADDP  add adjacent vector elements 
	res = vpaddq_u16(res, res);
	res = vpaddq_u16(res, res);

	return (res[0] & 0xFF);
}
//Copy the lower 64-bit integer in a to dst.
FORCE_INLINE __m64 _mm_movepi64_pi64(__m128i a)
{
	//int64_t   vgetq_lane_s64(int64x2_t vec, __constrange(0,1) int lane);
	return (__m64)vgetq_lane_s64((int64x2_t)a, 0);
}

//Copy the 64-bit integer a to the lower element of dst, and zero the upper element.
FORCE_INLINE __m128i _mm_movpi64_epi64(__m64 a)
{
	//int64x2_t vdupq_n_s64 (int64_t value)
	int64x2_t res = vdupq_n_s64((int64_t)0);
	
	//int64x2_t   vsetq_lane_s64(int64_t value, int64x2_t vec, __constrange(0,1) int lane);
	return (__m128i)vsetq_lane_s64((int64_t)a, res, 0);
}

//Compute the bitwise OR of 64 bits (representing integer data) in a and b, and store the result in dst
FORCE_INLINE __m64 _mm_or_si64(__m64 a, __m64 b)
{
	//int64x1_t  vorr_s64(int64x1_t a, int64x1_t b);
	return (__m64)vorr_s64((int64x1_t)a, (int64x1_t)b);
}

//Broadcast 64-bit integer a to all elements of dst.
FORCE_INLINE __m128i _mm_set1_epi64(__m64 a)
{
	//int64x2_t   vdupq_n_s64(int64_t value);
	return (__m128i)vdupq_n_s64((int64_t)a);
}

//Broadcast 32-bit integer a to all elements of dst.
FORCE_INLINE __m64 _mm_set1_pi32(int a)
{
	//int32x2_t   vdup_n_s32(int32_t value); 
	return (__m64)vdup_n_s32((int32_t)a);
}

//Broadcast 8-bit integer a to all elements of dst.
FORCE_INLINE __m64 _mm_set1_pi8(char a)
{
	//int8x8_t    vdup_n_s8(int8_t value); 
	return (__m64)vdup_n_s8((int8_t)a);
}

//Return vector of type __m64 with all elements set to zero.
FORCE_INLINE __m64 _mm_setzero_si64(void)
{
	//int64x1_t   vdup_n_s64(int64_t value); 
	return (__m64)vdup_n_s64((int64_t)0);
}

//Shift packed 32-bit integers in a left by imm8 while shifting in zeros, and store the results in dst.
FORCE_INLINE __m128i _mm_slli_epi32(__m128i a, const int imm8)
{
	//if(imm8 > 31 || imm8 < 0)
		//return (__m128i)vdupq_n_s32((int32_t)0);
	
    //int32x4_t  vshlq_n_s32(int32x4_t a, __constrange(0,31) int b);
	//return (__m128i)vshlq_n_s32((int32x4_t)a, imm8);
    switch(imm8)
    {
        case 0:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 0);
        case 1:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 1);
        case 2:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 2);
        case 3:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 3);
        case 4:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 4);
        case 5:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 5);
        case 6:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 6);
        case 7:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 7);
        case 8:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 8);
        case 9:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 9);
        case 10:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 10);
        case 11:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 11);
        case 12:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 12);
        case 13:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 13);
        case 14:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 14);
        case 15:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 15);
        case 16:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 16);
        case 17:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 17);
        case 18:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 18);
        case 19:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 19);
        case 20:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 20);
        case 21:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 21);
        case 22:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 22);
        case 23:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 23);
        case 24:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 24);
        case 25:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 25);
        case 26:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 26);
        case 27:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 27);
        case 28:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 28);
        case 29:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 29);
        case 30:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 30);
        case 31:
            return (__m128i)vshlq_n_s32((int32x4_t)a, 31);
        default:
            return (__m128i)vdupq_n_s32((int32_t)0);
    }

}

//Shift 64-bit integer a left by imm8 while shifting in zeros, and store the result in dst.
FORCE_INLINE __m64 _mm_slli_si64(__m64 a, const int imm8)
{
	//if(imm8 > 63 || imm8 < 0)
		//return (__m64)vdup_n_s64((int64_t)0);
		
    //int64x1_t  vshl_n_s64(int64x1_t a, __constrange(0,63) int b);
	//return (__m64)vshl_n_s64((int64x1_t)a, imm8);
    switch(imm8)
    {
        case 0:
            return (__m64)vshl_n_s64((int64x1_t)a, 0);
        case 1:
            return (__m64)vshl_n_s64((int64x1_t)a, 1);
        case 2:
            return (__m64)vshl_n_s64((int64x1_t)a, 2);
        case 3:
            return (__m64)vshl_n_s64((int64x1_t)a, 3);
        case 4:
            return (__m64)vshl_n_s64((int64x1_t)a, 4);
        case 5:
            return (__m64)vshl_n_s64((int64x1_t)a, 5);
        case 6:
            return (__m64)vshl_n_s64((int64x1_t)a, 6);
        case 7:
            return (__m64)vshl_n_s64((int64x1_t)a, 7);
        case 8:
            return (__m64)vshl_n_s64((int64x1_t)a, 8);
        case 9:
            return (__m64)vshl_n_s64((int64x1_t)a, 9);
        case 10:
            return (__m64)vshl_n_s64((int64x1_t)a, 10);
        case 11:
            return (__m64)vshl_n_s64((int64x1_t)a, 11);
        case 12:
            return (__m64)vshl_n_s64((int64x1_t)a, 12);
        case 13:
            return (__m64)vshl_n_s64((int64x1_t)a, 13);
        case 14:
            return (__m64)vshl_n_s64((int64x1_t)a, 14);
        case 15:
            return (__m64)vshl_n_s64((int64x1_t)a, 15);
        case 16:
            return (__m64)vshl_n_s64((int64x1_t)a, 16);
        case 17:
            return (__m64)vshl_n_s64((int64x1_t)a, 17);
        case 18:
            return (__m64)vshl_n_s64((int64x1_t)a, 18);
        case 19:
            return (__m64)vshl_n_s64((int64x1_t)a, 19);
        case 20:
            return (__m64)vshl_n_s64((int64x1_t)a, 20);
        case 21:
            return (__m64)vshl_n_s64((int64x1_t)a, 21);
        case 22:
            return (__m64)vshl_n_s64((int64x1_t)a, 22);
        case 23:
            return (__m64)vshl_n_s64((int64x1_t)a, 23);
        case 24:
            return (__m64)vshl_n_s64((int64x1_t)a, 24);
        case 25:
            return (__m64)vshl_n_s64((int64x1_t)a, 25);
        case 26:
            return (__m64)vshl_n_s64((int64x1_t)a, 26);
        case 27:
            return (__m64)vshl_n_s64((int64x1_t)a, 27);
        case 28:
            return (__m64)vshl_n_s64((int64x1_t)a, 28);
        case 29:
            return (__m64)vshl_n_s64((int64x1_t)a, 29);
        case 30:
            return (__m64)vshl_n_s64((int64x1_t)a, 30);
        case 31:
            return (__m64)vshl_n_s64((int64x1_t)a, 31);
        case 32:
            return (__m64)vshl_n_s64((int64x1_t)a, 32);
        case 33:
            return (__m64)vshl_n_s64((int64x1_t)a, 33);
        case 34:
            return (__m64)vshl_n_s64((int64x1_t)a, 34);
        case 35:
            return (__m64)vshl_n_s64((int64x1_t)a, 35);
        case 36:
            return (__m64)vshl_n_s64((int64x1_t)a, 36);
        case 37:
            return (__m64)vshl_n_s64((int64x1_t)a, 37);
        case 38:
            return (__m64)vshl_n_s64((int64x1_t)a, 38);
        case 39:
            return (__m64)vshl_n_s64((int64x1_t)a, 39);
        case 40:
            return (__m64)vshl_n_s64((int64x1_t)a, 40);
        case 41:
            return (__m64)vshl_n_s64((int64x1_t)a, 41);
        case 42:
            return (__m64)vshl_n_s64((int64x1_t)a, 42);
        case 43:
            return (__m64)vshl_n_s64((int64x1_t)a, 43);
        case 44:
            return (__m64)vshl_n_s64((int64x1_t)a, 44);
        case 45:
            return (__m64)vshl_n_s64((int64x1_t)a, 45);
        case 46:
            return (__m64)vshl_n_s64((int64x1_t)a, 46);
        case 47:
            return (__m64)vshl_n_s64((int64x1_t)a, 47);
        case 48:
            return (__m64)vshl_n_s64((int64x1_t)a, 48);
        case 49:
            return (__m64)vshl_n_s64((int64x1_t)a, 49);
        case 50:
            return (__m64)vshl_n_s64((int64x1_t)a, 50);
        case 51:
            return (__m64)vshl_n_s64((int64x1_t)a, 51);
        case 52:
            return (__m64)vshl_n_s64((int64x1_t)a, 52);
        case 53:
            return (__m64)vshl_n_s64((int64x1_t)a, 53);
        case 54:
            return (__m64)vshl_n_s64((int64x1_t)a, 54);
        case 55:
            return (__m64)vshl_n_s64((int64x1_t)a, 55);
        case 56:
            return (__m64)vshl_n_s64((int64x1_t)a, 56);
        case 57:
            return (__m64)vshl_n_s64((int64x1_t)a, 57);
        case 58:
            return (__m64)vshl_n_s64((int64x1_t)a, 58);
        case 59:
            return (__m64)vshl_n_s64((int64x1_t)a, 59);
        case 60:
            return (__m64)vshl_n_s64((int64x1_t)a, 60);
        case 61:
            return (__m64)vshl_n_s64((int64x1_t)a, 61);
        case 62:
            return (__m64)vshl_n_s64((int64x1_t)a, 62);
        case 63:
            return (__m64)vshl_n_s64((int64x1_t)a, 63);
        default:
            return (__m64)vdup_n_s64((int64_t)0);
    }
}

//Shift packed 32-bit integers in a right by imm8 while shifting in zeros, and store the results in dst.
FORCE_INLINE __m128i _mm_srli_epi32(__m128i a, const int imm8)
{
	//if(imm8 > 31 || imm8 < 0)
		//return (__m128i)vdupq_n_s32((int32_t)0);

	//int32x4_t  vshrq_n_s32(int32x4_t a, __constrange(1,32) int b);
	//return (__m128i)vshrq_n_u32((uint32x4_t)a, imm8);
    switch(imm8)
    {
        case 0:
            return (__m128i)a;
        case 1:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 1);
        case 2:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 2);
        case 3:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 3);
        case 4:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 4);
        case 5:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 5);
        case 6:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 6);
        case 7:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 7);
        case 8:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 8);
        case 9:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 9);
        case 10:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 10);
        case 11:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 11);
        case 12:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 12);
        case 13:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 13);
        case 14:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 14);
        case 15:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 15);
        case 16:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 16);
        case 17:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 17);
        case 18:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 18);
        case 19:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 19);
        case 20:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 20);
        case 21:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 21);
        case 22:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 22);
        case 23:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 23);
        case 24:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 24);
        case 25:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 25);
        case 26:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 26);
        case 27:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 27);
        case 28:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 28);
        case 29:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 29);
        case 30:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 30);
        case 31:
            return (__m128i)vshrq_n_u32((uint32x4_t)a, 31);
        default:
            return (__m128i)vdupq_n_s32((int32_t)0);
    }

}

//Shift 64-bit integer a right by imm8 while shifting in zeros, and store the result in dst.
FORCE_INLINE __m64 _mm_srli_si64(__m64 a, const int imm8)
{
	//int64x1_t  vshr_n_u64(uint64x1_t a, __constrange(1,64) int b);
    switch(imm8)
    {
        case 0:
            return (__m64)a;
        case 1:
            return (__m64)vshr_n_u64((uint64x1_t)a, 1);
        case 2:
            return (__m64)vshr_n_u64((uint64x1_t)a, 2);
        case 3:
            return (__m64)vshr_n_u64((uint64x1_t)a, 3);
        case 4:
            return (__m64)vshr_n_u64((uint64x1_t)a, 4);
        case 5:
            return (__m64)vshr_n_u64((uint64x1_t)a, 5);
        case 6:
            return (__m64)vshr_n_u64((uint64x1_t)a, 6);
        case 7:
            return (__m64)vshr_n_u64((uint64x1_t)a, 7);
        case 8:
            return (__m64)vshr_n_u64((uint64x1_t)a, 8);
        case 9:
            return (__m64)vshr_n_u64((uint64x1_t)a, 9);
        case 10:
            return (__m64)vshr_n_u64((uint64x1_t)a, 10);
        case 11:
            return (__m64)vshr_n_u64((uint64x1_t)a, 11);
        case 12:
            return (__m64)vshr_n_u64((uint64x1_t)a, 12);
        case 13:
            return (__m64)vshr_n_u64((uint64x1_t)a, 13);
        case 14:
            return (__m64)vshr_n_u64((uint64x1_t)a, 14);
        case 15:
            return (__m64)vshr_n_u64((uint64x1_t)a, 15);
        case 16:
            return (__m64)vshr_n_u64((uint64x1_t)a, 16);
        case 17:
            return (__m64)vshr_n_u64((uint64x1_t)a, 17);
        case 18:
            return (__m64)vshr_n_u64((uint64x1_t)a, 18);
        case 19:
            return (__m64)vshr_n_u64((uint64x1_t)a, 19);
        case 20:
            return (__m64)vshr_n_u64((uint64x1_t)a, 20);
        case 21:
            return (__m64)vshr_n_u64((uint64x1_t)a, 21);
        case 22:
            return (__m64)vshr_n_u64((uint64x1_t)a, 22);
        case 23:
            return (__m64)vshr_n_u64((uint64x1_t)a, 23);
        case 24:
            return (__m64)vshr_n_u64((uint64x1_t)a, 24);
        case 25:
            return (__m64)vshr_n_u64((uint64x1_t)a, 25);
        case 26:
            return (__m64)vshr_n_u64((uint64x1_t)a, 26);
        case 27:
            return (__m64)vshr_n_u64((uint64x1_t)a, 27);
        case 28:
            return (__m64)vshr_n_u64((uint64x1_t)a, 28);
        case 29:
            return (__m64)vshr_n_u64((uint64x1_t)a, 29);
        case 30:
            return (__m64)vshr_n_u64((uint64x1_t)a, 30);
        case 31:
            return (__m64)vshr_n_u64((uint64x1_t)a, 31);
        case 32:
            return (__m64)vshr_n_u64((uint64x1_t)a, 32);
        case 33:
            return (__m64)vshr_n_u64((uint64x1_t)a, 33);
        case 34:
            return (__m64)vshr_n_u64((uint64x1_t)a, 34);
        case 35:
            return (__m64)vshr_n_u64((uint64x1_t)a, 35);
        case 36:
            return (__m64)vshr_n_u64((uint64x1_t)a, 36);
        case 37:
            return (__m64)vshr_n_u64((uint64x1_t)a, 37);
        case 38:
            return (__m64)vshr_n_u64((uint64x1_t)a, 38);
        case 39:
            return (__m64)vshr_n_u64((uint64x1_t)a, 39);
        case 40:
            return (__m64)vshr_n_u64((uint64x1_t)a, 40);
        case 41:
            return (__m64)vshr_n_u64((uint64x1_t)a, 41);
        case 42:
            return (__m64)vshr_n_u64((uint64x1_t)a, 42);
        case 43:
            return (__m64)vshr_n_u64((uint64x1_t)a, 43);
        case 44:
            return (__m64)vshr_n_u64((uint64x1_t)a, 44);
        case 45:
            return (__m64)vshr_n_u64((uint64x1_t)a, 45);
        case 46:
            return (__m64)vshr_n_u64((uint64x1_t)a, 46);
        case 47:
            return (__m64)vshr_n_u64((uint64x1_t)a, 47);
        case 48:
            return (__m64)vshr_n_u64((uint64x1_t)a, 48);
        case 49:
            return (__m64)vshr_n_u64((uint64x1_t)a, 49);
        case 50:
            return (__m64)vshr_n_u64((uint64x1_t)a, 50);
        case 51:
            return (__m64)vshr_n_u64((uint64x1_t)a, 51);
        case 52:
            return (__m64)vshr_n_u64((uint64x1_t)a, 52);
        case 53:
            return (__m64)vshr_n_u64((uint64x1_t)a, 53);
        case 54:
            return (__m64)vshr_n_u64((uint64x1_t)a, 54);
        case 55:
            return (__m64)vshr_n_u64((uint64x1_t)a, 55);
        case 56:
            return (__m64)vshr_n_u64((uint64x1_t)a, 56);
        case 57:
            return (__m64)vshr_n_u64((uint64x1_t)a, 57);
        case 58:
            return (__m64)vshr_n_u64((uint64x1_t)a, 58);
        case 59:
            return (__m64)vshr_n_u64((uint64x1_t)a, 59);
        case 60:
            return (__m64)vshr_n_u64((uint64x1_t)a, 60);
        case 61:
            return (__m64)vshr_n_u64((uint64x1_t)a, 61);
        case 62:
            return (__m64)vshr_n_u64((uint64x1_t)a, 62);
        case 63:
            return (__m64)vshr_n_u64((uint64x1_t)a, 63);
		default:
			return (__m64)vdup_n_s64((int64_t)0);
    }
}
/*#define _mm_srli_si64(a,imm)\
({\
 __m64 ret;\
 int imm8=imm&0xff;\
 if(imm8>63)\
	ret=(__m64)vdup_n_s64((int64_t)0);\
 else\
	ret=(__m64)vshr_n_u64((uint64x1_t)a,(imm8));\
 ret;\
 })*/

/*#define _mm_srli_si64(a,imm)\
({\
 __m64 ret;\
	ret=(__m64)vshr_n_u64((uint64x1_t)a,(imm));\
 ret;\
 })*/

//Compare packed 8-bit integers in a and b for greater-than, and store the results in dst.
FORCE_INLINE __m128i _mm_cmplt_epi16(__m128i a, __m128i b)
 {
	//uint16x8_t vcltq_s16(int16x8_t a, int16x8_t b);     // VCGT.S16 q0, q0, q0
	return (__m128i)vcltq_s16((int16x8_t)a, (int16x8_t)b);
 }
 
 
/*Operation
dst[15:0] := (a >> (imm8[1:0] * 16))[15:0]
dst[31:16] := (a >> (imm8[3:2] * 16))[15:0]
dst[47:32] := (a >> (imm8[5:4] * 16))[15:0]
dst[63:48] := (a >> (imm8[7:6] * 16))[15:0]
dst[127:64] := a[127:64]*/
//Shuffle 16-bit integers in the low 64 bits of a using the control in imm8. Store the results in the low 64 bits of dst, with the high 64 bits being copied from from a to dst.
FORCE_INLINE __m128i _mm_shufflelo_epi16(__m128i a, int imm8)
{
	int16x8_t res;
	int16x8_t tmp = (int16x8_t)a;
	
	res[0] = tmp[imm8 & 0x03];
	res[1] = tmp[(imm8 >> 2) & 0x03];
	res[2] = tmp[(imm8 >> 4) & 0x03];
	res[3] = tmp[(imm8 >> 6) & 0x03];
	
	return (__m128i)vsetq_lane_s64(vgetq_lane_s64((int64x2_t)tmp, 1), (int64x2_t)res, 1);
}
 
FORCE_INLINE __m128i _mm_cmpgt_epu8(__m128i a, __m128i b)
{
	//uint8x16_t vcgtq_u8(uint8x16_t a, uint8x16_t b);    // VCGT.U8 q0, q0, q0 
	return (__m128i)vcgtq_u8((uint8x16_t)a, ( uint8x16_t)b);
}

FORCE_INLINE __m128i _mm_cmpeq_epu8(__m128i a, __m128i b)
{
	//uint8x16_t vceqq_u8(uint8x16_t a, uint8x16_t b);    // VCEQ.I8 q0, q0, q0
	return (__m128i)vceqq_u8((uint8x16_t)a, ( uint8x16_t)b);
}

FORCE_INLINE __m128i _mm_srli_epu8(__m128i a, const int imm8)
{
	//if(imm8 > 7 || imm8 < 0)
		//return (__m128i)vdupq_n_u8((uint8_t)0);

	//return (__m128i)vshrq_n_u8((uint8x16_t)a, imm8);
    switch(imm8)
    {
        case 0: 
            return (__m128i)a;
        case 1: 
            return (__m128i)vshrq_n_u8((uint8x16_t)a, 1);
        case 2: 
            return (__m128i)vshrq_n_u8((uint8x16_t)a, 2);
        case 3: 
            return (__m128i)vshrq_n_u8((uint8x16_t)a, 3);
        case 4: 
            return (__m128i)vshrq_n_u8((uint8x16_t)a, 4);
        case 5: 
            return (__m128i)vshrq_n_u8((uint8x16_t)a, 5);
        case 6: 
            return (__m128i)vshrq_n_u8((uint8x16_t)a, 6);
        case 7: 
            return (__m128i)vshrq_n_u8((uint8x16_t)a, 7);
        default:
            return (__m128i)vdupq_n_u8((uint8_t)0);
    }
}

FORCE_INLINE __m128i _mm_cmplt_epu8(__m128i a, __m128i b)
{
	//uint8x16_t vcltq_u8(uint8x16_t a, uint8x16_t b);    // VCGT.U8 q0, q0, q0 
	return (__m128i)vcltq_u8((uint8x16_t)a, ( uint8x16_t)b);
}

/*  expands to the following value */
#define _MM_SHUFFLE(z, y, x, w)    ( (z<<6) | (y<<4) | (x<<2) | w )


/***************************************************************************
 *                max and min
 ***************************************************************************/

FORCE_INLINE __m128i _mm_max_epu8 (__m128i a, __m128i b)
{
	return (__m128i)vmaxq_u8((uint8x16_t) a, (uint8x16_t) b);
}

FORCE_INLINE __m128i _mm_max_epi16 (__m128i a, __m128i b)
{
	return (__m128i)vmaxq_s16((int16x8_t) a, (int16x8_t) b);
}

/* Return value
 * A 128-bit parameter that can be defined with the following equations:
 * r0 := (a0 > b0) ? a0 : b0
 * r1 := (a1 > b1) ? a1 : b1
 * r2 := (a2 > b2) ? a2 : b2
 * r3 := (a3 > b3) ? a3 : b3
 * */
FORCE_INLINE __m128i _mm_max_epi32(__m128i a, __m128i b)
{
	return vmaxq_s32(a, b);
}


/* todo: when the input data contain the NaN. => different behave
	BUT, in actual use, NaN ?
Need MORE tests?
 */
FORCE_INLINE __m128 _mm_max_ps(__m128 a, __m128 b)
{
	return vmaxq_f32(a, b);
}

FORCE_INLINE __m128i _mm_min_epu8 (__m128i a, __m128i b)
{
	return (__m128i)vminq_u8((uint8x16_t) a, (uint8x16_t) b);
}

FORCE_INLINE __m128i _mm_min_epi16(__m128i a, __m128i b)
{
	return (__m128i)vminq_s16((int16x8_t)a, (int16x8_t)b);
}

/* Return value
 * A 128-bit parameter that can be defined with the following equations:
 * r0 := (a0 < b0) ? a0 : b0
 * r1 := (a1 < b1) ? a1 : b1
 * r2 := (a2 < b2) ? a2 : b2
 * r3 := (a3 < b3) ? a3 : b3
 * */
FORCE_INLINE __m128i _mm_min_epi32(__m128i a, __m128i b)
{
	return vminq_s32(a, b);
}

/* todo: when the input data contain the NaN. => different behave
	BUT, in actual use, NaN ?
Need MORE tests?
 */
FORCE_INLINE __m128 _mm_min_ps(__m128 a, __m128 b)
{
	return vminq_f32(a, b);
}

/***************************************************************************
 *                add and sub
 ***************************************************************************/
/* Subtracts the 16 unsigned 8-bit integers of b from the 16 unsigned 8-bit integers of a and saturates.
 * r0 := UnsignedSaturate(a0 - b0)
 * r1 := UnsignedSaturate(a1 - b1)
 * ...
 * r15 := UnsignedSaturate(a15 - b15)
 * */
FORCE_INLINE __m128i _mm_subs_epu8(__m128i a, __m128i b)
{
	return (__m128i)vqsubq_u8((uint8x16_t) a, (uint8x16_t) b);
}

FORCE_INLINE __m128i _mm_adds_epu8(__m128i a, __m128i b)
{
	return (__m128i)vqaddq_u8((uint8x16_t) a, (uint8x16_t) b);
}

FORCE_INLINE __m128i _mm_add_epi16(__m128i a, __m128i b)
{
	return (__m128i)vaddq_s16((int16x8_t)a, (int16x8_t)b);
}

FORCE_INLINE __m128i _mm_sub_epi16(__m128i a, __m128i b)
{
	return (__m128i)vsubq_s16((int16x8_t) a, (int16x8_t) b);
}

FORCE_INLINE __m128i _mm_adds_epu16(__m128i a, __m128i b)
{
	return (__m128i)vqaddq_u16((uint16x8_t) a, (uint16x8_t) b);
}

FORCE_INLINE __m128i _mm_subs_epu16(__m128i a, __m128i b)
{
	return (__m128i)vqsubq_u16((uint16x8_t) a, (uint16x8_t) b);
}

/* Adds the 8 signed 16-bit integers in a to the 8 signed 16-bit integers in b and saturates.
 * r0 := SignedSaturate(a0 + b0)
 * r1 := SignedSaturate(a1 + b1)
 * ...
 * r7 := SignedSaturate(a7 + b7)
 * */
FORCE_INLINE __m128i _mm_adds_epi16(__m128i a, __m128i b)
{
	return (__m128i)vqaddq_s16((int16x8_t)a, (int16x8_t)b); 
}

/* Subtracts the 8 signed 16-bit integers of b from the 8 signed 16-bit integers of a and saturates.
 * r0 := SignedSaturate(a0 - b0)
 * r1 := SignedSaturate(a1 - b1)
 * ...
 * r7 := SignedSaturate(a7 - b7)
 * */
FORCE_INLINE __m128i _mm_subs_epi16(__m128i a, __m128i b)
{
	return (__m128i)vqsubq_s16((int16x8_t) a, (int16x8_t) b);
}
/* Adds the 4 signed or unsigned 32-bit integers in a to the 4 signed or unsigned 32-bit integers in b.
 * r0 := a0 + b0
 * r1 := a1 + b1
 * r2 := a2 + b2
 * r3 := a3 + b3
 * */
FORCE_INLINE __m128i _mm_add_epi32(__m128i a, __m128i b)
{
	return vaddq_s32(a, b);
}

/* Subtracts the 4 signed or unsigned 32-bit integers of b from the 4 signed or unsigned 32-bit integers of a.
 * r0 := a0 - b0
 * r1 := a1 - b1
 * r2 := a2 - b2
 * r3 := a3 - b3
 * */
FORCE_INLINE __m128i _mm_sub_epi32(__m128i a, __m128i b)
{
	return vsubq_s32(a, b);
}

/* Adds the four single-precision, floating-point values of a and b.
 * r0 := a0 + b0
 * r1 := a1 + b1
 * r2 := a2 + b2
 * r3 := a3 + b3
 * */
FORCE_INLINE __m128 _mm_add_ps(__m128 a, __m128 b)
{
	return vaddq_f32(a, b);
}

/* Subtracts the four single-precision, floating-point values of a and b.
 * r0 := a0 - b0
 * r1 := a1 - b1
 * r2 := a2 - b2
 * r3 := a3 - b3
 * */
FORCE_INLINE __m128 _mm_sub_ps(__m128 a, __m128 b)
{
	return vsubq_f32(a, b);
}

/* The haddps instruction performs a horizontal add, meaning that adjacent elements in the same operand are added together. Each 128-bit argument is considered as four 32-bit floating-point elements, numbered from 0 to 3, with 3 being the high-order element. The result of the operation on operand a (A3, A2, A1, A0) and operand b (B3, B2, B1, B0) is (B3 + B2, B1 + B0, A3 + A2, A1 + A0).
 * This routine is only available as an intrinsic*/
FORCE_INLINE __m128 _mm_hadd_ps(__m128 a, __m128 b)
{
	return vcombine_f32(vpadd_f32(vget_low_f32(a), vget_high_f32(a)), vpadd_f32(vget_low_f32(b), vget_high_f32(b)));
}

/***************************************************************************
 *                Multiply
 ***************************************************************************/

/* Multiplies the 8 signed 16-bit integers from a by the 8 signed 16-bit integers from b.
 * r0 := (a0 * b0)[31:16]
 * r1 := (a1 * b1)[31:16]
 * ...
 * r7 := (a7 * b7)[31:16]
 * */
FORCE_INLINE __m128i _mm_mulhi_epi16(__m128i a, __m128i b)
{
	return (__m128i)vshrq_n_s16(vqdmulhq_s16((int16x8_t)a, (int16x8_t)b), 1);
}

/* Multiplies the 8 signed or unsigned 16-bit integers from a by the 8 signed or unsigned 16-bit integers from b.
 * r0 := (a0 * b0)[15:0]
 * r1 := (a1 * b1)[15:0]
 * ...
 * r7 := (a7 * b7)[15:0]
 * */
FORCE_INLINE __m128i _mm_mullo_epi16(__m128i a, __m128i b)
{
	return (__m128i)vmulq_s16((int16x8_t)a, (int16x8_t)b);
}

/* Multiplies the four single-precision, floating-point values of a and b.
 * r0 := a0 * b0
 * r1 := a1 * b1
 * r2 := a2 * b2
 * r3 := a3 * b3
 * */
FORCE_INLINE __m128 _mm_mul_ps(__m128 a, __m128 b)
{
	//todo:
	//NEON:(-2.33512e-28) * (-2.13992e-13)=0
	//SSE: (-2.33512e-28) * (-2.13992e-13)=4.99689e-41
//	return vmulq_f32(a, b);
	__m128 ret;
	ret[0] = a[0] * b[0];
	ret[1] = a[1] * b[1];
	ret[2] = a[2] * b[2];
	ret[3] = a[3] * b[3];
	return ret;
}

/* Multiplies the 8 signed 16-bit integers from a by the 8 signed 16-bit integers from b.
 * __m128i _mm_madd_epi16 (__m128i a, __m128i b);
 * PMADDWD
 * Return Value
 * Adds the signed 32-bit integer results pairwise and packs the 4 signed 32-bit integer results.
 * r0 := (a0 * b0) + (a1 * b1)
 * r1 := (a2 * b2) + (a3 * b3)
 * r2 := (a4 * b4) + (a5 * b5)
 * r3 := (a6 * b6) + (a7 * b7)
 * */
FORCE_INLINE __m128i _mm_madd_epi16 (__m128i a, __m128i b)
{
	int32x4_t r_l = vmull_s16(vget_low_s16((int16x8_t)a), vget_low_s16((int16x8_t)b));
	int32x4_t r_h = vmull_s16(vget_high_s16((int16x8_t)a), vget_high_s16((int16x8_t)b));
	return vcombine_s32(vpadd_s32(vget_low_s32(r_l), vget_high_s32(r_l)), vpadd_s32(vget_low_s32(r_h), vget_high_s32(r_h)));
}

/***************************************************************************
 *                absdiff
 ***************************************************************************/
//#define _mm_absdiff_epu16(a,b) _mm_adds_epu16(_mm_subs_epu16(a, b), _mm_subs_epu16(b, a))

/* Computes the absolute difference of the 16 unsigned 8-bit integers from a and the 16 unsigned 8-bit integers from b.
 * __m128i _mm_sad_epu8 (__m128i a, __m128i b);
 * PSADBW
 * Return Value
 * Sums the upper 8 differences and lower 8 differences and packs the resulting 2 unsigned 16-bit integers into the upper and lower 64-bit elements.
 * r0 := abs(a0 - b0) + abs(a1 - b1) +...+ abs(a7 - b7)
 * r1 := 0x0 ; r2 := 0x0 ; r3 := 0x0
 * r4 := abs(a8 - b8) + abs(a9 - b9) +...+ abs(a15 - b15)
 * r5 := 0x0 ; r6 := 0x0 ; r7 := 0x0
 * */
FORCE_INLINE __m128i _mm_sad_epu8 (__m128i a, __m128i b)
{
	uint16x8_t t = vpaddlq_u8(vabdq_u8((uint8x16_t)a, (uint8x16_t)b));
	uint16_t r0 = t[0] + t[1] + t[2] + t[3];
	uint16_t r4 = t[4] + t[5] + t[6] + t[7];
	uint16x8_t r = vsetq_lane_u16(r0,vdupq_n_u16(0), 0);
	
	return (__m128i)vsetq_lane_u16(r4, r, 4);
}
/***************************************************************************
 *                divides
 ***************************************************************************/
/* r0 := a0 / b0
 * r1 := a1 / b1
 * r2 := a2 / b2
 * r3 := a3 / b3
 * */
FORCE_INLINE __m128 _mm_div_ps(__m128 a, __m128 b)
{
	// get an initial estimate of 1/b.
	float32x4_t reciprocal = vrecpeq_f32(b);

	// use a couple Newton-Raphson steps to refine the estimate.  Depending on your
	// application's accuracy requirements, you may be able to get away with only
	// one refinement (instead of the two used here).  Be sure to test!
	reciprocal = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);
	//reciprocal = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);
	// and finally, compute a/b = a*(1/b)
	float32x4_t result = vmulq_f32(a, reciprocal);
	
	return result;
}


/* Computes the square roots of the four single-precision, floating-point values of a.
 * r0 := sqrt(a0)
 * r1 := sqrt(a1)
 * r2 := sqrt(a2)
 * r3 := sqrt(a3)
 * */
FORCE_INLINE __m128 _mm_sqrt_ps(__m128 in)
{
	__m128 recipsq = vrsqrteq_f32(in);
	__m128 sq = vrecpeq_f32(recipsq);
	// ??? use step versions of both sqrt and recip for better accuracy?
	//precision loss
	//	__m128 recipsq = vrsqrtsq_f32(in,vdupq_n_f32(1.0));
	//	__m128 sq = vrecpsq_f32(recipsq,vdupq_n_f32(1.0));

	return sq;
}

/***************************************************************************
 *                logic
 ***************************************************************************/
/* Computes the bitwise AND of the 128-bit value in b and the bitwise NOT of the 128-bit value in a. 
 * r := (~a) & b
 * */
FORCE_INLINE __m128i _mm_andnot_si128(__m128i a, __m128i b)
{
	return vbicq_s32(b, a); // *NOTE* argument swap
}

/* Computes the bitwise AND-NOT of the four single-precision, floating-point values of a and b.
 * r0 := ~a0 & b0
 * r1 := ~a1 & b1
 * r2 := ~a2 & b2
 * r3 := ~a3 & b3
 * */
FORCE_INLINE __m128 _mm_andnot_ps(__m128 a, __m128 b)
{
	return (__m128)vbicq_s32((__m128i)b, (__m128i)a); // *NOTE* argument swap
}



/* Computes the bitwise AND of the four single-precision, floating-point values of a and b.
 * r0 := a0 & b0
 * r1 := a1 & b1
 * r2 := a2 & b2
 * r3 := a3 & b3
 * */
FORCE_INLINE __m128 _mm_and_ps(__m128 a, __m128 b)
{
	return (__m128)vandq_s32((__m128i)a, (__m128i)b);
}

/* Compares the 16 signed 8-bit integers in a and the 16 signed 8-bit integers in b for greater than. 
 * r0 := (a0 > b0) ? 0xff : 0x0
 * r1 := (a1 > b1) ? 0xff : 0x0
 * ...
 * r15 := (a15 > b15) ? 0xff : 0x0
 */
FORCE_INLINE __m128i _mm_cmpgt_epi8(__m128i a, __m128i b)
{
	return (__m128i)vcgtq_s8((int8x16_t)a, ( int8x16_t) b);
}

FORCE_INLINE __m128i _mm_cmpeq_epi8 (__m128i a, __m128i b)
{
	return (__m128i)vceqq_s8((int8x16_t) a, ( int8x16_t) b);
}

FORCE_INLINE int _mm_movemask_epi8(__m128i a)
{
	uint8x16_t input = (uint8x16_t)a;
	const int8_t __attribute__((aligned(16))) xr[8] = {-7, -6, -5, -4, -3, -2, -1, 0};
	uint8x8_t mask_and = vdup_n_u8(0x80);
	int8x8_t mask_shift = vld1_s8(xr);

	uint8x8_t lo = vget_low_u8(input);
	uint8x8_t hi = vget_high_u8(input);

	lo = vand_u8(lo, mask_and);
	lo = vshl_u8(lo, mask_shift);

	hi = vand_u8(hi, mask_and);
	hi = vshl_u8(hi, mask_shift);

	lo = vpadd_u8(lo, lo);
	lo = vpadd_u8(lo, lo);
	lo = vpadd_u8(lo, lo);

	hi = vpadd_u8(hi, hi);
	hi = vpadd_u8(hi, hi);
	hi = vpadd_u8(hi, hi);

	return ((hi[0] << 8) | (lo[0] & 0xFF));
}

/* Compares the 8 signed 16-bit integers in a and the 8 signed 16-bit integers in b for greater than.
 * r0 := (a0 > b0) ? 0xffff : 0x0
 * r1 := (a1 > b1) ? 0xffff : 0x0
 * ...
 * r7 := (a7 > b7) ? 0xffff : 0x0
 * */
FORCE_INLINE __m128i _mm_cmpgt_epi16(__m128i a, __m128i b)
{
	return (__m128i)vcgtq_s16((int16x8_t)a, ( int16x8_t) b);
}

/* Compares for greater than.
 * r0 := (a0 > b0) ? 0xffffffff : 0x0
 * r1 := (a1 > b1) ? 0xffffffff : 0x0
 * r2 := (a2 > b2) ? 0xffffffff : 0x0
 * r3 := (a3 > b3) ? 0xffffffff : 0x0
 * */
FORCE_INLINE __m128 _mm_cmpgt_ps(__m128 a, __m128 b)
{
	return (__m128)vcgtq_f32(a, b);
}

/* Compares for less than or equal.
 * r0 := (a0 <= b0) ? 0xffffffff : 0x0
 * r1 := (a1 <= b1) ? 0xffffffff : 0x0
 * r2 := (a2 <= b2) ? 0xffffffff : 0x0
 * r3 := (a3 <= b3) ? 0xffffffff : 0x0
 * */
FORCE_INLINE __m128 _mm_cmple_ps(__m128 a, __m128 b)
{
	return (__m128)vcleq_f32(a, b);
}
/***************************************************************************
 *                load and store
 ***************************************************************************/

FORCE_INLINE __m128 _mm_load_ps(const float * p)
{
	return vld1q_f32(p);
}

FORCE_INLINE void _mm_store_ps(float *p, __m128 a)
{
	vst1q_f32(p, a);
}


FORCE_INLINE __m128i _mm_loadl_epi64(__m128i const*p)
{
	/* Load the lower 64 bits of the value pointed to by p into the lower 64 bits of the result, zeroing the upper 64 bits of the result. */
	return vcombine_s32(vld1_s32((int32_t const *)p), vcreate_s32(0));
}

FORCE_INLINE void _mm_storel_epi64(__m128i* a, __m128i b)
{
	/* Reads the lower 64 bits of b and stores them into the lower 64 bits of a. */
	//
	//*a = (__m128i)vsetq_lane_s64((int64_t)vget_low_s32(b), *(int64x2_t*)a, 0);
	//vst1_s64( (int64_t *) a, vget_low_s64((int64x2_t)b));
	vst1_s32( (int32_t *) a, vget_low_s32((int32x4_t)b));
}

/* Sets the lower two single-precision, floating-point values with 64 bits of data loaded from the address p; the upper two values are passed through from a.
 * __m128 _mm_loadl_pi( __m128 a , __m64 * p );
 * MOVLPS reg, mem
 * Return Value
 *  r0 := *p0
 *  r1 := *p1
 *  r2 := a2
 *  r3 := a3
 *  */
FORCE_INLINE __m128 _mm_loadl_pi( __m128 a, __m64 const * p)
{
	return vcombine_f32(vld1_f32((float32_t const *)p), vget_high_f32(a));
}

/* Stores the lower two single-precision, floating-point values of a to the address p.
 * *p0 := b0
 * *p1 := b1
 * */
FORCE_INLINE void _mm_storel_pi( __m64 * p, __m128 a)
{
	vst1_f32((float32_t *)p, vget_low_f32((float32x4_t)a));
}

FORCE_INLINE __m128 _mm_load_ss(const float * p)
{
	/* Loads an single-precision, floating-point value into the low word and clears the upper three words. */
	__m128 result = vdupq_n_f32(0);
	return vsetq_lane_f32(*p, result, 0);
}

FORCE_INLINE void _mm_store_ss(float *p, __m128 a)
{
	/* Stores the lower single-precision, floating-point value. */
	vst1q_lane_f32(p, a, 0);
}
/***************************************************************************
 *                SET 
 ***************************************************************************/
/* Moves 32-bit integer a to the least significant 32 bits of an __m128 object, zero extending the upper bits.
 * r0 := a
 * r1 := 0x0 ; r2 := 0x0 ; r3 := 0x0
 * */
FORCE_INLINE __m128i _mm_cvtsi32_si128(int a)
{
	__m128i result = vdupq_n_s32(0);
	
	return vsetq_lane_s32(a, result, 0);
}

/* Sets the 16 signed 8-bit integer values to b.
 * r0 := b
 * r1 := b
 * ...
 * r15 := b
 * */
FORCE_INLINE __m128i _mm_set1_epi8 (char b)
{
	return (__m128i)vdupq_n_s8((int8_t)b);
}

/* Sets the 8 signed 16-bit integer values to w.
 * r0 := w
 * r1 := w
 * ...
 * r7 := w
 * */
FORCE_INLINE __m128i _mm_set1_epi16 (short w)
{
	return (__m128i)vdupq_n_s16((int16_t)w);
}

/* Sets the 4 signed 32-bit integer values to i.
 * r0 := i
 * r1 := i
 * r2 := i
 * r3 := I
 * */
FORCE_INLINE __m128i _mm_set1_epi32(int i)
{
	return vdupq_n_s32(i);
}
/* Sets the four single-precision, floating-point values to w
 * r0 := r1 := r2 := r3 := w 
 * */
FORCE_INLINE __m128 _mm_set1_ps(float w)
{
	return vdupq_n_f32(w);
}

/* Sets the 8 signed 16-bit integer values in reverse order.
 * __m128i _mm_setr_epi16 (short w0, short w1,    short w2, short w3,   short w4, short w5,   short w6, short w7);
 * (composite)
 * Return Value
 *  r0 := w0
 *  r1 := w1
 *  ...
 *  r7 := w7
 *  */
FORCE_INLINE __m128i _mm_setr_epi16 (short w0, short w1, short w2, short w3, short w4, short w5, short w6, short w7)
{
	short __attribute__ ((aligned (16))) data[8] = {w0, w1, w2, w3, w4, w5, w6, w7};
	
	return (__m128i)vld1q_s16((int16_t*)data);
}

//todo ~~~~~~~~~~~~~~~~~~~~~~~~~
/* Snhuffles the 4 signed or unsigned 32-bit integers in a as specified by imm. */
FORCE_INLINE __m128i _mm_shuffle_epi32 (__m128i a, int imm)
{
	switch (imm)
	{
		case 0 : 
			return vdupq_n_s32(vgetq_lane_s32(a, 0)); 
			break;
		default: 
		{
			__m128i ret;
			ret[0] = a[imm & 0x3];
			ret[1] = a[(imm >> 2) & 0x3];
			ret[2] = a[(imm >> 4) & 0x03];
			ret[3] = a[(imm >> 6) & 0x03];
			return ret; 
		}
	}
}

//todo ~~~~~~~~~~~~~~~~~~~~~~~~~
/* Selects four specific single-precision, floating-point values from a and b, based on the mask i. */
FORCE_INLINE __m128 _mm_shuffle_ps(__m128 a, __m128 b, int i)
{
	switch (i)
	{
//		case 0 : 
//			return 
//			break;
		default: 
		{
			__m128 ret;                  
			ret[0] = a[i & 0x3];         
			ret[1] = a[(i >> 2) & 0x3];  
			ret[2] = b[(i >> 4) & 0x03]; 
			ret[3] = b[(i >> 6) & 0x03]; 
			return ret; 
		}
	}
}
/***************************************************************************
 *                GET 
 ***************************************************************************/
FORCE_INLINE int _mm_cvtsi128_si32(__m128i a)
{
	/* Moves the least significant 32 bits of a to a 32-bit integer. */
	return vgetq_lane_s32(a, 0);
}

/* Sets the 128-bit value to zero.
 * r := 0x0
 * */
FORCE_INLINE __m128i _mm_setzero_si128(void)
{
	return vdupq_n_s32(0);
}

/* Clears the four single-precision, floating-point values.
 * r0 := r1 := r2 := r3 := 0.0 
 * */
FORCE_INLINE __m128 _mm_setzero_ps(void)
{
	return vdupq_n_f32(0);
}
/***************************************************************************
 *                convert 
 ***************************************************************************/

/* Packs the 8 signed 32-bit integers from a and b into signed 16-bit integers and saturates.
 * r0 := SignedSaturate(a0)
 * r1 := SignedSaturate(a1)
 * r2 := SignedSaturate(a2)
 * r3 := SignedSaturate(a3)
 * r4 := SignedSaturate(b0)
 * r5 := SignedSaturate(b1)
 * r6 := SignedSaturate(b2)
 * r7 := SignedSaturate(b3)
 * */
FORCE_INLINE __m128i _mm_packs_epi32(__m128i a, __m128i b)
{
	return (__m128i)vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
}

/* Interleaves the upper 8 signed or unsigned 8-bit integers in a with the upper 8 signed or unsigned 8-bit integers in b.
 * r0 := a8 ; r1 := b8
 * r2 := a9 ; r3 := b9
 * ...
 * r14 := a15 ; r15 := b15
 * */
FORCE_INLINE __m128i _mm_unpackhi_epi8(__m128i a, __m128i b)
{
	int8x8_t a_h = vget_high_s8((int8x16_t)a);
	int8x8_t b_h = vget_high_s8((int8x16_t)b);
	int8x8x2_t r = vzip_s8(a_h, b_h);
	
	return (__m128i)vcombine_s8(r.val[0], r.val[1]);
}

/* Interleaves the lower 8 signed or unsigned 8-bit integers in a with the lower 8 signed or unsigned 8-bit integers in b.
 * r0 := a0 ; r1 := b0
 * r2 := a1 ; r3 := b1
 * ...
 * r14 := a7 ; r15 := b7
 * */
FORCE_INLINE __m128i _mm_unpacklo_epi8(__m128i a, __m128i b)
{
	int8x8_t a_l = vget_low_s8((int8x16_t)a);
	int8x8_t b_l = vget_low_s8((int8x16_t)b);
	int8x8x2_t r = vzip_s8(a_l, b_l);
	
	return (__m128i)vcombine_s8(r.val[0], r.val[1]);
}

/* Interleaves the lower 2 signed or unsigned 32-bit integers in a with the lower 2 signed or unsigned 32-bit integers in b.
 * r0 := a0 ; r1 := b0
 * r2 := a1 ; r3 := b1
 * */
FORCE_INLINE __m128i _mm_unpacklo_epi32(__m128i a, __m128i b)
{
	int32x2_t a_l = vget_low_s32((int32x4_t)a);
	int32x2_t b_l = vget_low_s32((int32x4_t)b);
	int32x2x2_t r = vzip_s32(a_l, b_l);
	
	return (__m128i)vcombine_s32(r.val[0], r.val[1]);
}

FORCE_INLINE __m128i _mm_unpacklo_epi64(__m128i a, __m128i b)
{
	int64x1_t a_l = vget_low_s64((int64x2_t)a);
	int64x1_t b_l = vget_low_s64((int64x2_t)b);
	
	return (__m128i)vcombine_s64(a_l, b_l);
}

/* Interleaves the upper signed or unsigned 64-bit integer in a with the upper signed or unsigned 64-bit integer in b.
 * __m128i _mm_unpackhi_epi64 (__m128i a, __m128i b);
 * PUNPCKHQDQ
 * Return Value
 *  r0 := a1 ; r1 := b1
 *  */
FORCE_INLINE __m128i _mm_unpackhi_epi64(__m128i a, __m128i b)
{
	int64x1_t a_h = vget_high_s64((int64x2_t)a);
	int64x1_t b_h = vget_high_s64((int64x2_t)b);
	
	return (__m128i)vcombine_s64(a_h, b_h);
}

/* Interleaves the upper 4 signed or unsigned 16-bit integers in a with the upper 4 signed or unsigned 16-bit integers in b.
 * r0 := a4 ; r1 := b4
 * r2 := a5 ; r3 := b5
 * r4 := a6 ; r5 := b6
 * r6 := a7 ; r7 := b7
 * */
FORCE_INLINE __m128i _mm_unpackhi_epi16(__m128i a, __m128i b)
{
	int16x4_t a_h = vget_high_s16((int16x8_t)a);
	int16x4_t b_h = vget_high_s16((int16x8_t)b);
	int16x4x2_t result = vzip_s16(a_h, b_h);
	
	return (__m128i)vcombine_s16(result.val[0], result.val[1]);
}

/* Interleaves the lower 4 signed or unsigned 16-bit integers in a with the lower 4 signed or unsigned 16-bit integers in b.
 * r0 := a0 ; r1 := b0
 * r2 := a1 ; r3 := b1
 * r4 := a2 ; r5 := b2
 * r6 := a3 ; r7 := b3
 * */
FORCE_INLINE __m128i _mm_unpacklo_epi16(__m128i a, __m128i b)
{
	int16x4_t a_l = vget_low_s16((int16x8_t)a);
	int16x4_t b_l = vget_low_s16((int16x8_t)b);
	int16x4x2_t result = vzip_s16(a_l, b_l);
	
	return (__m128i)vcombine_s16(result.val[0], result.val[1]);
}

FORCE_INLINE __m128i _mm_unpackhi_epi32(__m128i a, __m128i b)
{
	int32x2_t a1 = vget_high_s32(a);
	int32x2_t b1 = vget_high_s32(b);
	int32x2x2_t result = vzip_s32(a1, b1);

	return vcombine_s32(result.val[0], result.val[1]);
}

/* Selects and interleaves the lower two single-precision, floating-point values from a and b.
 * r0 := a0
 * r1 := b0
 * r2 := a1
 * r3 := b1
 * */
FORCE_INLINE __m128 _mm_unpacklo_ps(__m128 a, __m128 b)
{
	float32x2x2_t result = vzip_f32(vget_low_f32(a), vget_low_f32(b));
	
	return vcombine_f32(result.val[0], result.val[1]);
}

/* Selects and interleaves the upper two single-precision, floating-point values from a and b.
 * r0 := a2
 * r1 := b2
 * r2 := a3
 * r3 := b3
 * */
FORCE_INLINE __m128 _mm_unpackhi_ps(__m128 a, __m128 b)
{
	float32x2x2_t result = vzip_f32(vget_high_f32(a), vget_high_f32(b));
	
	return vcombine_f32(result.val[0], result.val[1]);
}

// Extracts the selected signed or unsigned 16-bit integer from a and zero extends.  https://msdn.microsoft.com/en-us/library/6dceta0c(v=vs.100).aspx
#define _mm_extract_epi16( a, imm ) vgetq_lane_s16((int16x8_t)a, imm)

FORCE_INLINE __m128 _mm_cvtepi32_ps(__m128i a)
{
	return vcvtq_f32_s32(a);
}
/* Converts the four single-precision, floating-point values of a to signed 32-bit integer values.
 * r0 := (int) a0
 * r1 := (int) a1
 * r2 := (int) a2
 * r3 := (int) a3
 * */
FORCE_INLINE __m128i _mm_cvtps_epi32(__m128 a)
{
	//todo:precision
	//NaN -0
	//
	return vcvtq_s32_f32(a);
//	__m128i ret;
//	ret[0] = (int64_t)a[0];
//	ret[1] = (int64_t)a[1];
//	ret[2] = (int64_t)a[2];
//	ret[3] = (int64_t)a[3];
//	return ret;
}

/* Packs the 16 signed 16-bit integers from a and b into 8-bit unsigned integers and saturates.
 *
 * r0 := UnsignedSaturate(a0)
 * r1 := UnsignedSaturate(a1)
 * ...
 * r7 := UnsignedSaturate(a7)
 * r8 := UnsignedSaturate(b0)
 * r9 := UnsignedSaturate(b1)
 * ...
 * r15 := UnsignedSaturate(b7)
 * */
FORCE_INLINE __m128i _mm_packus_epi16(const __m128i a, const __m128i b)
{
	return (__m128i)vcombine_u8(vqmovun_s16((int16x8_t)a), vqmovun_s16((int16x8_t)b));
}

/* Moves the lower two single-precision, floating-point values of b to the upper two single-precision, floating-point values of the result. 
 * r3 := b1
 * r2 := b0
 * r1 := a1
 * r0 := a0
 * */
FORCE_INLINE __m128 _mm_movelh_ps(__m128 a, __m128 b)
{
	return vcombine_f32(vget_low_f32(a), vget_low_f32(b));
}

/* The upper two single-precision, floating-point values of a are passed through to the result.
 * r3 := a3
 * r2 := a2
 * r1 := b3
 * r0 := b2
 * */
FORCE_INLINE __m128 _mm_movehl_ps(__m128 a, __m128 b)
{
	return vcombine_f32(vget_high_f32(b), vget_high_f32(a));
}

/***************************************************************************
 *                shift 
 ***************************************************************************/
/* Shifts the 4 signed 32-bit integers in a right by count bits while shifting in the sign bit.
* r0 := a0 >> count
* r1 := a1 >> count
* r2 := a2 >> count
* r3 := a3 >> count
* immediate ,use  #define _mm_srai_epi32(a, imm) vshrq_n_s32(a, imm)
* */
FORCE_INLINE __m128i _mm_srai_epi32(__m128i a, int count)
{
//	return vshrq_n_s32(a, count);
//	todo :
//	if immediate
	return vshlq_s32(a, vdupq_n_s32(-count));
}

/* Shifts the 8 signed 16-bit integers in a right by count bits while shifting in the sign bit.
 *  r0 := a0 >> count
 *  r1 := a1 >> count
 *  ...
 *  r7 := a7 >> count
 *  */
FORCE_INLINE __m128i _mm_srai_epi16 (__m128i a, int count)
{
//	return vshrq_n_s16(a, count);
//	todo :
//	if immediate
	return (__m128i)vshlq_s16((int16x8_t)a, vdupq_n_s16(-count));
}

/* Shifts the 8 signed or unsigned 16-bit integers in a left by count bits while shifting in zeros.
 * r0 := a0 << count
 * r1 := a1 << count
 * ...
 * r7 := a7 << count
 * */
FORCE_INLINE __m128i _mm_slli_epi16(__m128i a, int count)
{
//	todo :
//	if immediate
	return (__m128i)vshlq_s16((int16x8_t)a, vdupq_n_s16(count));
}

/* Shifts the 8 signed or unsigned 16-bit integers in a right by count bits while shifting in zeros.
 * r0 := srl(a0, count)
 * r1 := srl(a1, count)
 * ...
 * r7 := srl(a7, count)
 * */
FORCE_INLINE __m128i _mm_srli_epi16(__m128i a, int count)
{
//	todo :
//	if immediate
	return (__m128i)vshlq_u16((uint16x8_t)a, vdupq_n_s16(-count));
}

/* Shifts the 128-bit value in a right by imm bytes while shifting in zeros. imm must be an immediate.
 * r := srl(a, imm*8)
 * */
//#define _mm_srli_si128( a, imm ) (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), (imm))
FORCE_INLINE __m128i _mm_srli_si128 (__m128i a, const int imm8)
{
	//if(imm8 < 0)
		//return (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), imm8);
	if(imm8 <= 0)
        return a;
	if(imm8 > 15)
		return _mm_setzero_si128();
	
	switch(imm8)
    {
        //不需要break，匹配后直接return
        case 1:
            return (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), 1);
        case 2:
            return (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), 2);
        case 3:
            return (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), 3);
        case 4:
            return (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), 4);
        case 5:
            return (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), 5);
        case 6:
            return (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), 6);
        case 7:
            return (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), 7);
		case 8:
            return (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), 8);
        case 9:
            return (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), 9);
        case 10:
            return (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), 10);
        case 11:
            return (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), 11);
        case 12:
            return (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), 12);
        case 13:
            return (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), 13);
        case 14:
            return (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), 14);
        case 15:
            return (__m128i)vextq_s8((int8x16_t)a, vdupq_n_s8(0), 15);
    }
}

/* Shifts the 128-bit value in a left by imm bytes while shifting in zeros. imm must be an immediate.
 * r := a << (imm * 8)*/
//todo ::imm =0, compile error
//#define _mm_slli_si128( a, imm ) (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 16 - (imm))
FORCE_INLINE __m128i _mm_slli_si128 (__m128i a, int imm8)
{
	if(imm8<=0)
	//return (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 16-(imm8));
		return a;
	//if(imm8==0)
	//	return a;
	if(imm8>15)
		return _mm_setzero_si128();
	switch(imm8)
	{
		case 1:
			return (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 15);
		case 2:
			return (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 14);
		case 3:
			return (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 13);
		case 4:
			return (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 12);
		case 5:
			return (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 11);
		case 6:
			return (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 10);
		case 7:
			return (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 9);
		case 8:
			return (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 8);
		case 9:
			return (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 7);
		case 10:
			return (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 6);
		case 11:
			return (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 5);
		case 12:
			return (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 4);
		case 13:
			return (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 3);
		case 14:
			return (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 2);
		case 15:
			return (__m128i)vextq_s8(vdupq_n_s8(0), (int8x16_t)a, 1);
	}
}

//--------------------------------------------------------------------------------------------------second phase----------------------------------------------------------------------------------------------
// Starting with the initial value in crc, accumulates a CRC32 value for unsigned 32-bit integer v, and stores the result in dst.
FORCE_INLINE unsigned int _mm_crc32_u32(unsigned int crc, unsigned int v)
{
	//#define CRC32CW(crc, value) __asm__("crc32cw %w[c], %w[c], %w[v]":[c]"+r"(crc):[v]"r"(value))
	asm volatile("crc32cw %w[c], %w[c], %w[val]" : [c]"+r"(crc) : [val]"r"(v));

	return crc;
}

#if 0
// Perform a carry-less multiplication of two 64-bit integers, selected from a and b according to imm8, and store the results in dst.
// test other imm8 value
FORCE_INLINE __m128i _mm_clmulepi64_si128(__m128i a, __m128i b, const int imm8)
{	
	switch(imm8 & 0x11)
	{
		case 0x00:
			return (__m128i)vmull_p64(vgetq_lane_u64((uint64x2_t)a, 0), vgetq_lane_u64((uint64x2_t)b, 0));
		case 0x01:
			return (__m128i)vmull_p64(vgetq_lane_u64((uint64x2_t)a, 1), vgetq_lane_u64((uint64x2_t)b, 0));
		case 0x10:
			return (__m128i)vmull_p64(vgetq_lane_u64((uint64x2_t)a, 0), vgetq_lane_u64((uint64x2_t)b, 1));
		case 0x11:
			return (__m128i)vmull_p64(vgetq_lane_u64((uint64x2_t)a, 1), vgetq_lane_u64((uint64x2_t)b, 1));
	}
}
#endif

// Cast vector of type __m128i to type __m128. This intrinsic is only used for compilation and does not generate any instructions,
// thus it has zero latency.
FORCE_INLINE __m128 _mm_castsi128_ps(__m128i a)
{
	return (__m128)a;
	/*//float32x4_t vcvtq_f32_s32(int32x4_t a);
	
	return (__m128)vcvtq_f32_s32((int32x4_t)a);*/
}

// Compute the bitwise XOR of packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
FORCE_INLINE __m128 _mm_xor_ps(__m128 a, __m128 b)
{
	//int32x4_t  veorq_s32(int32x4_t a, int32x4_t b);
	//return (__m128)vcvtq_f32_s32(veorq_s32(vcvtq_s32_f32((float32x4_t )a), vcvtq_s32_f32((float32x4_t )b)));
	
	return (__m128)veorq_s32((int32x4_t)a, (int32x4_t)b);
}

// Cast vector of type __m128 to type __m128i. 
// This intrinsic is only used for compilation and does not generate any instructions, thus it has zero latency.
FORCE_INLINE __m128i _mm_castps_si128(__m128 a)
{
	return (__m128i)a;
	//return  (__m128i)vcvtq_s32_f32((float32x4_t)a);
}

// Shuffle packed 8-bit integers in a according to shuffle control mask in the corresponding 8-bit element of b,
// and store the results in dst.
FORCE_INLINE __m128i _mm_shuffle_epi8(__m128i a, __m128i b)
{
	int8x16_t res;
	int8x16_t a_tmp = (int8x16_t)a;
	int8x16_t b_tmp = (int8x16_t)b;
	
	res[0] = (b_tmp[0] & 0x80) ? 0 : a_tmp[b_tmp[0] & 0x0f];
	res[1] = (b_tmp[1] & 0x80) ? 0 : a_tmp[b_tmp[1] & 0x0f];
	res[2] = (b_tmp[2] & 0x80) ? 0 : a_tmp[b_tmp[2] & 0x0f];
	res[3] = (b_tmp[3] & 0x80) ? 0 : a_tmp[b_tmp[3] & 0x0f];
	res[4] = (b_tmp[4] & 0x80) ? 0 : a_tmp[b_tmp[4] & 0x0f];
	res[5] = (b_tmp[5] & 0x80) ? 0 : a_tmp[b_tmp[5] & 0x0f];
	res[6] = (b_tmp[6] & 0x80) ? 0 : a_tmp[b_tmp[6] & 0x0f];
	res[7] = (b_tmp[7] & 0x80) ? 0 : a_tmp[b_tmp[7] & 0x0f];
	res[8] = (b_tmp[8] & 0x80) ? 0 : a_tmp[b_tmp[8] & 0x0f];
	res[9] = (b_tmp[9] & 0x80) ? 0 : a_tmp[b_tmp[9] & 0x0f];
	res[10] = (b_tmp[10] & 0x80) ? 0 : a_tmp[b_tmp[10] & 0x0f];
	res[11] = (b_tmp[11] & 0x80) ? 0 : a_tmp[b_tmp[11] & 0x0f];
	res[12] = (b_tmp[12] & 0x80) ? 0 : a_tmp[b_tmp[12] & 0x0f];
	res[13] = (b_tmp[13] & 0x80) ? 0 : a_tmp[b_tmp[13] & 0x0f];
	res[14] = (b_tmp[14] & 0x80) ? 0 : a_tmp[b_tmp[14] & 0x0f];
	res[15] = (b_tmp[15] & 0x80) ? 0 : a_tmp[b_tmp[15] & 0x0f];

	return (__m128i)res;
}

// Extract a 32-bit integer from a, selected with imm8, and store the result in dst.
//test imm8 range
FORCE_INLINE int _mm_extract_epi32(__m128i a, const int imm8)
{
	//int32_t vgetq_lane_s32 (int32x4_t v, const int lane)
	switch(imm8)
	{
		case 0:
			return vgetq_lane_s32((int32x4_t)a, 0);
		case 1:
			return vgetq_lane_s32((int32x4_t)a, 1);
		case 2:
			return vgetq_lane_s32((int32x4_t)a, 2);
		case 3:
			return vgetq_lane_s32((int32x4_t)a, 3);
		default:
			return vgetq_lane_s32((int32x4_t)a, 0);
	}
}

// Compare packed strings in a and b with lengths la and lb using the control in imm8, and store the generated index in dst. 
FORCE_INLINE int _mm_cmpestri(__m128i a, int la, __m128i b, int lb, const int imm8)
{
	int i = 0, j = 0, k = 0;
	int size = (imm8 & 0x01) ? 16 : 8;
    int upperbound = 128 / size;
	int sel = imm8 & 0x0c;
	int ai_invalid;
	int negative_mask = imm8 & 0x30;
	
	int bool_res_mask[16];
	int res_mask, res2_mask, ans;
	
	if(la < 0)
		la = -la;
	if(lb < 0)
		lb = -lb;
	if(la > upperbound)
		la = upperbound;
	if(lb > upperbound)
		lb = upperbound;
			
	if(upperbound == 8){
		int16x8_t a_tmp = (int16x8_t)a;
		int16x8_t b_tmp = (int16x8_t)b;
		int16x8_t bool_res[8];
		int16x8_t ai_dup;
		int a_invalid = 0xff << la;
		int b_invalid = 0xff << lb;
	
		for(i = 0; i < upperbound; i++){
			ai_dup = (int16x8_t)_mm_set1_epi16(a_tmp[i]);
			bool_res[i] = (int16x8_t)_mm_cmpeq_epi16((__m128i)ai_dup, (__m128i)b_tmp);
			bool_res_mask[i] = _mm_movemask_epi16((__m128i)bool_res[i]);
			
			ai_invalid = (a_invalid >> i & 0x01) ? 0xff : 0;
			switch(sel){
			case 0x00:
			case 0x04:
				bool_res_mask[i] &= ~(ai_invalid | b_invalid);
				break;
			case 0x08:
				bool_res_mask[i] &= ~(ai_invalid ^ b_invalid);
				bool_res_mask[i] |= (ai_invalid & b_invalid);
				break;
			case 0x0c:
				bool_res_mask[i] &= ~(~ai_invalid & b_invalid);
				bool_res_mask[i] |= ai_invalid;
				break;
			}
		}
		
		switch(sel){
			case 0x00:
				res_mask = 0;
				for(i = 0; i < upperbound; i++)
					res_mask |= (((bool_res_mask[i])? 1 : 0) << i);
				break;
			case 0x04:
				res_mask = 0;
				for(i = 0; i < upperbound; i++){
					for(j = 0; j < upperbound; j += 2){
						if((bool_res_mask[i] >> j & 0x11) == 0x11){
							res_mask |= (1 << i);
							break;
						}
					}
				}
				break;
			case 0x08:
				res_mask = 0;
				for(i = 0; i < upperbound; i++)
					res_mask |= (bool_res_mask[i] >> i & 0x01) << i;
				break;
			case 0x0c:
				res_mask = 0xff;
				for(i = 0; i < upperbound; i++){
					k = i;
					for(j = 0; j < upperbound - i; j++){
						if(!((bool_res_mask[j] >> k) & 0x01)){
							res_mask &= (~(1 << i));
							break;
						}	
						k++;
					}
				}
				break;
        }

		res2_mask = res_mask;
		
		if(negative_mask == 0x10)
			//res2_mask = 0xff;
			res2_mask = ~res2_mask;
		if(negative_mask == 0x30)
			//res2_mask |= (0xff >> (upperbound - lb));
			res2_mask ^= (0xff >> (upperbound - lb));
		if(imm8 & 0x40){
			asm volatile("clz %w[wd], %w[wn]" : [wd]"=r"(ans) : [wn]"r"(res2_mask));
			ans = 31 - ans;
        }
        else{
			int r_res2_mask = 0;
			asm volatile("rbit %w[wd], %w[wn]" : [wd]"=r"(r_res2_mask) : [wn]"r"(res2_mask));
			asm volatile("clz %w[wd], %w[wn]" : [wd]"=r"(ans) : [wn]"r"(r_res2_mask));
			
			ans = (ans > 7) ? 8 : ans;
        }
	}
	else{
		int8x16_t a_tmp = (int8x16_t)a;
		int8x16_t b_tmp = (int8x16_t)b;
		int8x16_t bool_res[16];
		int8x16_t ai_dup;
		int a_invalid = 0xffff << la;
		int b_invalid = 0xffff << lb;
		
		for(i = 0; i < upperbound; i++){
			ai_dup = (int8x16_t)_mm_set1_epi8(a_tmp[i]);
			bool_res[i] = (int8x16_t)_mm_cmpeq_epi8((__m128i)ai_dup, (__m128i)b_tmp);
			bool_res_mask[i] = _mm_movemask_epi8((__m128i)bool_res[i]);
			
			ai_invalid = (a_invalid >> i & 0x01) ? 0xffff : 0;
			switch(sel){
			case 0x00:
			case 0x04:
					
				bool_res_mask[i] &= ~(ai_invalid | b_invalid);
				break;
			case 0x08:
				bool_res_mask[i] &= ~(ai_invalid ^ b_invalid);
				bool_res_mask[i] |= (ai_invalid & b_invalid);
				break;
			case 0x0c:
				bool_res_mask[i] &= ~(~ai_invalid & b_invalid);
				bool_res_mask[i] |= ai_invalid;
				break;
			}
		}
