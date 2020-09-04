#ifndef FASTFLOAT_FLOAT_COMMON_H
#define FASTFLOAT_FLOAT_COMMON_H

#include <cstdint>
#ifndef _WIN32
// strcasecmp, strncasecmp 
#include <strings.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define FASTFLOAT_VISUAL_STUDIO 1
#endif

#ifdef FASTFLOAT_VISUAL_STUDIO
#define fastfloat_really_inline __forceinline
#else
#define fastfloat_really_inline inline __attribute__((always_inline))
#endif 

#ifdef _MSC_VER
#define fastfloat_strcasecmp _stricmp
#define fastfloat_strncasecmp _strnicmp
#else
#define fastfloat_strcasecmp strcasecmp
#define fastfloat_strncasecmp strncasecmp
#endif
namespace fastfloat {


bool is_space(uint8_t c) {
    static const bool table[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    return table[c];
}

namespace {
constexpr uint32_t max_digits = 800;

constexpr int32_t decimal_point_range = 2047;
} // namespace


struct value128 {
  uint64_t low;
  uint64_t high;
  value128(uint64_t _low, uint64_t _high) : low(_low), high(_high) {}
  value128() : low(0), high(0) {}
};


/* result might be undefined when input_num is zero */
fastfloat_really_inline 
int leading_zeroes(uint64_t input_num) {
#ifdef FASTFLOAT_VISUAL_STUDIO
  unsigned long leading_zero = 0;
  // Search the mask data from most significant bit (MSB)
  // to least significant bit (LSB) for a set bit (1).
  if (_BitScanReverse64(&leading_zero, input_num))
    return (int)(63 - leading_zero);
  else
    return 64;
#else
  return __builtin_clzll(input_num);
#endif
}


#ifdef FASTFLOAT_VISUAL_STUDIO


#if !defined(_M_X64) && !defined(_M_ARM64)// _umul128 for x86, arm
// this is a slow emulation routine for 32-bit Windows
//
fastfloat_really_inline uint64_t __emulu(uint32_t x, uint32_t y) {
  return x * (uint64_t)y;
}
fastfloat_really_inline uint64_t _umul128(uint64_t ab, uint64_t cd, uint64_t *hi) {
  uint64_t ad = __emulu((uint32_t)(ab >> 32), (uint32_t)cd);
  uint64_t bd = __emulu((uint32_t)ab, (uint32_t)cd);
  uint64_t adbc = ad + __emulu((uint32_t)ab, (uint32_t)(cd >> 32));
  uint64_t adbc_carry = !!(adbc < ad);
  uint64_t lo = bd + (adbc << 32);
  *hi = __emulu((uint32_t)(ab >> 32), (uint32_t)(cd >> 32)) + (adbc >> 32) +
        (adbc_carry << 32) + !!(lo < bd);
  return lo;
}
#endif

fastfloat_really_inline value128 full_multiplication(uint64_t value1, uint64_t value2) {
  value128 answer;
#ifdef _M_ARM64
  // ARM64 has native support for 64-bit multiplications, no need to emultate
  answer.high = __umulh(value1, value2);
  answer.low = value1 * value2;
#else
  answer.low = _umul128(value1, value2, &answer.high); // _umul128 not available on ARM64
#endif // _M_ARM64
  return answer;
}

#else

// compute value1 * value2
fastfloat_really_inline
value128 full_multiplication(uint64_t value1, uint64_t value2) {
  value128 answer;
  __uint128_t r = ((__uint128_t)value1) * value2;
  answer.low = uint64_t(r);
  answer.high = uint64_t(r >> 64);
  return answer;
}

#endif

struct adjusted_mantissa {
  uint64_t mantissa;
  int power2;
  adjusted_mantissa() : mantissa(0), power2(0) {}
};

struct decimal {
  uint32_t num_digits;
  int32_t decimal_point;
  bool negative;
  bool truncated;
  uint8_t digits[max_digits];
};

template <typename T>
struct binary_format {
  static constexpr int mantissa_explicit_bits();
  static constexpr int minimum_exponent();
  static constexpr int infinite_power();
  static constexpr int max_power_for_even();
  static constexpr int sign_index();
};
template <>
constexpr int binary_format<double>::mantissa_explicit_bits() {
  return 52;
}
template <>
constexpr int binary_format<float>::mantissa_explicit_bits() { 
  return 23;
}

template <>
constexpr int binary_format<double>::minimum_exponent() { 
  return -1023;
}
template <>
constexpr int binary_format<float>::minimum_exponent() {
  return -127;
}

template <>
constexpr int binary_format<double>::infinite_power() {
  return 0x7FF; 
}
template <>
constexpr int binary_format<float>::infinite_power() { 
  return 0xFF;
}

template <>
constexpr int binary_format<double>::max_power_for_even() {
  return 23;
}
template <>
constexpr int binary_format<float>::max_power_for_even() { 
  return 10;
}

template <>
constexpr int binary_format<double>::sign_index() { 
  return 63;
}
template <>
constexpr int binary_format<float>::sign_index() {
  return 31;
}


} // namespace fastfloat

#endif