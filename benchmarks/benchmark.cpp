#ifdef __CYGWIN__
#define _GNU_SOURCE // for strtod_l
#endif

#ifndef __CYGWIN__
#include "absl/strings/charconv.h"
#include "absl/strings/numbers.h"
#endif
#include "fast_float/fast_float.h"

#ifdef ENABLE_RYU
#include "ryu_parse.h"
#endif


#include "double-conversion/ieee.h"
#include "double-conversion/double-conversion.h"

#define IEEE_8087
#include "cxxopts.hpp"
#if defined(__linux__) || (__APPLE__ &&  __aarch64__)
#define USING_COUNTERS
#include "event_counter.h"
#endif
#include "dtoa.c"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <locale.h>

#include "random_generators.h"

/**
 * Determining whether we should import xlocale.h or not is
 * a bit of a nightmare.
 */
#ifdef __GLIBC__
#include <features.h>
#if !((__GLIBC__ > 2) || ((__GLIBC__ == 2) && (__GLIBC_MINOR__ > 25)))
#include <xlocale.h> // old glibc
#endif
#else            // not glibc
#if !defined(_MSC_VER) && !defined(__CYGWIN__) // assume that everything that is not GLIBC, Cygwin or Visual
                                               // Studio needs xlocale.h
#include <xlocale.h>
#endif
#endif

template <typename CharT>
double findmax_doubleconversion(std::vector<std::basic_string<CharT>> &s) {
  double answer = 0;
  double x;
  // from_chars does not allow leading spaces:
  // double_conversion::StringToDoubleConverter::ALLOW_LEADING_SPACES |
  int flags = double_conversion::StringToDoubleConverter::ALLOW_TRAILING_JUNK |
              double_conversion::StringToDoubleConverter::ALLOW_TRAILING_SPACES;
  double empty_string_value = 0.0;
  uc16 separator = double_conversion::StringToDoubleConverter::kNoSeparator;
  double_conversion::StringToDoubleConverter converter(
      flags, empty_string_value, double_conversion::Double::NaN(), NULL, NULL,
      separator);
  int processed_characters_count;
  for (auto &st : s) {
    if constexpr (std::is_same<CharT, char16_t>::value) {
      x = converter.StringToDouble((const uc16*)st.data(), st.size(), &processed_characters_count);
    } else { 
      x = converter.StringToDouble(st.data(), st.size(), &processed_characters_count);
    }
    if (processed_characters_count == 0) {
      throw std::runtime_error("bug in findmax_doubleconversion");
    }
    answer = answer > x ? answer : x;
  }
  return answer;
}
#ifdef _WIN32
double findmax_strtod_16(std::vector<std::u16string>& s) {
  double answer = 0;
  double x = 0;
  for (auto& st : s) {
    auto* pr = (wchar_t*)st.data();
    static _locale_t c_locale = _create_locale(LC_ALL, "C");
    x = _wcstod_l((const wchar_t *)st.data(), &pr, c_locale);

    if (pr == (const wchar_t*)st.data()) {
      throw std::runtime_error("bug in findmax_strtod");
    }
    answer = answer > x ? answer : x;
  }
  return answer;
}
#endif
double findmax_netlib(std::vector<std::string> &s) {
  double answer = 0;
  double x = 0;
  for (std::string &st : s) {
    char *pr = (char *)st.data();
    x = netlib_strtod(st.data(), &pr);
    if (pr == st.data()) {
      throw std::runtime_error(std::string("bug in findmax_netlib ")+st);
    }
    answer = answer > x ? answer : x;
  }
  return answer;
}

#ifdef ENABLE_RYU
double findmax_ryus2d(std::vector<std::string> &s) {
  double answer = 0;
  double x = 0;
  for (std::string &st : s) {
    // Ryu does not track character consumption (boo), but we can at least...
    Status stat = s2d(st.data(), &x);
    if (stat != SUCCESS) {
      throw std::runtime_error(std::string("bug in findmax_ryus2d ")+st + " " + std::to_string(stat));
    }
    answer = answer > x ? answer : x;
  }
  return answer;
}
#endif

double findmax_strtod(std::vector<std::string> &s) {
  double answer = 0;
  double x = 0;
  for (std::string &st : s) {
    char *pr = (char *)st.data();
#ifdef _WIN32
    static _locale_t c_locale = _create_locale(LC_ALL, "C");
    x = _strtod_l(st.data(), &pr, c_locale);
#else
    static locale_t c_locale = newlocale(LC_ALL_MASK, "C", NULL);
    x = strtod_l(st.data(), &pr, c_locale);
#endif
    if (pr == st.data()) {
      throw std::runtime_error("bug in findmax_strtod");
    }
    answer = answer > x ? answer : x;
  }
  return answer;
}
// Why not `|| __cplusplus > 201703L`? Because GNU libstdc++ does not have
// float parsing for std::from_chars.
#if defined(_MSC_VER)
#define FROM_CHARS_AVAILABLE_MAYBE
#endif

#ifdef FROM_CHARS_AVAILABLE_MAYBE
double findmax_from_chars(std::vector<std::string> &s) {
  double answer = 0;
  double x = 0;
  for (std::string &st : s) {
    auto [p, ec] = std::from_chars(st.data(), st.data() + st.size(), x);
    if (p == st.data()) {
      throw std::runtime_error("bug in findmax_from_chars");
    }
    answer = answer > x ? answer : x;
  }
  return answer;
}
#endif

template <typename CharT>
double findmax_fastfloat(std::vector<std::basic_string<CharT>> &s) {
  double answer = 0;
  double x = 0;
  for (auto &st : s) {
    auto [p, ec] = fast_float::from_chars(st.data(), st.data() + st.size(), x);
    if (p == st.data()) {
      throw std::runtime_error("bug in findmax_fastfloat");
    }
    answer = answer > x ? answer : x;
  }
  return answer;
}

#ifndef __CYGWIN__
double findmax_absl_from_chars(std::vector<std::string> &s) {
  double answer = 0;
  double x = 0;
  for (std::string &st : s) {
    auto [p, ec] = absl::from_chars(st.data(), st.data() + st.size(), x);
    if (p == st.data()) {
      throw std::runtime_error("bug in findmax_absl_from_chars");
    }
    answer = answer > x ? answer : x;
  }
  return answer;
}
#endif

enum ConversionFlag {
  NO_CONVERSION_FLAG,
  ALLOW_NON_DECIMAL_PREFIX,
  ALLOW_TRAILING_JUNK
};

#define V8_INFINITY std::numeric_limits<double>::infinity()
#define DCHECK_EQ(...)
#define DCHECK_NE(...)
#define DCHECK_LT(...)
#define DCHECK_LE(...)
#define DCHECK_GE(...)
#define DCHECK(...)
#define SLOW_DCHECK(...)

inline double JunkStringValue() {
  return std::numeric_limits<double>::quiet_NaN();
}

enum OneByteCharFlags {
  kIsIdentifierStart = 1 << 0,
  kIsIdentifierPart = 1 << 1,
  kIsWhiteSpace = 1 << 2,
  kIsWhiteSpaceOrLineTerminator = 1 << 3,
  kMaybeLineEnd = 1 << 4
};

// See http://www.unicode.org/Public/UCD/latest/ucd/DerivedCoreProperties.txt
// ID_Start. Additionally includes '_' and '$'.
constexpr bool IsOneByteIDStart(uint32_t c) {
  return c == 0x0024 || (c >= 0x0041 && c <= 0x005A) || c == 0x005F ||
         (c >= 0x0061 && c <= 0x007A) || c == 0x00AA || c == 0x00B5 ||
         c == 0x00BA || (c >= 0x00C0 && c <= 0x00D6) ||
         (c >= 0x00D8 && c <= 0x00F6) || (c >= 0x00F8 && c <= 0x00FF);
}

// See http://www.unicode.org/Public/UCD/latest/ucd/DerivedCoreProperties.txt
// ID_Continue. Additionally includes '_' and '$'.
constexpr bool IsOneByteIDContinue(uint32_t c) {
  return c == 0x0024 || (c >= 0x0030 && c <= 0x0039) || c == 0x005F ||
         (c >= 0x0041 && c <= 0x005A) || (c >= 0x0061 && c <= 0x007A) ||
         c == 0x00AA || c == 0x00B5 || c == 0x00B7 || c == 0x00BA ||
         (c >= 0x00C0 && c <= 0x00D6) || (c >= 0x00D8 && c <= 0x00F6) ||
         (c >= 0x00F8 && c <= 0x00FF);
}

constexpr bool IsOneByteWhitespace(uint32_t c) {
  return c == '\t' || c == '\v' || c == '\f' || c == ' ' || c == u'\xa0';
}

constexpr uint8_t BuildOneByteCharFlags(uint32_t c) {
  uint8_t result = 0;
  if (IsOneByteIDStart(c) || c == '\\') result |= kIsIdentifierStart;
  if (IsOneByteIDContinue(c) || c == '\\') result |= kIsIdentifierPart;
  if (IsOneByteWhitespace(c)) {
    result |= kIsWhiteSpace | kIsWhiteSpaceOrLineTerminator;
  }
  if (c == '\r' || c == '\n') {
    result |= kIsWhiteSpaceOrLineTerminator | kMaybeLineEnd;
  }
  // Add markers to identify 0x2028 and 0x2029.
  if (c == static_cast<uint8_t>(0x2028) || c == static_cast<uint8_t>(0x2029)) {
    result |= kMaybeLineEnd;
  }
  return result;
}

#define INT_0_TO_127_LIST(V)                                          \
V(0)   V(1)   V(2)   V(3)   V(4)   V(5)   V(6)   V(7)   V(8)   V(9)   \
V(10)  V(11)  V(12)  V(13)  V(14)  V(15)  V(16)  V(17)  V(18)  V(19)  \
V(20)  V(21)  V(22)  V(23)  V(24)  V(25)  V(26)  V(27)  V(28)  V(29)  \
V(30)  V(31)  V(32)  V(33)  V(34)  V(35)  V(36)  V(37)  V(38)  V(39)  \
V(40)  V(41)  V(42)  V(43)  V(44)  V(45)  V(46)  V(47)  V(48)  V(49)  \
V(50)  V(51)  V(52)  V(53)  V(54)  V(55)  V(56)  V(57)  V(58)  V(59)  \
V(60)  V(61)  V(62)  V(63)  V(64)  V(65)  V(66)  V(67)  V(68)  V(69)  \
V(70)  V(71)  V(72)  V(73)  V(74)  V(75)  V(76)  V(77)  V(78)  V(79)  \
V(80)  V(81)  V(82)  V(83)  V(84)  V(85)  V(86)  V(87)  V(88)  V(89)  \
V(90)  V(91)  V(92)  V(93)  V(94)  V(95)  V(96)  V(97)  V(98)  V(99)  \
V(100) V(101) V(102) V(103) V(104) V(105) V(106) V(107) V(108) V(109) \
V(110) V(111) V(112) V(113) V(114) V(115) V(116) V(117) V(118) V(119) \
V(120) V(121) V(122) V(123) V(124) V(125) V(126) V(127)
const constexpr uint8_t kOneByteCharFlags[256] = {
#define BUILD_CHAR_FLAGS(N) BuildOneByteCharFlags(N),
    INT_0_TO_127_LIST(BUILD_CHAR_FLAGS)
#undef BUILD_CHAR_FLAGS
#define BUILD_CHAR_FLAGS(N) BuildOneByteCharFlags(N + 128),
        INT_0_TO_127_LIST(BUILD_CHAR_FLAGS)
#undef BUILD_CHAR_FLAGS
};

template <typename T, typename U>
inline constexpr bool IsInRange(T value, U lower_limit, U higher_limit) {
  DCHECK_LE(lower_limit, higher_limit);
  static_assert(sizeof(U) <= sizeof(T));
  using unsigned_T = typename std::make_unsigned<T>::type;
  // Use static_cast to support enum classes.
  return static_cast<unsigned_T>(static_cast<unsigned_T>(value) -
                                 static_cast<unsigned_T>(lower_limit)) <=
         static_cast<unsigned_T>(static_cast<unsigned_T>(higher_limit) -
                                 static_cast<unsigned_T>(lower_limit));
}

bool IsWhiteSpaceOrLineTerminator(uint32_t c) {
  if (!IsInRange(c, 0, 255)) return false;
  return kOneByteCharFlags[c] & kIsWhiteSpaceOrLineTerminator;
}


template <typename T>
class Vector {
 public:
  using value_type = T;
  using iterator = T*;
  using const_iterator = const T*;

  constexpr Vector() : start_(nullptr), length_(0) {}

  constexpr Vector(T* data, size_t length) : start_(data), length_(length) {
    DCHECK(length == 0 || data != nullptr);
  }

  static Vector<T> New(size_t length) {
    return Vector<T>(new T[length], length);
  }

  // Returns a vector using the same backing storage as this one,
  // spanning from and including 'from', to but not including 'to'.
  Vector<T> SubVector(size_t from, size_t to) const {
    DCHECK_LE(from, to);
    DCHECK_LE(to, length_);
    return Vector<T>(begin() + from, to - from);
  }
  Vector<T> SubVectorFrom(size_t from) const {
    return SubVector(from, length_);
  }

  template <class U>
  void OverwriteWith(Vector<U> other) {
    DCHECK_EQ(size(), other.size());
    std::copy(other.begin(), other.end(), begin());
  }

  template <class U, size_t n>
  void OverwriteWith(const std::array<U, n>& other) {
    DCHECK_EQ(size(), other.size());
    std::copy(other.begin(), other.end(), begin());
  }

  // Returns the length of the vector. Only use this if you really need an
  // integer return value. Use {size()} otherwise.
  int length() const {
    DCHECK_GE(std::numeric_limits<int>::max(), length_);
    return static_cast<int>(length_);
  }

  // Returns the length of the vector as a size_t.
  constexpr size_t size() const { return length_; }

  // Returns whether or not the vector is empty.
  constexpr bool empty() const { return length_ == 0; }

  // Access individual vector elements - checks bounds in debug mode.
  T& operator[](size_t index) const {
    DCHECK_LT(index, length_);
    return start_[index];
  }

  const T& at(size_t index) const { return operator[](index); }

  T& first() { return start_[0]; }
  const T& first() const { return start_[0]; }

  T& last() {
    DCHECK_LT(0, length_);
    return start_[length_ - 1];
  }
  const T& last() const {
    DCHECK_LT(0, length_);
    return start_[length_ - 1];
  }

  // Returns a pointer to the start of the data in the vector.
  constexpr T* begin() const { return start_; }
  constexpr const T* cbegin() const { return start_; }

  // For consistency with other containers, do also provide a {data} accessor.
  constexpr T* data() const { return start_; }

  // Returns a pointer past the end of the data in the vector.
  constexpr T* end() const { return start_ + length_; }
  constexpr const T* cend() const { return start_ + length_; }

  constexpr std::reverse_iterator<T*> rbegin() const {
    return std::make_reverse_iterator(end());
  }
  constexpr std::reverse_iterator<T*> rend() const {
    return std::make_reverse_iterator(begin());
  }

  // Returns a clone of this vector with a new backing store.
  Vector<T> Clone() const {
    T* result = new T[length_];
    for (size_t i = 0; i < length_; i++) result[i] = start_[i];
    return Vector<T>(result, length_);
  }

  void Truncate(size_t length) {
    DCHECK(length <= length_);
    length_ = length;
  }

  // Releases the array underlying this vector. Once disposed the
  // vector is empty.
  void Dispose() {
    delete[] start_;
    start_ = nullptr;
    length_ = 0;
  }

  Vector<T> operator+(size_t offset) {
    DCHECK_LE(offset, length_);
    return Vector<T>(start_ + offset, length_ - offset);
  }

  Vector<T> operator+=(size_t offset) {
    DCHECK_LE(offset, length_);
    start_ += offset;
    length_ -= offset;
    return *this;
  }

  // Implicit conversion from Vector<T> to Vector<const T>.
  operator Vector<const T>() const { return {start_, length_}; }

  template <typename S>
  static Vector<T> cast(Vector<S> input) {
    // Casting is potentially dangerous, so be really restrictive here. This
    // might be lifted once we have use cases for that.
    static_assert(std::is_trivial_v<S> && std::is_standard_layout_v<S>);
    static_assert(std::is_trivial_v<T> && std::is_standard_layout_v<T>);
    DCHECK_EQ(0, (input.size() * sizeof(S)) % sizeof(T));
    DCHECK_EQ(0, reinterpret_cast<uintptr_t>(input.begin()) % alignof(T));
    return Vector<T>(reinterpret_cast<T*>(input.begin()),
                     input.size() * sizeof(S) / sizeof(T));
  }

  bool operator==(const Vector<T>& other) const {
    return std::equal(begin(), end(), other.begin(), other.end());
  }

  bool operator!=(const Vector<T>& other) const {
    return !operator==(other);
  }

  template<typename TT = T>
  std::enable_if_t<!std::is_const_v<TT>, bool> operator==(
      const Vector<const T>& other) const {
    return std::equal(begin(), end(), other.begin(), other.end());
  }

  template<typename TT = T>
  std::enable_if_t<!std::is_const_v<TT>, bool> operator!=(
      const Vector<const T>& other) const {
    return !operator==(other);
  }

 private:
  T* start_;
  size_t length_;
};

static Vector<const char> TrimLeadingZeros(Vector<const char> buffer) {
  for (int i = 0; i < buffer.length(); i++) {
    if (buffer[i] != '0') {
      return buffer.SubVector(i, buffer.length());
    }
  }
  return Vector<const char>(buffer.begin(), 0);
}

static Vector<const char> TrimTrailingZeros(Vector<const char> buffer) {
  for (int i = buffer.length() - 1; i >= 0; --i) {
    if (buffer[i] != '0') {
      return buffer.SubVector(0, i + 1);
    }
  }
  return Vector<const char>(buffer.begin(), 0);
}

// 2^53 = 9007199254740992.
// Any integer with at most 15 decimal digits will hence fit into a double
// (which has a 53bit significand) without loss of precision.
static const int kMaxExactDoubleIntegerDecimalDigits = 15;
// 2^64 = 18446744073709551616 > 10^19
static const int kMaxUint64DecimalDigits = 19;

// Max double: 1.7976931348623157 x 10^308
// Min non-zero double: 4.9406564584124654 x 10^-324
// Any x >= 10^309 is interpreted as +infinity.
// Any x <= 10^-324 is interpreted as 0.
// Note that 2.5e-324 (despite being smaller than the min double) will be read
// as non-zero (equal to the min non-zero double).
static const int kMaxDecimalPower = 309;
static const int kMinDecimalPower = -324;

// 2^64 = 18446744073709551616
static const uint64_t kMaxUint64 = 0xFFFF'FFFF'FFFF'FFFF;

// clang-format off
static const double exact_powers_of_ten[] = {
  1.0,  // 10^0
  10.0,
  100.0,
  1000.0,
  10000.0,
  100000.0,
  1000000.0,
  10000000.0,
  100000000.0,
  1000000000.0,
  10000000000.0,  // 10^10
  100000000000.0,
  1000000000000.0,
  10000000000000.0,
  100000000000000.0,
  1000000000000000.0,
  10000000000000000.0,
  100000000000000000.0,
  1000000000000000000.0,
  10000000000000000000.0,
  100000000000000000000.0,  // 10^20
  1000000000000000000000.0,
  // 10^22 = 0x21E19E0C9BAB2400000 = 0x878678326EAC9 * 2^22
  10000000000000000000000.0
};
// clang-format on
static const int kExactPowersOfTenSize = sizeof(exact_powers_of_ten)/sizeof(double);

// Maximum number of significant digits in the decimal representation.
// In fact the value is 772 (see conversions.cc), but to give us some margin
// we round up to 780.
static const int kMaxSignificantDecimalDigits = 780;

static void TrimToMaxSignificantDigits(Vector<const char> buffer, int exponent,
                                       char* significant_buffer,
                                       int* significant_exponent) {
  for (int i = 0; i < kMaxSignificantDecimalDigits - 1; ++i) {
    significant_buffer[i] = buffer[i];
  }
  // The input buffer has been trimmed. Therefore the last digit must be
  // different from '0'.
  DCHECK_NE(buffer[buffer.length() - 1], '0');
  // Set the last digit to be non-zero. This is sufficient to guarantee
  // correct rounding.
  significant_buffer[kMaxSignificantDecimalDigits - 1] = '1';
  *significant_exponent =
      exponent + (buffer.length() - kMaxSignificantDecimalDigits);
}

// Reads digits from the buffer and converts them to a uint64.
// Reads in as many digits as fit into a uint64.
// When the string starts with "1844674407370955161" no further digit is read.
// Since 2^64 = 18446744073709551616 it would still be possible read another
// digit if it was less or equal than 6, but this would complicate the code.
static uint64_t ReadUint64(Vector<const char> buffer,
                           int* number_of_read_digits) {
  uint64_t result = 0;
  int i = 0;
  while (i < buffer.length() && result <= (kMaxUint64 / 10 - 1)) {
    int digit = buffer[i++] - '0';
    DCHECK(0 <= digit && digit <= 9);
    result = 10 * result + digit;
  }
  *number_of_read_digits = i;
  return result;
}

static bool DoubleStrtod(Vector<const char> trimmed, int exponent,
                         double* result) {
#if (V8_TARGET_ARCH_IA32 || defined(USE_SIMULATOR)) && !defined(_MSC_VER)
  // On x86 the floating-point stack can be 64 or 80 bits wide. If it is
  // 80 bits wide (as is the case on Linux) then double-rounding occurs and the
  // result is not accurate.
  // We know that Windows32 with MSVC, unlike with MinGW32, uses 64 bits and is
  // therefore accurate.
  // Note that the ARM simulators are compiled for 32bits. They
  // therefore exhibit the same problem.
  USE(exact_powers_of_ten);
  USE(kMaxExactDoubleIntegerDecimalDigits);
  USE(kExactPowersOfTenSize);
  return false;
#else
  if (trimmed.length() <= kMaxExactDoubleIntegerDecimalDigits) {
    int read_digits;
    // The trimmed input fits into a double.
    // If the 10^exponent (resp. 10^-exponent) fits into a double too then we
    // can compute the result-double simply by multiplying (resp. dividing) the
    // two numbers.
    // This is possible because IEEE guarantees that floating-point operations
    // return the best possible approximation.
    if (exponent < 0 && -exponent < kExactPowersOfTenSize) {
      // 10^-exponent fits into a double.
      *result = static_cast<double>(ReadUint64(trimmed, &read_digits));
      DCHECK(read_digits == trimmed.length());
      *result /= exact_powers_of_ten[-exponent];
      return true;
    }
    if (0 <= exponent && exponent < kExactPowersOfTenSize) {
      // 10^exponent fits into a double.
      *result = static_cast<double>(ReadUint64(trimmed, &read_digits));
      DCHECK(read_digits == trimmed.length());
      *result *= exact_powers_of_ten[exponent];
      return true;
    }
    int remaining_digits =
        kMaxExactDoubleIntegerDecimalDigits - trimmed.length();
    if ((0 <= exponent) &&
        (exponent - remaining_digits < kExactPowersOfTenSize)) {
      // The trimmed string was short and we can multiply it with
      // 10^remaining_digits. As a result the remaining exponent now fits
      // into a double too.
      *result = static_cast<double>(ReadUint64(trimmed, &read_digits));
      DCHECK(read_digits == trimmed.length());
      *result *= exact_powers_of_ten[remaining_digits];
      *result *= exact_powers_of_ten[exponent - remaining_digits];
      return true;
    }
  }
  return false;
#endif
}

double Strtod(Vector<const char> buffer, int exponent) {
  Vector<const char> left_trimmed = TrimLeadingZeros(buffer);
  Vector<const char> trimmed = TrimTrailingZeros(left_trimmed);
  exponent += left_trimmed.length() - trimmed.length();
  if (trimmed.empty()) return 0.0;
  if (trimmed.length() > kMaxSignificantDecimalDigits) {
    char significant_buffer[kMaxSignificantDecimalDigits];
    int significant_exponent;
    TrimToMaxSignificantDigits(trimmed, exponent, significant_buffer,
                               &significant_exponent);
    return Strtod(
        Vector<const char>(significant_buffer, kMaxSignificantDecimalDigits),
        significant_exponent);
  }
  if (exponent + trimmed.length() - 1 >= kMaxDecimalPower)
    return std::numeric_limits<double>::infinity();
  if (exponent + trimmed.length() <= kMinDecimalPower) return 0.0;

  double guess;
  DoubleStrtod(trimmed, exponent, &guess);
  return guess;
}

// Returns true if a nonspace character has been found and false if the
// end was been reached before finding a nonspace character.
template <class Char>
inline bool AdvanceToNonspace(const Char** current, const Char* end) {
  while (*current != end) {
    if (!IsWhiteSpaceOrLineTerminator(**current)) return true;
    ++*current;
  }
  return false;
}

template <class Char>
bool SubStringEquals(const Char** current, const Char* end,
                     const char* substring) {
  DCHECK(**current == *substring);
  for (substring++; *substring != '\0'; substring++) {
    ++*current;
    if (*current == end || **current != *substring) return false;
  }
  ++*current;
  return true;
}

inline double SignedZero(bool negative) {
  return negative ? -0.0 : 0.0;
}

// Converts a string to a double value.
template <class Char>
double InternalStringToDoubleV8(const Char* current, const Char* end,
                              ConversionFlag flag, double empty_string_val) {  
  // From here we are parsing a StrDecimalLiteral, as per
  // https://tc39.es/ecma262/#sec-tonumber-applied-to-the-string-type

  const bool allow_trailing_junk = flag == ALLOW_TRAILING_JUNK;

  // Maximum number of significant digits in decimal representation.
  // The longest possible double in decimal representation is
  // (2^53 - 1) * 2 ^ -1074 that is (2 ^ 53 - 1) * 5 ^ 1074 / 10 ^ 1074
  // (768 digits). If we parse a number whose first digits are equal to a
  // mean of 2 adjacent doubles (that could have up to 769 digits) the result
  // must be rounded to the bigger one unless the tail consists of zeros, so
  // we don't need to preserve all the digits.
  const int kMaxSignificantDigits = 772;

  // The longest form of simplified number is: "-<significant digits>'.1eXXX\0".
  const int kBufferSize = kMaxSignificantDigits + 10;
  char buffer[kBufferSize];
  int buffer_pos = 0;

  // Exponent will be adjusted if insignificant digits of the integer part
  // or insignificant leading zeros of the fractional part are dropped.
  int exponent = 0;
  int significant_digits = 0;
  int insignificant_digits = 0;
  bool nonzero_digit_dropped = false;

  enum class Sign { kNone, kNegative, kPositive };

  Sign sign = Sign::kNone;

  if (*current == '-') {
    ++current;
    if (current == end) return JunkStringValue();
    sign = Sign::kNegative;
  }

  // Ignore leading zeros in the integer part.
  bool leading_zero = false;
  if (*current == '0') {
    do {
      current++;
      if (current == end) return SignedZero(sign == Sign::kNegative);
    } while (*current == '0');
    leading_zero = true;
  }

  // Copy significant digits of the integer part (if any) to the buffer.
  while (*current >= '0' && *current <= '9') {
    if (significant_digits < kMaxSignificantDigits) {
      DCHECK_LT(buffer_pos, kBufferSize);
      buffer[buffer_pos++] = static_cast<char>(*current);
      significant_digits++;
      // Will later check if it's an octal in the buffer.
    } else {
      insignificant_digits++;  // Move the digit into the exponential part.
      nonzero_digit_dropped = nonzero_digit_dropped || *current != '0';
    }
    ++current;
    if (current == end) goto parsing_done;
  }

  if (*current == '.') {
    ++current;
    if (current == end) {
      if (significant_digits == 0 && !leading_zero) {
        return JunkStringValue();
      } else {
        goto parsing_done;
      }
    }

    if (significant_digits == 0) {
      // octal = false;
      // Integer part consists of 0 or is absent. Significant digits start after
      // leading zeros (if any).
      while (*current == '0') {
        ++current;
        if (current == end) return SignedZero(sign == Sign::kNegative);
        exponent--;  // Move this 0 into the exponent.
      }
    }

    // There is a fractional part.  We don't emit a '.', but adjust the exponent
    // instead.
    while (*current >= '0' && *current <= '9') {
      if (significant_digits < kMaxSignificantDigits) {
        DCHECK_LT(buffer_pos, kBufferSize);
        buffer[buffer_pos++] = static_cast<char>(*current);
        significant_digits++;
        exponent--;
      } else {
        // Ignore insignificant digits in the fractional part.
        nonzero_digit_dropped = nonzero_digit_dropped || *current != '0';
      }
      ++current;
      if (current == end) goto parsing_done;
    }
  }

  if (!leading_zero && exponent == 0 && significant_digits == 0) {
    // If leading_zeros is true then the string contains zeros.
    // If exponent < 0 then string was [+-]\.0*...
    // If significant_digits != 0 the string is not equal to 0.
    // Otherwise there are no digits in the string.
    return JunkStringValue();
  }

  // Parse exponential part.
  if (*current == 'e' || *current == 'E') {
    ++current;
    if (current == end) {
      if (allow_trailing_junk) {
        goto parsing_done;
      } else {
        return JunkStringValue();
      }
    }
    char exponent_sign = '+';
    if (*current == '+' || *current == '-') {
      exponent_sign = static_cast<char>(*current);
      ++current;
      if (current == end) {
        if (allow_trailing_junk) {
          goto parsing_done;
        } else {
          return JunkStringValue();
        }
      }
    }

    if (current == end || *current < '0' || *current > '9') {
      if (allow_trailing_junk) {
        goto parsing_done;
      } else {
        return JunkStringValue();
      }
    }

    const int max_exponent = INT_MAX / 2;
    DCHECK(-max_exponent / 2 <= exponent && exponent <= max_exponent / 2);
    int num = 0;
    do {
      // Check overflow.
      int digit = *current - '0';
      if (num >= max_exponent / 10 &&
          !(num == max_exponent / 10 && digit <= max_exponent % 10)) {
        num = max_exponent;
      } else {
        num = num * 10 + digit;
      }
      ++current;
    } while (current != end && *current >= '0' && *current <= '9');

    exponent += (exponent_sign == '-' ? -num : num);
  }

  if (!allow_trailing_junk && AdvanceToNonspace(&current, end)) {
    return JunkStringValue();
  }

parsing_done:
  exponent += insignificant_digits;

  if (nonzero_digit_dropped) {
    buffer[buffer_pos++] = '1';
    exponent--;
  }

  SLOW_DCHECK(buffer_pos < kBufferSize);
  buffer[buffer_pos] = '\0';

  double converted =
      Strtod(Vector<const char>(buffer, buffer_pos), exponent);
  return (sign == Sign::kNegative) ? -converted : converted;
}
double StringToDoubleV8(const char16_t* begin, const char16_t* end) {
  return InternalStringToDoubleV8(reinterpret_cast<const uint16_t*>(begin), reinterpret_cast<const uint16_t*>(end), ALLOW_TRAILING_JUNK,
                                 std::numeric_limits<double>::quiet_NaN());
}
double StringToDoubleV8(const char* begin, const char* end) {
  return InternalStringToDoubleV8(reinterpret_cast<const uint8_t*>(begin), reinterpret_cast<const uint8_t*>(end), ALLOW_TRAILING_JUNK,
                                 std::numeric_limits<double>::quiet_NaN());
}

template <typename CharT>
double findmax_v8(std::vector<std::basic_string<CharT>> &s) {
  double answer = 0;
  double x = 0;
  for (auto &st : s) {
    x = StringToDoubleV8(st.data(), st.data() + st.size());
    answer = answer > x ? answer : x;
  }
  return answer;
}

#ifdef USING_COUNTERS
template <class T, class CharT>
std::vector<event_count> time_it_ns(std::vector<std::basic_string<CharT>> &lines,
                                     T const &function, size_t repeat) {
  std::vector<event_count> aggregate;
  event_collector collector;
  bool printed_bug = false;
  for (size_t i = 0; i < repeat; i++) {
    collector.start();
    double ts = function(lines);
    if (ts == 0 && !printed_bug) {
      printf("bug\n");
      printed_bug = true;
    }
    aggregate.push_back(collector.end());
  }
  return aggregate;
}

void pretty_print(double volume, size_t number_of_floats, std::string name, std::vector<event_count> events) {
  double volumeMB = volume / (1024. * 1024.);
  double average_ns{0};
  double min_ns{DBL_MAX};
  double cycles_min{DBL_MAX};
  double instructions_min{DBL_MAX};
  double cycles_avg{0};
  double instructions_avg{0};
  double branches_min{0};
  double branches_avg{0};
  double branch_misses_min{0};
  double branch_misses_avg{0};
  for(event_count e : events) {
    double ns = e.elapsed_ns();
    average_ns += ns;
    min_ns = min_ns < ns ? min_ns : ns;

    double cycles = e.cycles();
    cycles_avg += cycles;
    cycles_min = cycles_min < cycles ? cycles_min : cycles;

    double instructions = e.instructions();
    instructions_avg += instructions;
    instructions_min = instructions_min < instructions ? instructions_min : instructions;

    double branches = e.branches();
    branches_avg += branches;
    branches_min = branches_min < branches ? branches_min : branches;

    double branch_misses = e.missed_branches();
    branch_misses_avg += branch_misses;
    branch_misses_min = branch_misses_min < branch_misses ? branch_misses_min : branch_misses;
  }
  cycles_avg /= events.size();
  instructions_avg /= events.size();
  average_ns /= events.size();
  branches_avg /= events.size();
  printf("%-40s: %8.2f MB/s (+/- %.1f %%) ", name.data(),
           volumeMB * 1000000000 / min_ns,
           (average_ns - min_ns) * 100.0 / average_ns);
  printf("%8.2f Mfloat/s  ", 
           number_of_floats * 1000 / min_ns);
  if(instructions_min > 0) {
    printf(" %8.2f i/B %8.2f i/f (+/- %.1f %%) ", 
           instructions_min / volume,
           instructions_min / number_of_floats, 
           (instructions_avg - instructions_min) * 100.0 / instructions_avg);

    printf(" %8.2f c/B %8.2f c/f (+/- %.1f %%) ", 
           cycles_min / volume,
           cycles_min / number_of_floats, 
           (cycles_avg - cycles_min) * 100.0 / cycles_avg);
    printf(" %8.2f i/c ", 
           instructions_min /cycles_min);
    printf(" %8.2f b/f ",
           branches_avg /number_of_floats);
    printf(" %8.2f bm/f ",
           branch_misses_avg /number_of_floats);
    printf(" %8.2f GHz ", 
           cycles_min / min_ns);
  }
  printf("\n");

}
#else
template <class T, class CharT>
std::pair<double, double> time_it_ns(std::vector<std::basic_string<CharT>> &lines,
                                     T const &function, size_t repeat) {
  std::chrono::high_resolution_clock::time_point t1, t2;
  double average = 0;
  double min_value = DBL_MAX;
  bool printed_bug = false;
  for (size_t i = 0; i < repeat; i++) {
    t1 = std::chrono::high_resolution_clock::now();
    double ts = function(lines);
    if (ts == 0 && !printed_bug) {
      printf("bug\n");
      printed_bug = true;
    }
    t2 = std::chrono::high_resolution_clock::now();
    double dif =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    average += dif;
    min_value = min_value < dif ? min_value : dif;
  }
  average /= repeat;
  return std::make_pair(min_value, average);
}




void pretty_print(double volume, size_t number_of_floats, std::string name, std::pair<double,double> result) {
  double volumeMB = volume / (1024. * 1024.);
  printf("%-40s: %8.2f MB/s (+/- %.1f %%) ", name.data(),
           volumeMB * 1000000000 / result.first,
           (result.second - result.first) * 100.0 / result.second);
  printf("%8.2f Mfloat/s  ", 
           number_of_floats * 1000 / result.first);
  printf(" %8.2f ns/f \n", 
           double(result.first) /number_of_floats );
}
#endif 


// this is okay, all chars are ASCII
inline std::u16string widen(std::string line) {
  std::u16string u16line;
  u16line.resize(line.size());
  for (size_t i = 0; i < line.size(); ++i) {
    u16line[i] = char16_t(line[i]);
  }
  return u16line;
}

std::vector<std::u16string> widen(const std::vector<std::string> &lines) {
  std::vector<std::u16string> u16lines;
  u16lines.reserve(lines.size());
  for (auto const &line : lines) {
    u16lines.push_back(widen(line));
  }
  return u16lines;
}


void process(std::vector<std::string> &lines, size_t volume) {
  size_t repeat = 100;
  double volumeMB = volume / (1024. * 1024.);
  std::cout << "ASCII volume = " << volumeMB << " MB " << std::endl;
  pretty_print(volume, lines.size(), "netlib", time_it_ns(lines, findmax_netlib, repeat));
  pretty_print(volume, lines.size(), "doubleconversion", time_it_ns(lines, findmax_doubleconversion<char>, repeat));
  pretty_print(volume, lines.size(), "strtod", time_it_ns(lines, findmax_strtod, repeat));
#ifdef ENABLE_RYU
  pretty_print(volume, lines.size(), "ryu_parse", time_it_ns(lines, findmax_ryus2d, repeat));
#endif
#ifndef __CYGWIN__
  pretty_print(volume, lines.size(), "abseil", time_it_ns(lines, findmax_absl_from_chars, repeat));
#endif
  pretty_print(volume, lines.size(), "fastfloat", time_it_ns(lines, findmax_fastfloat<char>, repeat));
  pretty_print(volume, lines.size(), "v8", time_it_ns(lines, findmax_v8<char>, repeat));
#ifdef FROM_CHARS_AVAILABLE_MAYBE
  pretty_print(volume, lines.size(), "from_chars", time_it_ns(lines, findmax_from_chars, repeat));
#endif
  std::vector<std::u16string> lines16 = widen(lines);
  volume = 2 * volume;
  volumeMB = volume / (1024. * 1024.);
  std::cout << "UTF-16 volume = " << volumeMB << " MB " << std::endl;
  pretty_print(volume, lines.size(), "doubleconversion", time_it_ns(lines16, findmax_doubleconversion<char16_t>, repeat));
#ifdef _WIN32
  pretty_print(volume, lines.size(), "wcstod", time_it_ns(lines16, findmax_strtod_16, repeat));
#endif
  pretty_print(volume, lines.size(), "fastfloat", time_it_ns(lines16, findmax_fastfloat<char16_t>, repeat));
  pretty_print(volume, lines.size(), "v8", time_it_ns(lines16, findmax_v8<char16_t>, repeat));
}

void fileload(const char *filename) {
  std::ifstream inputfile(filename);
  if (!inputfile) {
    std::cerr << "can't open " << filename << std::endl;
    return;
  }
  std::string line;
  std::vector<std::string> lines;
  lines.reserve(10000); // let us reserve plenty of memory.
  size_t volume = 0;
  while (getline(inputfile, line)) {
    volume += line.size();
    lines.push_back(line);
  }
  std::cout << "# read " << lines.size() << " lines " << std::endl;
  process(lines, volume);
}


void parse_random_numbers(size_t howmany, bool concise, std::string random_model) {
  std::cout << "# parsing random numbers" << std::endl;
  std::vector<std::string> lines;
  auto g = std::unique_ptr<string_number_generator>(get_generator_by_name(random_model));
  std::cout << "model: " << g->describe() << std::endl;
  if(concise) { std::cout << "concise (using as few digits as possible)"  << std::endl; }
  std::cout << "volume: "<< howmany << " floats"  << std::endl;
  lines.reserve(howmany); // let us reserve plenty of memory.
  size_t volume = 0;
  for (size_t i = 0; i < howmany; i++) {
    std::string line =  g->new_string(concise);
    volume += line.size();
    lines.push_back(line);
  }
  process(lines, volume);
}

void parse_contrived(size_t howmany, const char *filename) {
  std::cout << "# parsing contrived numbers" << std::endl;
  std::cout << "# these are contrived test cases to test specific algorithms" << std::endl;
  std::vector<std::string> lines;
  std::cout << "volume: "<< howmany << " floats"  << std::endl;

  std::ifstream inputfile(filename);
  if (!inputfile) {
    std::cerr << "can't open " << filename << std::endl;
    return;
  }
  lines.reserve(howmany); // let us reserve plenty of memory.
  std::string line;
  while (getline(inputfile, line)) {
    std::cout << "testing contrived case: \"" << line << "\"" << std::endl;
    size_t volume = 0;
    for (size_t i = 0; i < howmany; i++) {
      lines.push_back(line);
    }
    process(lines, volume);
    lines.clear();
    std::cout << "-----------------------" << std::endl;
  }
}

cxxopts::Options
    options("benchmark",
            "Compute the parsing speed of different number parsers.");

int main(int argc, char **argv) {
  try {
    options.add_options()
        ("c,concise", "Concise random floating-point strings (if not 17 digits are used)")
        ("f,file", "File name.", cxxopts::value<std::string>()->default_value(""))
        ("v,volume", "Volume (number of floats generated).", cxxopts::value<size_t>()->default_value("100000"))
        ("m,model", "Random Model.", cxxopts::value<std::string>()->default_value("uniform"))
        ("h,help","Print usage.");
    auto result = options.parse(argc, argv);
    if(result["help"].as<bool>()) {
      std::cout << options.help() << std::endl;
      return EXIT_SUCCESS;
    }
    auto filename = result["file"].as<std::string>();
    if (filename.find("contrived") != std::string::npos) {
      parse_contrived(result["volume"].as<size_t>(), filename.c_str());
      std::cout << "# You can also provide a filename (with the -f flag): it should contain one "
                   "string per line corresponding to a number"
                << std::endl;
    } else if (filename.empty()) {
      parse_random_numbers(result["volume"].as<size_t>(), result["concise"].as<bool>(), result["model"].as<std::string>());
      std::cout << "# You can also provide a filename (with the -f flag): it should contain one "
                   "string per line corresponding to a number"
                << std::endl;
    } else {
      fileload(filename.c_str());
    }
  } catch (const cxxopts::OptionException &e) {
    std::cout << "error parsing options: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
