#ifndef __UTIL_H__
#define __UTIL_H__

#include <cstdint>
#include <iostream>
#include <memory>

/**
 * Helper utility to allocated aligned arrays in pre-allocated buffer.
 */
class WorkspaceBuffer {
public:
  WorkspaceBuffer(uint8_t *base, size_t max_size)
      : base_(base), end_(base + max_size), cur_(base) {}

  uint8_t *allocate_raw(size_t size, size_t align = 32) {
    uint8_t *start = reinterpret_cast<uint8_t *>(((uintptr_t)cur_ + align - 1) &
                                                 ~(align - 1));

    cur_ = start; // bump cur to starting offset

    if (start >= end_ || start + size >= end_) {
      std::cerr << "allocate_raw failed: size=" << size << " align=" << align
                << " start=" << (void *)start << " end_ = " << (void *)end_
                << "\n";
      return nullptr;
    }

    cur_ += size;
    return start;
  }

  template <class T, size_t Align = 32> T *allocate(size_t n) {
    static_assert(Align >= std::alignment_of<T>(), "Alignment too small");
    static_assert(Align % std::alignment_of<T>() == 0,
                  "Type must be minimally aligned.");
    uint8_t *p = allocate_raw(sizeof(T) * n, Align);
    if (p == nullptr) {
      return nullptr;
    }
    return reinterpret_cast<T *>(p);
  }

private:
  uint8_t *base_;
  uint8_t *end_;
  uint8_t *cur_;
};

#endif
