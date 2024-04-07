#ifndef ELEMENTWISE_INTERFACE_H_
#define ELEMENTWISE_INTERFACE_H_

#include <cstdint>
#include <cstddef>

namespace ew {
template<typename T, size_t N>
struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};
} // namespace ew

#endif // ELEMENTWISE_INTERFACE_H_