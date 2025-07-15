#pragma once
#include <cuda_runtime.h>

namespace {
dim3 div3(dim3 l, dim3 subl) {
  return dim3((l.x - 1) / subl.x + 1, (l.y - 1) / subl.y + 1,
              (l.z - 1) / subl.z + 1);
};
}  // namespace

namespace fz {
template <
  uint16_t _TileDim, uint8_t _Seq,  // required
  uint8_t _TileDimY = (uint8_t)_TileDim, uint8_t _SeqY = _Seq,
  uint8_t _TileDimZ = (uint8_t)_TileDim, uint8_t _SeqZ = _Seq>
struct PredPerf {
  static const uint16_t TileDim = _TileDim;
  static const uint16_t TiledimX = TileDim;
  static const uint8_t TiledimY = _TileDimY;
  static const uint8_t TiledimZ = _TileDimZ;

  static const uint8_t Seq = _Seq;
  static const uint8_t SeqX = Seq;
  static const uint8_t SeqY = _SeqY;
  static const uint8_t SeqZ = _SeqZ;

  static_assert(Seq < 16, "Sequentiality must be less than 16.");
};
}  // namespace fz

namespace fz::kernelconfig {

struct lorenzo_utils {
  static int ndim(dim3 len3)
  {
    if (len3.z == 1 and len3.y == 1)
      return 1;
    else if (len3.z == 1 and len3.y != 1)
      return 2;
    else
      return 3;
  };

  static int ndim(std::array<size_t, 3> len3)
  {
    if (len3[2] == 1 and len3[1] == 1)
      return 1;
    else if (len3[2] == 1 and len3[1] != 1)
      return 2;
    else
      return 3;
  };
};

template <int dim, int X = 0, int Y = 0>
struct c_lorenzo;

template <int dim, int X = 0, int Y = 0>
struct x_lorenzo;

template <>
struct c_lorenzo<1> {
  static constexpr dim3 tile = dim3(1024, 1, 1);
  static constexpr dim3 sequentiality = dim3(4, 1, 1);  // x-sequentiality == 4
  static constexpr dim3 seq = sequentiality;
  static constexpr dim3 thread_block = dim3(1024 / 4, 1, 1);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  using Perf = PredPerf<tile.x, seq.x, tile.y, seq.y, tile.z, seq.z>;
};

template <>
struct c_lorenzo<2> {
  static constexpr dim3 tile = dim3(16, 16, 1);
  static constexpr dim3 sequentiality = dim3(1, 8, 1);  // y-sequentiality == 8
  static constexpr dim3 seq = sequentiality;
  static constexpr dim3 thread_block = dim3(16, 2, 1);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  using Perf = PredPerf<tile.x, seq.x, tile.y, seq.y, tile.z, seq.z>;
};

template <>
struct c_lorenzo<2, 32, 32> {
  static constexpr dim3 tile = dim3(32, 32, 1);
  static constexpr dim3 sequentiality = dim3(1, 8, 1);  // y-sequentiality == 8
  static constexpr dim3 seq = sequentiality;
  static constexpr dim3 thread_block = dim3(32, 4, 1);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  static_assert(thread_block.x * sequentiality.x == tile.x);
  static_assert(thread_block.y * sequentiality.y == tile.y);

  using Perf = PredPerf<tile.x, seq.x, tile.y, seq.y, tile.z, seq.z>;
};

template <>
struct c_lorenzo<2, 64, 32> {                           // for uint16_t
  static constexpr dim3 tile = dim3(64, 32, 1);         // 2-unit alignment
  static constexpr dim3 sequentiality = dim3(2, 8, 1);  // y-sequentiality == 8
  static constexpr dim3 seq = sequentiality;
  static constexpr dim3 thread_block = dim3(32, 4, 1);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  static_assert(thread_block.x * sequentiality.x == tile.x);
  static_assert(thread_block.y * sequentiality.y == tile.y);

  using Perf = PredPerf<tile.x, seq.x, tile.y, seq.y, tile.z, seq.z>;
};

template <>
struct c_lorenzo<2, 128, 32> {                          // for uint8_t
  static constexpr dim3 tile = dim3(128, 32, 1);        // 4-unit x-alignment
  static constexpr dim3 sequentiality = dim3(4, 8, 1);  // y-sequentiality == 8
  static constexpr dim3 seq = sequentiality;
  static constexpr dim3 thread_block = dim3(32, 4, 1);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  static_assert(thread_block.x * sequentiality.x == tile.x);
  static_assert(thread_block.y * sequentiality.y == tile.y);

  using Perf = PredPerf<tile.x, seq.x, tile.y, seq.y, tile.z, seq.z>;
};

template <>
struct c_lorenzo<3> {
  static constexpr dim3 tile = dim3(32, 8, 8);
  static constexpr dim3 sequentiality = dim3(1, 1, 8);  // z-sequentiality == 8
  static constexpr dim3 seq = sequentiality;
  static constexpr dim3 thread_block = dim3(32, 8, 1);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  using Perf = PredPerf<8, seq.x, 8, seq.y, 8, seq.z>;
};

template <>
struct x_lorenzo<1> {
  static constexpr dim3 tile = dim3(1024, 1, 1);
  static constexpr dim3 sequentiality = dim3(4, 1, 1);  // x-sequentiality == 8
  static constexpr dim3 seq = sequentiality;
  static constexpr dim3 thread_block = dim3(1024 / 4, 1, 1);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  using Perf = PredPerf<tile.x, seq.x, tile.y, seq.y, tile.z, seq.z>;
};

template <>
struct x_lorenzo<2> {
  static constexpr dim3 tile = dim3(16, 16, 1);
  static constexpr dim3 sequentiality = dim3(1, 8, 1);  // y-sequentiality == 8
  static constexpr dim3 seq = sequentiality;
  static constexpr dim3 thread_block = dim3(16, 2, 1);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  using Perf = PredPerf<tile.x, seq.x, tile.y, seq.y, tile.z, seq.z>;
};

template <>
struct x_lorenzo<2, 32> {
  static constexpr dim3 tile = dim3(32, 32, 1);
  static constexpr dim3 sequentiality = dim3(1, 8, 1);  // y-sequentiality == 8
  static constexpr dim3 seq = sequentiality;
  static constexpr dim3 thread_block = dim3(32, 4, 1);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  using Perf = PredPerf<tile.x, seq.x, tile.y, seq.y, tile.z, seq.z>;
};

template <>
struct x_lorenzo<3> {
  static constexpr dim3 tile = dim3(32, 8, 8);
  static constexpr dim3 sequentiality = dim3(1, 8, 1);  // y-sequentiality == 8
  static constexpr dim3 seq = sequentiality;
  static constexpr dim3 thread_block = dim3(32, 1, 8);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };

  using Perf = PredPerf<tile.x, seq.x, tile.y, seq.y, tile.z, seq.z>;
};

};  // namespace fz::kernelconfig