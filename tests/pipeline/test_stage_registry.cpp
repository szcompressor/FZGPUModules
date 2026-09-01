/**
 * tests/pipeline/test_stage_registry.cpp
 *
 * Guards the self-registration mechanism (include/stage/stage_registry.h): every
 * shipped StageType must have a header factory registered by its module's `.cu`
 * static initializer. A registrar dropped by the linker (e.g. a static-archive
 * build without --whole-archive) would surface here rather than as a corrupt
 * decode of one specific archive in the field.
 */

#include <gtest/gtest.h>
#include "fzgpumodules.h"
#include "stage/stage_registry.h"

using namespace fz;

TEST(StageRegistry, EveryShippedStageTypeHasAFactory) {
    // Every value that stageTypeToString() names as a real stage. UNKNOWN and the
    // reserved-but-unimplemented slots (SCALE/PASSTHROUGH/SPLIT) are excluded.
    const StageType shipped[] = {
        StageType::LORENZO_QUANT, StageType::DIFFERENCE, StageType::RLE,
        StageType::HUFFMAN, StageType::BITPACK, StageType::MERGE,
        StageType::LORENZO, StageType::QUANTIZER, StageType::ZIGZAG,
        StageType::NEGABINARY, StageType::BITSHUFFLE, StageType::RZE,
        StageType::ANS, StageType::ADM, StageType::G_INTERP,
        StageType::BITPLANE_RZE, StageType::ADAPTIVE_BITPACK,
        StageType::TILED_LORENZO, StageType::RRE, StageType::RARE,
        StageType::RAZE, StageType::CLOG, StageType::HCLOG, StageType::TUPL,
        StageType::GPULZ, StageType::LOG_TRANSFORM, StageType::ADAPTIVE_LORENZO,
        StageType::ROIBIN_SPLIT, StageType::SZX,
        StageType::CDF97, StageType::SPECK2D, StageType::CDF97_OUTLIER_CORRECT,
        // SZP is a quarantined experimental reference compressor
        // (experimental/reference_compressors/szp) — no longer a public module,
        // but its header factory MUST stay linked so pre-existing FZM archives
        // remain decodable. This is the regression guard for that guarantee.
        StageType::SZP,
    };
    for (StageType t : shipped) {
        EXPECT_TRUE(hasStageHeaderFactory(t))
            << "no header factory registered for StageType "
            << static_cast<int>(static_cast<uint16_t>(t))
            << " (" << stageTypeToString(t) << ") — its module's REGISTER_STAGE "
            << "registrar was not linked in";
    }
}

TEST(StageRegistry, UnknownTypeThrows) {
    uint8_t cfg[1] = {0};
    EXPECT_THROW(createStage(StageType::UNKNOWN, cfg, 0), std::runtime_error);
}
