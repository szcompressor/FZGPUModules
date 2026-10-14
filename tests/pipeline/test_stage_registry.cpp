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
    // Every shipped stage has a factory; reserved development IDs remain unavailable.
    const StageType shipped[] = {
        StageType::LORENZO_QUANT, StageType::DIFFERENCE, StageType::RLE,
        StageType::HUFFMAN, StageType::BITPACK, StageType::MERGE,
        StageType::LORENZO, StageType::QUANTIZER, StageType::ZIGZAG,
        StageType::NEGABINARY, StageType::BITSHUFFLE, StageType::RZE,
        StageType::ANS, StageType::G_INTERP,
        StageType::BITPLANE_RZE, StageType::ADAPTIVE_BITPACK,
        StageType::TILED_LORENZO, StageType::RRE, StageType::RARE,
        StageType::RAZE, StageType::CLOG, StageType::HCLOG, StageType::TUPL,
        StageType::GPULZ, StageType::LOG_TRANSFORM, StageType::ADAPTIVE_LORENZO,
        StageType::ROIBIN_SPLIT,
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

TEST(StageRegistry, DevelopmentStageIdsStayUnavailable) {
    const StageType types[] = {StageType::SZX, StageType::SZP, StageType::CDF97,
        StageType::SPECK2D, StageType::CDF97_OUTLIER_CORRECT,
        StageType::FUSED_QUANT_ADAPTIVE_LORENZO};
    const uint16_t ids[] = {36, 37, 39, 40, 41, 43};
    for (size_t i = 0; i < 6; ++i) {
        const auto type = types[i];
        EXPECT_EQ(static_cast<uint16_t>(type), ids[i]);
        EXPECT_FALSE(hasStageHeaderFactory(type));
        try {
            createStage(type, nullptr, 0);
            FAIL() << "development archive stage was accepted";
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find(stageTypeToString(type)), std::string::npos);
            EXPECT_NE(message.find("development branch"), std::string::npos);
            EXPECT_NE(message.find(std::to_string(ids[i])), std::string::npos);
        }
    }
}
