/**
 * @file modules/stage_registry.cpp
 * @brief Storage and lookup for the stage header-factory registry.
 *
 * The map is a function-local static so it is initialized before any stage's
 * static-init registrar runs (construct-on-first-use). Each stage `.cu`
 * contributes an entry via FZ_REGISTER_STAGE_FACTORY / FZ_REGISTER_SIMPLE_STAGE;
 * createStage() replaces what used to be a hand-maintained switch.
 */
#include "stage/stage_registry.h"

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace fz {

namespace {
std::unordered_map<uint16_t, StageHeaderFactory>& registry() {
    static std::unordered_map<uint16_t, StageHeaderFactory> m;
    return m;
}
}  // namespace

void registerStageHeaderFactory(StageType type, StageHeaderFactory fn) {
    // Last registration wins; duplicate StageTypes are a build-time programming
    // error, but replacing keeps behavior deterministic rather than order-dependent.
    registry()[static_cast<uint16_t>(type)] = fn;
}

bool hasStageHeaderFactory(StageType type) {
    auto& m = registry();
    auto it = m.find(static_cast<uint16_t>(type));
    return it != m.end() && it->second != nullptr;
}

Stage* createStage(StageType type, const uint8_t* config, size_t config_size) {
    auto& m = registry();
    auto it = m.find(static_cast<uint16_t>(type));
    if (it == m.end() || it->second == nullptr) {
        throw std::runtime_error("createStage: no factory registered for stage type "
                                 + std::to_string(static_cast<uint16_t>(type)));
    }
    return it->second(config, config_size);
}

}  // namespace fz
