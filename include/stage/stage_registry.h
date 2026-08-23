#pragma once

/**
 * @file stage_registry.h
 * @brief Central registry that reconstructs a Stage from a serialized FZM header.
 *
 * Stages self-register their reconstruction function at static-init time via
 * FZ_REGISTER_STAGE_FACTORY / FZ_REGISTER_SIMPLE_STAGE (placed at file scope in
 * the stage's own `.cu`). The decompressor calls createStage(), which looks the
 * StageType up in the registry — there is no central switch to edit when a new
 * stage is added.
 *
 * Adding a stage therefore touches only the stage's own directory for this axis;
 * see docs/how_to_add_a_stage.md.
 */

#include "stage/stage.h"
#include "fzm_format.h"

#include <cstddef>
#include <cstdint>

namespace fz {

/**
 * Build a Stage from its serialized config bytes. A factory owns the type
 * dispatch (e.g. picking the right template instantiation from the config) and
 * must call deserializeHeader() itself. It returns a heap-allocated Stage the
 * caller owns, or throws std::runtime_error on an unsupported config.
 */
using StageHeaderFactory = Stage* (*)(const uint8_t* config, size_t config_size);

/** Register (or, for a duplicate StageType, replace) a header factory. */
void registerStageHeaderFactory(StageType type, StageHeaderFactory fn);

/** True if a header factory is registered for `type`. For coverage tests. */
bool hasStageHeaderFactory(StageType type);

/**
 * Reconstruct a Stage from a serialized FZM header. Used by the decompressor to
 * rebuild the inverse pipeline from the file. Throws if no factory is
 * registered for `type`.
 */
Stage* createStage(StageType type, const uint8_t* config, size_t config_size);

namespace detail {
/** RAII-free registrar: constructing one registers `fn` for `type`. */
struct StageFactoryRegistrar {
    StageFactoryRegistrar(StageType type, StageHeaderFactory fn) {
        registerStageHeaderFactory(type, fn);
    }
};
}  // namespace detail

}  // namespace fz

/* Token-paste helpers so __LINE__ expands before concatenation. */
#define FZ_STAGE_CONCAT_(a, b) a##b
#define FZ_STAGE_CONCAT(a, b) FZ_STAGE_CONCAT_(a, b)

/**
 * Register a custom header factory for a StageType. Place at file scope in the
 * stage's `.cu`, after the factory function is defined:
 *
 *   static fz::Stage* MyStage_fromHeader(const uint8_t* c, size_t n) { ... }
 *   FZ_REGISTER_STAGE_FACTORY(fz::StageType::MY_STAGE, MyStage_fromHeader);
 */
#define FZ_REGISTER_STAGE_FACTORY(TYPE, FN)                                     \
    namespace {                                                                 \
    const ::fz::detail::StageFactoryRegistrar                                   \
        FZ_STAGE_CONCAT(FZ_STAGE_FACTORY_REGISTRAR_, __LINE__){(TYPE), (FN)};                    \
    }

/**
 * Convenience for a stage with no template dispatch — reconstruction is just
 * `new StageClass()` + deserializeHeader(). Place at file scope in the `.cu`:
 *
 *   FZ_REGISTER_SIMPLE_STAGE(fz::StageType::RZE, fz::RZEStage);
 */
#define FZ_REGISTER_SIMPLE_STAGE(TYPE, STAGE_CLASS)                             \
    namespace {                                                                 \
    ::fz::Stage* FZ_STAGE_CONCAT(FZ_STAGE_SIMPLE_FACTORY_, __LINE__)(const uint8_t* config,      \
                                                    size_t config_size) {       \
        auto* s = new STAGE_CLASS();                                            \
        s->deserializeHeader(config, config_size);                             \
        return s;                                                               \
    }                                                                           \
    const ::fz::detail::StageFactoryRegistrar                                   \
        FZ_STAGE_CONCAT(FZ_STAGE_SIMPLE_REGISTRAR_, __LINE__){(TYPE),                            \
            &FZ_STAGE_CONCAT(FZ_STAGE_SIMPLE_FACTORY_, __LINE__)};                              \
    }
