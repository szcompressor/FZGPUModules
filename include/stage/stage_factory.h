#pragma once

/**
 * @file stage_factory.h
 * @brief Backward-compatible shim.
 *
 * `createStage()` and the stage reconstruction mechanism now live in
 * stage/stage_registry.h — stages self-register their FZM-header factory, so
 * there is no central switch here any more. This header remains only so that
 * existing includes of "stage/stage_factory.h" keep compiling.
 */

#include "stage/stage_registry.h"
