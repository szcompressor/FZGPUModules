/**
 * @file test_module_cards.cpp
 * @brief Registry coverage test for module cards.
 *
 * Every `card.json` in the modules/ tree must be registered with at least one
 * concrete stage instantiation so cards can never silently go unvalidated.
 *
 * Adding a card: register a representative instantiation in `registry()` below.
 * `EveryCardHasARepresentative` fails if a `card.json` exists with no entry.
 *
 * NOTE: The TOML contract-validation phase (ContractMatchesInterface) was removed
 * because toml++ triggered a SIGSEGV under nvc++ -O2 -DNDEBUG. Cards were
 * migrated to JSON (card.json); a JSON-parsing validation phase can be added
 * without the toml++ dependency.
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "stage/stage.h"
#include "fzm_format.h"
#include "predictors/tiled_lorenzo/tiled_lorenzo_stage.h"
#include "fused/ginterp/ginterp_stage.h"

#ifndef FZ_MODULES_DIR
#error "FZ_MODULES_DIR must be defined by CMake (path to the modules/ tree)."
#endif

using namespace fz;
namespace fs = std::filesystem;

namespace {

// ── A carded stage + the concrete instantiations that validate its card ──────
struct CardEntry {
    std::string card_rel_path;   // relative to FZ_MODULES_DIR
    std::vector<std::function<std::unique_ptr<Stage>()>> make_reps;
};

// The single source of truth for which cards exist and how to instantiate them.
// Every card.json in the tree MUST appear here (enforced below).
std::vector<CardEntry> registry() {
    std::vector<CardEntry> r;
    r.push_back({"predictors/tiled_lorenzo/card.json", {
        [] { return std::make_unique<TiledLorenzoStage<int16_t>>(); },
        [] { return std::make_unique<TiledLorenzoStage<int32_t>>(); },
    }});
    r.push_back({"fused/ginterp/card.json", {
        [] { return std::make_unique<GInterpStage<float,  uint16_t>>(); },
        [] { return std::make_unique<GInterpStage<double, uint16_t>>(); },
    }});
    return r;
}

}  // namespace

// Every card.json on disk must be registered with a representative — otherwise
// it ships unvalidated.  Catches "added a card, forgot to register it".
TEST(ModuleCards, EveryCardHasARepresentative) {
    std::set<std::string> registered;
    for (const auto& e : registry()) registered.insert(e.card_rel_path);

    const fs::path root = FZ_MODULES_DIR;
    ASSERT_TRUE(fs::is_directory(root)) << "FZ_MODULES_DIR not a directory: " << root;

    size_t found = 0;
    for (const auto& p : fs::recursive_directory_iterator(root)) {
        if (p.path().filename() != "card.json") continue;
        ++found;
        const std::string rel = fs::relative(p.path(), root).generic_string();
        EXPECT_TRUE(registered.count(rel))
            << "card.json has no representative in test_module_cards.cpp registry(): " << rel;
    }
    EXPECT_EQ(found, registered.size())
        << "registry() lists " << registered.size()
        << " cards but " << found << " card.json files exist on disk";
}
