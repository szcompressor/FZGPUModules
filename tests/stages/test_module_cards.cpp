/**
 * @file test_module_cards.cpp
 * @brief Contract-validation test for module cards (memory/module_cards.md §4).
 *
 * Each `card.toml` declares machine-checkable facts about a stage in its
 * `[identity]` and `[facets.contract]` / `[facets.data]` blocks.  This test
 * instantiates representative concrete versions of each carded stage and
 * asserts the card matches the *live* `Stage` interface — so a card can never
 * silently drift from the code (the "anti-rot" mechanism that distinguishes
 * cards from documentation that rots).
 *
 * Adding a card: register a representative instantiation in `registry()` below.
 * `EveryCardHasARepresentative` fails if a `card.toml` exists with no entry,
 * so breadth stays enforced as the catalog grows.
 */

#include <gtest/gtest.h>
#include <toml++/toml.hpp>

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
// Every card.toml in the tree MUST appear here (enforced below).
std::vector<CardEntry> registry() {
    std::vector<CardEntry> r;
    r.push_back({"predictors/tiled_lorenzo/card.toml", {
        [] { return std::make_unique<TiledLorenzoStage<int16_t>>(); },
        [] { return std::make_unique<TiledLorenzoStage<int32_t>>(); },
    }});
    r.push_back({"fused/ginterp/card.toml", {
        [] { return std::make_unique<GInterpStage<float,  uint16_t>>(); },
        [] { return std::make_unique<GInterpStage<double, uint16_t>>(); },
    }});
    return r;
}

// Map a DataType to the element-type token used in [facets].data.
// Returns "" for types that opt out of the check (byte-transparent / unknown).
std::string elementFacetToken(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32: return "float32";
        case DataType::FLOAT64: return "float64";
        case DataType::INT16:
        case DataType::INT32:   return "signed-int";
        case DataType::UINT16:
        case DataType::UINT32:   return "integer";
        default:                 return "";  // UNKNOWN etc.
    }
}

template <typename NodeView>
std::set<std::string> toStringSet(const NodeView& n) {
    std::set<std::string> out;
    if (const toml::array* arr = n.as_array()) {
        for (const toml::node& el : *arr)
            if (auto s = el.value<std::string>()) out.insert(*s);
    }
    return out;
}

fs::path cardPath(const std::string& rel) { return fs::path(FZ_MODULES_DIR) / rel; }

}  // namespace

// Every card.toml on disk must be registered with a representative — otherwise
// it ships unvalidated.  Catches "added a card, forgot to register it".
TEST(ModuleCards, EveryCardHasARepresentative) {
    std::set<std::string> registered;
    for (const auto& e : registry()) registered.insert(e.card_rel_path);

    const fs::path root = FZ_MODULES_DIR;
    ASSERT_TRUE(fs::is_directory(root)) << "FZ_MODULES_DIR not a directory: " << root;

    size_t found = 0;
    for (const auto& p : fs::recursive_directory_iterator(root)) {
        if (p.path().filename() != "card.toml") continue;
        ++found;
        const std::string rel = fs::relative(p.path(), root).generic_string();
        EXPECT_TRUE(registered.count(rel))
            << "card.toml has no representative in test_module_cards.cpp registry(): " << rel;
    }
    EXPECT_EQ(found, registered.size())
        << "registry() lists " << registered.size()
        << " cards but " << found << " card.toml files exist on disk";
}

// The card's machine-checkable facets must match the live Stage interface.
TEST(ModuleCards, ContractMatchesInterface) {
    for (const auto& entry : registry()) {
        SCOPED_TRACE("card: " + entry.card_rel_path);
        const fs::path path = cardPath(entry.card_rel_path);
        ASSERT_TRUE(fs::exists(path)) << "missing card: " << path;

        toml::table card;
        ASSERT_NO_THROW(card = toml::parse_file(path.string()))
            << "card.toml failed to parse: " << path;

        const auto identity = card["identity"];
        const auto expect_name = identity["get_name"].value<std::string>();
        const auto expect_id   = identity["stage_type_id"].value<int64_t>();
        ASSERT_TRUE(expect_name) << "[identity].get_name missing";
        ASSERT_TRUE(expect_id)   << "[identity].stage_type_id missing";

        const std::set<std::string> contract = toStringSet(card["facets"]["contract"]);
        const std::set<std::string> data     = toStringSet(card["facets"]["data"]);

        for (const auto& make : entry.make_reps) {
            std::unique_ptr<Stage> s = make();  // forward mode (is_inverse_ = false)
            SCOPED_TRACE("stage: " + s->getName());

            // identity
            EXPECT_EQ(s->getName(), *expect_name);
            EXPECT_EQ(static_cast<int64_t>(s->getStageTypeId()), *expect_id);

            // contract: multi-output  <->  >1 output port
            const bool multi = s->getOutputNames().size() > 1;
            EXPECT_EQ(multi, contract.count("multi-output") > 0)
                << "multi-output facet vs getOutputNames().size()=" << s->getOutputNames().size();

            // contract: graph-capturable  <->  isGraphCompatible() (forward path)
            EXPECT_EQ(s->isGraphCompatible(), contract.count("graph-capturable") > 0)
                << "graph-capturable facet vs isGraphCompatible() (forward)";

            // data: the input element type must be advertised in [facets].data
            const std::string tok = elementFacetToken(
                static_cast<DataType>(s->getInputDataType(0)));
            if (!tok.empty()) {
                EXPECT_TRUE(data.count(tok))
                    << "input element type '" << tok << "' not listed in [facets].data";
            }
        }
    }
}
