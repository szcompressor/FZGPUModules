/**
 * @file test_module_cards.cpp
 * @brief Consistency checks for docs/cards/ FAIR module card files.
 *
 * Verifies that index.json is in sync with the card files on disk and that each
 * card contains the required top-level fields.  Heavy JSON-schema validation
 * is done by validate-cards.py (Z-Hub repo); this C++ test catches missing /
 * orphaned files and obvious format regressions without an external dependency.
 *
 * CMake variable required:  FZ_CARDS_DIR  (absolute path to docs/cards/).
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifndef FZ_CARDS_DIR
#error "FZ_CARDS_DIR must be defined by CMake (absolute path to docs/cards/)."
#endif

namespace fs = std::filesystem;

namespace {

// Files in docs/cards/ that are not card files and should be skipped.
const std::set<std::string> kNonCardFiles = { "index.json", "schema.json" };

// Extract all "*.json" filename strings from index.json (simple regex scan;
// no full JSON parser needed for this flat array format).
std::vector<std::string> parseIndexJson(const fs::path& index_path) {
    std::ifstream f(index_path);
    if (!f) return {};
    const std::string text((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
    std::vector<std::string> out;
    const std::regex re(R"re("([^"]+\.json)")re");
    for (auto it = std::sregex_iterator(text.begin(), text.end(), re);
         it != std::sregex_iterator(); ++it) {
        out.push_back((*it)[1].str());
    }
    return out;
}

} // namespace

// index.json lists exactly the card files on disk — no orphans, no missing.
TEST(ModuleCards, IndexConsistentWithDisk) {
    const fs::path cards_dir = FZ_CARDS_DIR;
    ASSERT_TRUE(fs::is_directory(cards_dir))
        << "FZ_CARDS_DIR is not a directory: " << cards_dir;

    const fs::path index = cards_dir / "index.json";
    ASSERT_TRUE(fs::exists(index)) << "index.json missing from " << cards_dir;

    const auto listed = parseIndexJson(index);
    EXPECT_GT(listed.size(), 0u) << "index.json parsed empty";

    // Every filename listed in index.json must exist on disk.
    for (const auto& name : listed)
        EXPECT_TRUE(fs::exists(cards_dir / name))
            << "index.json lists '" << name << "' but file is missing";

    // Every *.json card on disk must be listed in index.json.
    const std::set<std::string> indexed(listed.begin(), listed.end());
    for (const auto& entry : fs::directory_iterator(cards_dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.size() < 5 || name.substr(name.size() - 5) != ".json") continue;
        if (kNonCardFiles.count(name)) continue;
        EXPECT_TRUE(indexed.count(name))
            << "'" << name << "' exists on disk but is not listed in index.json";
    }
}

// Every card has the required top-level field keys (lightweight text scan).
// Full schema validation (types, enum values, conditional requirements) is done
// by validate-cards.py; this check catches obviously malformed cards quickly.
TEST(ModuleCards, EachCardHasRequiredFields) {
    const fs::path cards_dir = FZ_CARDS_DIR;
    ASSERT_TRUE(fs::is_directory(cards_dir));

    const auto listed = parseIndexJson(cards_dir / "index.json");
    ASSERT_GT(listed.size(), 0u);

    static const std::vector<std::string> kRequired = {
        "\"schemaVersion\"", "\"id\"", "\"stage\"", "\"name\"",
        "\"zhubModule\"",    "\"category\"", "\"algorithm\"",
        "\"description\"",  "\"lossy\"", "\"hardware\"", "\"provenance\""
    };

    for (const auto& name : listed) {
        const fs::path p = cards_dir / name;
        if (!fs::exists(p)) continue;  // already reported by IndexConsistentWithDisk
        std::ifstream f(p);
        const std::string text((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
        for (const auto& key : kRequired)
            EXPECT_NE(text.find(key), std::string::npos)
                << name << " is missing required field " << key;
    }
}
