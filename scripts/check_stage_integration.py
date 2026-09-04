#!/usr/bin/env python3
"""Check that shipped stages are wired into every required integration surface.

This deliberately validates the existing hand-written C++ rather than generating it.
Factory construction and TOML load/save logic are type-specific and should remain easy
to review. Run from any directory; no GPU, compiler, or configured build is required.
"""

import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
FORMAT = ROOT / "include" / "fzm_format.h"
MODULES = ROOT / "modules"
CONFIG = ROOT / "src" / "pipeline" / "config.cpp"
UMBRELLA = ROOT / "include" / "fzgpumodules.h"
CMAKE = ROOT / "CMakeLists.txt"

# Historical/reserved IDs that intentionally have no shipped implementation.
RESERVED = {"UNKNOWN", "SCALE", "PASSTHROUGH", "SPLIT"}

# Quarantined experimental/reference compressors: their StageType ID and FZM
# header factory stay compiled (so pre-existing archives decode), but they are
# deliberately NOT public modules — absent from kStageRegistry, the umbrella
# header, and the module catalogs. Their factory lives under
# modules/experimental/, not the ordinary modules/<category>/ tree. Keep the ID
# here forever once quarantined; never reuse it.
EXPERIMENTAL = {"SZP"}
EXPERIMENTAL_DIR = ROOT / "modules" / "experimental"


def require_match(pattern, text, label, flags=0):
    match = re.search(pattern, text, flags)
    if not match:
        raise RuntimeError(f"could not parse {label}")
    return match


def duplicates(items):
    seen = set()
    return sorted({item for item in items if item in seen or seen.add(item)})


def main():
    fmt = FORMAT.read_text(encoding="utf-8")
    config = CONFIG.read_text(encoding="utf-8")
    umbrella = UMBRELLA.read_text(encoding="utf-8")
    cmake = CMAKE.read_text(encoding="utf-8")

    # Stages self-register their FZM-header factory from their own .cu via
    # FZ_REGISTER_SIMPLE_STAGE(StageType::X, Class) or
    # FZ_REGISTER_STAGE_FACTORY(StageType::X, fn) — see include/stage/stage_registry.h.
    # There is no central factory switch to parse any more, so scan modules/ instead.
    registrar_pattern = re.compile(
        r"FZ_REGISTER_(?:SIMPLE_STAGE|STAGE_FACTORY)\s*\(\s*(?:fz::)?StageType::([A-Z0-9_]+)"
    )
    # modules/experimental/ is quarantined stages, not public modules — scanned
    # separately below so its factories never count toward the public
    # implemented_enums comparison.
    factory_cases = set()
    for cu_path in MODULES.rglob("*.cu"):
        if EXPERIMENTAL_DIR in cu_path.parents:
            continue
        factory_cases.update(registrar_pattern.findall(cu_path.read_text(encoding="utf-8")))

    experimental_factories = set()
    if EXPERIMENTAL_DIR.is_dir():
        for cu_path in EXPERIMENTAL_DIR.rglob("*.cu"):
            experimental_factories.update(
                registrar_pattern.findall(cu_path.read_text(encoding="utf-8"))
            )

    enum_body = require_match(
        r"enum\s+class\s+StageType[^\{]*\{(.*?)\};", fmt, "StageType", re.S
    ).group(1)
    enum_rows = re.findall(r"\b([A-Z][A-Z0-9_]*)\s*=\s*([0-9]+)", enum_body)
    enum_names = [name for name, _ in enum_rows]
    enum_ids = [int(value) for _, value in enum_rows]

    registry_body = require_match(
        r"kStageRegistry\[\]\s*=\s*\{(.*?)\n\};",
        config,
        "kStageRegistry",
        re.S,
    ).group(1)
    registry_rows = re.findall(
        r'\{\s*"([^"]+)"\s*,\s*StageType::([A-Z0-9_]+)\s*,\s*'
        r'(\w+)\s*,\s*(\w+)\s*,\s*"([^"]+)"\s*\}',
        registry_body,
    )
    registry_names = [row[0] for row in registry_rows]
    registry_enums = [row[1] for row in registry_rows]
    registry_dirs = [row[4] for row in registry_rows]

    string_cases = set(re.findall(r"case\s+StageType::([A-Z0-9_]+)", fmt))
    implemented_enums = set(enum_names) - RESERVED - EXPERIMENTAL
    registered_enums = set(registry_enums)

    errors = []
    for name in sorted(EXPERIMENTAL):
        if name not in factory_cases and name not in experimental_factories:
            errors.append(
                f"quarantined StageType::{name} has no FZ_REGISTER factory under "
                f"modules/experimental/ — pre-existing archives would fail to decode"
            )
        if name in registered_enums:
            errors.append(f"quarantined StageType::{name} must not be in the public kStageRegistry")
    for value in duplicates(enum_names):
        errors.append(f"duplicate StageType token {value}")
    for value in duplicates(enum_ids):
        owners = [name for name, ident in enum_rows if int(ident) == value]
        errors.append(f"duplicate StageType numeric ID {value}: {', '.join(owners)}")
    for value in duplicates(registry_names):
        errors.append(f'duplicate TOML stage name "{value}" in kStageRegistry')
    for value in duplicates(registry_enums):
        errors.append(f"duplicate StageType::{value} in kStageRegistry")
    for value in duplicates(registry_dirs):
        errors.append(f'duplicate source directory "{value}" in kStageRegistry')

    for name in sorted(implemented_enums - factory_cases):
        errors.append(f"StageType::{name} has no FZ_REGISTER_SIMPLE_STAGE/FZ_REGISTER_STAGE_FACTORY registration")
    for name in sorted(factory_cases - implemented_enums):
        errors.append(f"a stage .cu registers unknown/reserved StageType::{name}")
    for name in sorted(implemented_enums - registered_enums):
        errors.append(f"StageType::{name} is missing from kStageRegistry")
    for name in sorted(registered_enums - implemented_enums):
        errors.append(f"kStageRegistry contains unknown/reserved StageType::{name}")
    for name in sorted(implemented_enums - string_cases):
        errors.append(f"StageType::{name} has no stageTypeToString() case")

    for public_name, enum_name, load_fn, save_fn, source_dir in registry_rows:
        source_path = ROOT / source_dir
        if not source_path.is_dir():
            errors.append(f'{public_name}: source directory "{source_dir}" does not exist')
            continue
        if source_dir not in cmake:
            errors.append(f'{public_name}: source directory "{source_dir}" is absent from CMakeLists.txt')

        include_prefix = source_dir.removeprefix("modules/") + "/"
        if include_prefix not in umbrella:
            errors.append(
                f'{public_name}: fzgpumodules.h has no include below "{include_prefix}"'
            )

        if not re.search(rf"\b{re.escape(load_fn)}\s*\(", config):
            errors.append(f"{public_name}: loader {load_fn} is not defined")
        if not re.search(rf"\b{re.escape(save_fn)}\s*\(", config):
            errors.append(f"{public_name}: saver {save_fn} is not defined")
        if enum_name not in factory_cases:
            errors.append(f"{public_name}: StageType::{enum_name} has no self-registered factory")

    for error in errors:
        print(f"FAIL {error}")
    if errors:
        print(f"\n{len(errors)} stage integration problem(s)")
        return 1

    print(
        f"OK: {len(registry_rows)} shipped stages consistent across StageType, "
        "factory, TOML registry, CMake, and umbrella header"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
