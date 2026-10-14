#!/usr/bin/env python3
"""Check that shipped stages are wired into every required integration surface.

This deliberately validates the existing hand-written C++ rather than generating it.
Factory construction and TOML load/save logic are type-specific and should remain easy
to review. Run from any directory; no GPU, compiler, or configured build is required.
"""

import pathlib
import re
import shlex
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
FORMAT = ROOT / "include" / "fzm_format.h"
MODULES = ROOT / "modules"
CONFIG = ROOT / "src" / "pipeline" / "config.cpp"
UMBRELLA = ROOT / "include" / "fzgpumodules.h"
CMAKE = MODULES / "CMakeLists.txt"

# Historical/reserved IDs that intentionally have no shipped implementation.
RESERVED = {"UNKNOWN", "SCALE", "PASSTHROUGH", "SPLIT"}

# Development stage IDs stay reserved without implementations or archive factories.
DEVELOPMENT = {"SZX", "SZP", "CDF97", "SPECK2D", "CDF97_OUTLIER_CORRECT",
               "FUSED_QUANT_ADAPTIVE_LORENZO"}


def require_match(pattern, text, label, flags=0):
    match = re.search(pattern, text, flags)
    if not match:
        raise RuntimeError(f"could not parse {label}")
    return match


def duplicates(items):
    seen = set()
    return sorted({item for item in items if item in seen or seen.add(item)})


def cmake_commands(text, command):
    # Module declarations use literal paths; do not interpret arbitrary CMake.
    text = re.sub(r"#[^\n]*", "", text)
    return re.findall(rf"\b{command}\s*\(([^()]*)\)", text, re.I | re.S)


def module_build_errors(cmake, registry_dirs):
    errors = []
    catalogs = {}
    sets = [shlex.split(body) for body in cmake_commands(cmake, "set")]
    for name in ("FZGMOD_STAGE_MODULES", "FZGMOD_INTERNAL_MODULES", "FZGMOD_LEGACY_MODULES"):
        matches = [tokens for tokens in sets if tokens and tokens[0] == name]
        if len(matches) != 1:
            errors.append(f"modules/CMakeLists.txt must declare {name} exactly once")
        catalogs[name] = matches[0][1:] if matches else []

    entries = [entry for values in catalogs.values() for entry in values]
    for entry in duplicates(entries):
        errors.append(f'duplicate module catalog entry "{entry}"')
    registered = {path.removeprefix("modules/") for path in registry_dirs}
    stages = set(catalogs["FZGMOD_STAGE_MODULES"])
    for entry in sorted(registered - stages):
        errors.append(f'registered stage "modules/{entry}" is missing from FZGMOD_STAGE_MODULES')
    for entry in sorted(stages - registered):
        errors.append(f'public module "modules/{entry}" has no kStageRegistry entry')

    all_sources = []
    for entry in sorted(set(entries)):
        relative = pathlib.PurePosixPath(entry)
        if (relative.is_absolute() or ".." in relative.parts or
                str(relative) != entry or not re.fullmatch(r"[A-Za-z0-9_/-]+", entry)):
            errors.append(f'malformed module catalog path "{entry}"')
            continue
        local_cmake = MODULES / entry / "CMakeLists.txt"
        if not local_cmake.is_file():
            errors.append(f'module "{entry}" has no local CMakeLists.txt')
            continue
        declarations = cmake_commands(local_cmake.read_text(encoding="utf-8"), "target_sources")
        sources = []
        target = "fzgmod_utils" if entry == "util/histogram" else "fzgmod_modules"
        for body in declarations:
            tokens = shlex.split(body)
            if tokens[:2] != [target, "PRIVATE"]:
                errors.append(f'{entry}: target_sources must declare {target} PRIVATE')
                continue
            sources.extend(tokens[2:])
        if not sources:
            errors.append(f'{entry}: no module-local source declarations')
        for source in sources:
            prefix = "${CMAKE_CURRENT_LIST_DIR}/"
            path = source.removeprefix(prefix)
            relative = pathlib.PurePosixPath(path)
            if (not source.startswith(prefix) or relative.is_absolute() or
                    ".." in relative.parts or str(relative) != path or
                    not re.fullmatch(r"[A-Za-z0-9_./-]+", path) or
                    relative.suffix not in {".cu", ".cpp", ".c", ".cc", ".cxx"}):
                errors.append(f'{entry}: malformed source declaration "{source}"')
                continue
            source_path = local_cmake.parent / path
            if not source_path.is_file():
                errors.append(f'{entry}: declared source "{path}" does not exist')
            all_sources.append(str(source_path.relative_to(MODULES)))
    registrar = re.compile(r"FZ_REGISTER_(?:SIMPLE_STAGE|STAGE_FACTORY)\s*\(")
    for source_path in MODULES.rglob("*.cu"):
        if registrar.search(source_path.read_text(encoding="utf-8")):
            source = str(source_path.relative_to(MODULES))
            if source not in all_sources:
                errors.append(f'factory source "{source}" is not declared by a module')
    for source in duplicates(all_sources):
        errors.append(f'duplicate module source declaration "{source}"')

    # A new local build file must be explicitly classified, including internal
    # infrastructure and archive-compatibility implementations.
    for local_cmake in MODULES.rglob("CMakeLists.txt"):
        entry = str(local_cmake.parent.relative_to(MODULES))
        if entry != "." and entry not in entries:
            errors.append(f'module "{entry}" has a local CMakeLists.txt but no catalog entry')
    return errors


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
    factory_cases = set()
    for cu_path in MODULES.rglob("*.cu"):
        factory_cases.update(registrar_pattern.findall(cu_path.read_text(encoding="utf-8")))

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
    implemented_enums = set(enum_names) - RESERVED - DEVELOPMENT
    registered_enums = set(registry_enums)

    errors = module_build_errors(cmake, registry_dirs)
    for name in sorted(DEVELOPMENT):
        if name in factory_cases or name in registered_enums:
            errors.append(f"development StageType::{name} must not ship on main")
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
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
