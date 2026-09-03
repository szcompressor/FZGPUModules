# Adding a stage outside the main repo {#out_of_tree_stage}

<!-- doc-check: skip-file — placeholders (`MyStage`, `MY_STAGE`, id `40000`); the
     snippets show an out-of-tree build pattern, not compilable in this tree. -->

How to build, use, and file-serialize a new stage that lives in **your own**
project and links against an installed FZGPUModules — without patching the
library. Use this while a stage is unpublished, private, or still being
prototyped. When it is ready to upstream, follow
[How to Add a New Stage](\ref how_to_add_a_stage) instead and delete the
workarounds below.

The stage *class* is written exactly as the in-tree guide describes — same
`Stage` base class, same required overrides, same HIP rules, same
`execute()` contract. Everything on this page is only about the parts that
normally touch shared files (`fzm_format.h`, the root `CMakeLists.txt`,
`config.cpp`, `include/fzgpumodules.h`) and how to replace them from outside.

---

## 1. Depend on the installed library

```cmake
find_package(FZGPUModules REQUIRED)

add_library(my_stage STATIC my_stage/my_stage_stage.cu)
target_link_libraries(my_stage PUBLIC FZGMOD::fzgmod)
```

The install tree exports every header under `<prefix>/include/fzgmod/` —
including the internal ones this pattern needs (`stage/stage.h`,
`stage/stage_registry.h`, `fzm_format.h`). Point `CMAKE_PREFIX_PATH` at the
install prefix (or the build tree).

Your `.cu` includes the public umbrella header for the pipeline API and the
internal headers for the base class:

```cpp
#include <fzgpumodules.h>          // Pipeline, MemoryStrategy, CudaStream
#include "stage/stage.h"           // fz::Stage base class
#include "stage/stage_registry.h"  // FZ_REGISTER_* (only if you need file I/O)
#include "backend/types.h"         // NOT <cuda_runtime.h> — see the in-tree guide
```

---

## 2. Pick a stage-type id without editing the enum

`Stage::getStageTypeId()` returns a plain `uint16_t`, **not** `fz::StageType`,
so an out-of-tree stage never has to add an enumerator:

```cpp
class MyStage : public fz::Stage {
    // ...
    uint16_t getStageTypeId() const override { return kMyStageType; }
    std::string getName() const override { return "MyStage"; }
};

// Private id. Keep it well clear of the shipped range (currently < 64) and of
// anything you might reasonably merge from upstream. A 5-digit value is a safe
// scratch choice; it is NOT permanent.
static constexpr uint16_t kMyStageType = 40000;
```

> **This id is provisional.** It is written into every `.fzm` file your stage
> produces. Two different out-of-tree stages that both pick `40000` will write
> archives that silently collide. Before you publish anything other people will
> read, reserve a real `StageType` value upstream and migrate. If you must ship
> archives in the interim, treat the id as coupled to *your* build only.

`stageTypeToString()` lives in the library and will not know your name; that only
affects diagnostic strings, not correctness.

---

## 3. Use it in a pipeline (C++ only)

Nothing special — `addStage` is a template and only needs the full class
definition:

```cpp
fz::Pipeline p(in_bytes, fz::MemoryStrategy::PREALLOCATE);
auto* s = p.addStage<MyStage>();
s->setSomeParam(42);
p.finalize();
```

In-memory compress/decompress with the same `fz::Pipeline` object works with no
further registration, because `decompress()` rebuilds the inverse DAG from the
live forward pipeline.

---

## 4. File round-trip: self-register a header factory

`writeToFile` / `decompressFromFile` rebuild the inverse pipeline from the
archive alone, so they look the stage up in the stage registry by id. Register a
reconstruction factory at file scope in your `.cu` — the macro is header-only and
works identically outside the tree:

```cpp
// Simple stage (no template dispatch): default-construct + deserializeHeader.
FZ_REGISTER_SIMPLE_STAGE(static_cast<fz::StageType>(kMyStageType), MyStage);

// Or, if reconstruction needs to choose a template instantiation from the
// config bytes:
static fz::Stage* MyStage_fromHeader(const uint8_t* c, size_t n) { /* ... */ }
FZ_REGISTER_STAGE_FACTORY(static_cast<fz::StageType>(kMyStageType),
                          MyStage_fromHeader);
```

Registration happens at static-init time, so **the translation unit must
actually be linked in.** With a static library, force it:

```cmake
target_link_libraries(my_app PRIVATE
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,my_stage>"
    FZGMOD::fzgmod)
```

(or `-Wl,--whole-archive` / `--no-whole-archive`, or `ALWAYS_LINK` /
`+load-all`). Without this the linker drops the object and the factory never
registers, so decode fails with "no factory registered for stage type 40000".

---

## 5. What you do *not* get out of tree

- **TOML config** (`type = "MyStage"` in a `.toml`). The toml++ loader is
  confined to the library's `config.cpp` and has no extension point. Configure
  the stage in C++.
- **`--stages MyStage`** on the bundled `fzgmod-cli`. Build your own driver.
- **Pipeline Specialization / fusion.** The specialization planner only knows
  stages compiled into the library. An out-of-tree stage always runs as its own
  kernel; that is fine for correctness. If fusion matters, that is a reason to
  upstream — see \ref pipeline_specialization_internals.
- **The stage catalog and `docs/stages/` page.** Keep your own docs until the
  stage lands.

---

## 6. When you upstream

Move the `.cu`/`.h` into `modules/<category>/<name>/`, run
`scripts/new_stage.sh <Name> <category>` to claim a real `StageType` id and wire
CMake/tests, then work through the [How to Add a New
Stage](\ref how_to_add_a_stage) checklist. The class body and kernels carry over
unchanged; only the id and the shared-file edits differ. Note the id change in
your migration notes so pre-upstream archives can be regenerated.
