# Stable main and development {#branch_policy}

`main` contains the stable CUDA library, its CLI and public stage catalog,
validated specialization, installation support, Python/libpressio integration
capability, completed extension contracts, and regression/developer checks.
`ENABLE_NVTX` adds production profiling annotations without research programs.

`development` preserves experimental modules (CDF97, SPECK2D, wavelet correction,
SZx, and reference SZp), AMD backend work, research profiling programs, paper
artifacts, design notes, capability cards, and demonstration material. The
pre-cleanup development snapshot retains these files and their historical tests.
The stable-main cleanup removes them only from the stable branch.

Experimental archive stage IDs 36, 37, 39–41, and 43 remain reserved forever.
Stable main does not construct or decode those stages. Use a compatible
`development` decoder for existing experimental archives; stable main rejects
unavailable stages rather than interpreting their payloads as another codec.
Stable archive compatibility applies only to the shipped stage catalog and its
supported configurations.

Promote bounded changes from development only after their input/numeric contracts,
archive behavior, installation, and appropriate tests are reviewed. Keep fixes
for production behavior on main and cherry-pick those commits into development.
Do not merge the stable-main cleanup wholesale into development: that would
remove the experimental and research tree that development is meant to retain.

`Force` remains an API/parser compatibility option. All currently shipped
specialization implementations are already admitted for `Auto`; keeping `Force`
does not ship additional experimental strategies. Normal adaptive dispatch and
validated generated kernels remain part of stable specialization.
