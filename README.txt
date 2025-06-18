Place to create modules for the FZ project.

cmake -S . -B build

Options:

-DBUILD_CUDASTF=ON # To build the CUDASTF implementation

-DCMAKE_BUILD_TYPE=Release # For release build
-DCMAKE_BUILD_TYPE=Debug   # For debug build
-DCMAKE_CUDA_ARCHITECTURE=<arch> # For CUDA architecture, e.g. "86" for 3080Ti

example:

cmake -S . -B stf_build/ -DBUILD_CUDASTF=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES="86"
cd stf_build/
make -j
make install


cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES="86"