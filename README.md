# Par

`par` is a small wrapper around common parallelization frameworks.
It is designed to be extendable to other parallelization frameworks as well,
and provides a simple sequential backend that executes every construct serially.

## Installing

Since this project is a header-only library, it can be added to your project simply by
writing the following CMake code to your CMakeLists.txt:

    add_subdirectory(/path/to/par)

This line will provide the target `par::par`, which you can link against.

Alternatively, if you have several projects that depend on par, you can use a call to
`find_package(par)`:

    find_package(par REQUIRED)

This will require to set the CMake variable `par_DIR`. For that, you can alternatively
use the build directory, or install the library somewhere, say `/path/to/install`, and
set `par_DIR` to the path `/path/to/install/lib/cmake`.
