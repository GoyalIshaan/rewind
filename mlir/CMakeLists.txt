cmake_minimum_required(VERSION 3.20)
project(RewindMLIR LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIR from ${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/Rewind/IR)

# TableGen generates declarations/definitions; generated files stay in build/.
set(LLVM_TARGET_DEFINITIONS include/Rewind/IR/RewindDialect.td)
mlir_tablegen(include/Rewind/IR/RewindDialect.h.inc -gen-dialect-decls -dialect=rewind)
mlir_tablegen(include/Rewind/IR/RewindDialect.cpp.inc -gen-dialect-defs -dialect=rewind)

set(LLVM_TARGET_DEFINITIONS include/Rewind/IR/RewindTypes.td)
mlir_tablegen(include/Rewind/IR/RewindTypes.h.inc -gen-typedef-decls -typedefs-dialect=rewind)
mlir_tablegen(include/Rewind/IR/RewindTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=rewind)

set(LLVM_TARGET_DEFINITIONS include/Rewind/IR/RewindOps.td)
mlir_tablegen(include/Rewind/IR/RewindOps.h.inc -gen-op-decls)
mlir_tablegen(include/Rewind/IR/RewindOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(RewindIncGen)

add_library(RewindIR lib/RewindDialect.cpp)
add_dependencies(RewindIR RewindIncGen)
target_link_libraries(RewindIR PUBLIC MLIRIR MLIRSideEffectInterfaces)

add_executable(rewind-opt tools/rewind-opt.cpp)
target_link_libraries(rewind-opt PRIVATE RewindIR MLIROptLib
  MLIRArithDialect MLIRFuncDialect MLIRTransforms)

enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME rewind-dialect
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_dialect.py
          $<TARGET_FILE:rewind-opt>)
