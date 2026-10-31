"""Check lowering structure, rejection boundaries, and actual CPU execution.

Only the standard library is required. Execution uses the installed MLIR tools;
the reference rounds each multiply and add separately to the input element type.
"""

import math
from pathlib import Path
import random
import signal
import struct
import subprocess
import sys
import unittest


REWIND_OPT = str(Path(sys.argv.pop(1)).resolve())
MLIR_OPT = str(Path(sys.argv.pop(1)).resolve()) if len(sys.argv) > 1 else None
MLIR_RUNNER = str(Path(sys.argv.pop(1)).resolve()) if len(sys.argv) > 1 else None
C_RUNNER_UTILS = str(Path(sys.argv.pop(1)).resolve()) if len(sys.argv) > 1 else None

CPU_PIPELINE = [
    "--one-shot-bufferize=bufferize-function-boundaries",
    "--buffer-deallocation-pipeline",
    "--convert-bufferization-to-memref",
    "--convert-linalg-to-loops",
    "--convert-scf-to-cf",
    "--convert-cf-to-llvm",
    "--convert-arith-to-llvm",
    "--finalize-memref-to-llvm",
    "--convert-func-to-llvm",
    "--reconcile-unrealized-casts",
]


def types(element="f64", shape=(2, 3)):
    d, k = shape
    return (f"tensor<{d}x{k}x{element}>", f"tensor<{element}>",
            f"tensor<{d}x{element}>", f"tensor<{k}x{element}>")


def history_source(n=2, when=None, element="f64", shape=(2, 3)):
    """Expose runtime inputs and retrieve after the append counts in `when`."""
    when = [n] if when is None else when
    matrix, scalar, vector, key = types(element, shape)
    store = f"!rewind.store<{matrix}>"
    args = [f"%initial: {matrix}"]
    for i in range(n):
        args += [f"%a{i}: {scalar}", f"%v{i}: {vector}", f"%k{i}: {key}"]
    args += [f"%accepted{i}: index" for i in range(len(when))]
    outputs = [matrix] * len(when)
    result_types = outputs[0] if len(outputs) == 1 else f"({', '.join(outputs)})"
    result_signature = f" -> {result_types}" if outputs else ""
    lines = [f"func.func @history({', '.join(args)}){result_signature} {{",
             f"  %h = rewind.initialize_store %initial : {matrix} -> {store}"]
    for boundary in range(n + 1):
        for i, available in enumerate(when):
            if available == boundary:
                lines.append(f"  %selected{i} = rewind.retrieve %h[%accepted{i}]"
                             f" : {store} -> {matrix}")
        if boundary < n:
            i = boundary
            lines.append(f"  rewind.append %h, %a{i}, %v{i}, %k{i} : "
                         f"{store}, {scalar}, {vector}, {key}")
    returned = ", ".join(f"%selected{i}" for i in range(len(when)))
    lines += [f"  return {returned} : {', '.join(outputs)}" if outputs
              else "  return", "}"]
    return "\n".join(lines) + "\n"


def rounded(value, element):
    if element == "f64":
        return float(value)
    try:
        return struct.unpack(">f", struct.pack(">f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def bits(value, element):
    fmt, integer = (">f", ">I") if element == "f32" else (">d", ">Q")
    return struct.unpack(integer, struct.pack(fmt, rounded(value, element)))[0]


def literal(value, element):
    width = 8 if element == "f32" else 16
    return f"0x{bits(value, element):0{width}X}"


def dense(value, element):
    if isinstance(value, list):
        return "[" + ", ".join(dense(v, element) for v in value) + "]"
    return literal(value, element)


def replay(initial, updates, count, element):
    state = [[rounded(x, element) for x in row] for row in initial]
    for a, v, k in updates[:count]:
        a = rounded(a, element)
        v = [rounded(x, element) for x in v]
        k = [rounded(x, element) for x in k]
        state = [[rounded(rounded(a * s, element)
                          + rounded(v[d] * k[c], element), element)
                  for c, s in enumerate(row)] for d, row in enumerate(state)]
    return state


def execution_source(initial, updates, indices, when=None, element="f64",
                     source=None, expected=None):
    """Call a function with dynamic parameters; check every returned element."""
    shape = (len(initial), len(initial[0]))
    when = [len(updates)] * len(indices) if when is None else when
    source = source or history_source(len(updates), when, element, shape)
    matrix, scalar, vector, key = types(element, shape)
    args = ["%initial"]
    arg_types = [matrix]
    lines = [source, "func.func @main() -> i32 {",
             f"  %initial = arith.constant dense<{dense(initial, element)}> : {matrix}"]
    for i, (a, v, k) in enumerate(updates):
        for name, val, ty in ((f"a{i}", a, scalar), (f"v{i}", v, vector),
                              (f"k{i}", k, key)):
            lines.append(f"  %{name} = arith.constant dense<{dense(val, element)}> : {ty}")
            args.append(f"%{name}")
            arg_types.append(ty)
    for i, index in enumerate(indices):
        lines.append(f"  %index{i} = arith.constant {index} : index")
        args.append(f"%index{i}")
        arg_types.append("index")
    lhs = ", ".join(f"%result{i}" for i in range(len(indices)))
    result_types = ", ".join([matrix] * len(indices))
    lines.append(f"  {lhs} = func.call @history({', '.join(args)}) : "
                 f"({', '.join(arg_types)}) -> ({result_types})")
    for d in range(shape[0]):
        lines.append(f"  %d{d} = arith.constant {d} : index")
    for k in range(shape[1]):
        lines.append(f"  %kidx{k} = arith.constant {k} : index")
    for i, (available, index) in enumerate(zip(when, indices)):
        # Invalid indices must fail in @history. Do not let a later matrix
        # assertion accidentally stand in for a missing history bounds check.
        if not 0 <= index <= available:
            continue
        want = (expected[i] if expected is not None else
                replay(initial, updates, index, element))
        for d, row in enumerate(want):
            for k, expected_value in enumerate(row):
                name = f"{i}_{d}_{k}"
                lines.append(f"  %x{name} = tensor.extract %result{i}[%d{d}, %kidx{k}] : {matrix}")
                if math.isnan(expected_value):
                    lines.append(f"  %ok{name} = arith.cmpf uno, %x{name}, %x{name} : {element}")
                else:
                    integer = "i32" if element == "f32" else "i64"
                    # Comparing bits also checks the sign of zero.
                    lines += [f"  %bits{name} = arith.bitcast %x{name} : {element} to {integer}",
                              f"  %want{name} = arith.constant {bits(expected_value, element)} : {integer}",
                              f"  %ok{name} = arith.cmpi eq, %bits{name}, %want{name} : {integer}"]
                lines.append(f'  cf.assert %ok{name}, "matrix mismatch at {name}"')
    lines += ["  %zero = arith.constant 0 : i32", "  return %zero : i32", "}"]
    return "\n".join(lines) + "\n"


class LoweringTests(unittest.TestCase):
    def run_opt(self, source, *flags):
        return subprocess.run([REWIND_OPT, "--rewind-lower-history", "--verify-each", *flags],
                              input=source, text=True, capture_output=True, timeout=30)

    def lower(self, source, *flags):
        result = self.run_opt(source, *flags)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def reject(self, source, diagnostic="history", *flags):
        result = self.run_opt(source, *flags)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(diagnostic, result.stderr)

    def test_all_three_operations_are_removed(self):
        lowered = self.lower(history_source())
        self.assertNotIn("!rewind.store", lowered)
        self.assertNotRegex(lowered, r"rewind\.(initialize_store|append|retrieve)\s+%")
        self.assertIn("tensor.generate", lowered)
        self.assertIn("cf.assert", lowered)

    def test_existing_float_types_and_rectangular_shapes(self):
        for element in ("f16", "bf16", "f32", "f64"):
            with self.subTest(element=element):
                self.lower(history_source(element=element))

    def test_separate_arithmetic_without_fast_math(self):
        lowered = self.lower(history_source(n=1))
        self.assertEqual(lowered.count("arith.mulf"), 2)
        self.assertEqual(lowered.count("arith.addf"), 1)
        self.assertNotIn("fastmath", lowered)

    def test_empty_history_still_checks_bounds(self):
        lowered = self.lower(history_source(n=0))
        self.assertIn("cf.assert", lowered)
        self.assertNotIn("tensor.generate", lowered)
        self.assertNotIn("scf.if", lowered)

    def test_no_retrieval_needs_no_runtime_history(self):
        lowered = self.lower(history_source(when=[]))
        self.assertNotIn("!rewind.store", lowered)
        self.assertNotIn("rewind.append", lowered)
        self.assertNotIn("tensor.generate", lowered)
        self.assertNotIn("cf.assert", lowered)

    def test_repeated_retrieval_and_only_preceding_appends(self):
        lowered = self.lower(history_source(n=2, when=[0, 1, 2, 2]))
        self.assertEqual(lowered.count("cf.assert"), 4)
        self.assertEqual(lowered.count("scf.if"), 0 + 1 + 2 + 2)

    def test_unused_retrieval_remains_fallible_after_cleanup(self):
        source = history_source(n=1).replace("return %selected0", "return %initial")
        lowered = self.lower(source, "--canonicalize", "--cse")
        self.assertIn('"rewind.retrieve index out of range"', lowered)

    def test_opaque_matrix_consumer_is_preserved(self):
        matrix = types()[0]
        source = f"func.func private @consume({matrix})\n" + history_source(n=1)
        source = source.replace("  return %selected0", f"  func.call @consume(%selected0) : ({matrix}) -> ()\n  return %selected0")
        lowered = self.lower(source)
        self.assertIn("call @consume", lowered)
        self.assertLess(lowered.index("cf.assert"), lowered.index("call @consume"))

    def test_opaque_handle_consumer_rejected(self):
        source = history_source().replace("  return %selected0", '  "opaque.consume"(%h) : (!rewind.store<tensor<2x3xf64>>) -> ()\n  return %selected0')
        self.reject(source, "history", "--allow-unregistered-dialect")

    def test_handle_arguments_and_external_declarations_rejected(self):
        self.reject("func.func private @escape(!rewind.store<tensor<2x3xf64>>)\n")
        self.reject("""func.func @input(%h: !rewind.store<tensor<2x3xf64>>,
                         %i: index) -> tensor<2x3xf64> {
          %s = rewind.retrieve %h[%i] : !rewind.store<tensor<2x3xf64>> -> tensor<2x3xf64>
          return %s : tensor<2x3xf64>
        }""")

    def test_returning_handle_rejected(self):
        source = """func.func @escape(%s: tensor<2x3xf64>) -> !rewind.store<tensor<2x3xf64>> {
          %h = rewind.initialize_store %s : tensor<2x3xf64> -> !rewind.store<tensor<2x3xf64>>
          return %h : !rewind.store<tensor<2x3xf64>>
        }"""
        self.reject(source)

    def test_branch_spanning_history_rejected(self):
        source = history_source(n=1).replace("  rewind.append", "  cf.br ^next\n^next:\n  rewind.append")
        self.reject(source, "single-block")

    def test_nested_history_use_rejected(self):
        source = history_source(n=1)
        append = next(line for line in source.splitlines() if "rewind.append" in line)
        source = source.replace(append, f"  %true = arith.constant true\n  scf.if %true {{\n{append}\n  }}")
        self.reject(source)

    def test_nested_initialization_rejected(self):
        source = history_source(n=1)
        body_start = source.index("  %h =")
        source = source[:body_start] + "  %true = arith.constant true\n  scf.if %true {\n" + source[body_start:]
        source = source.replace("  return %selected0", "  }\n  return %initial")
        self.reject(source, "single-block")

    def test_failure_does_not_partially_rewrite_other_functions(self):
        source = history_source(n=1) + "func.func private @escape(!rewind.store<tensor<2x3xf64>>)\n"
        result = self.run_opt(source, "--mlir-print-ir-after-failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rewind.initialize_store", result.stderr)
        self.assertNotIn("cf.assert", result.stderr)

    def test_pass_is_idempotent(self):
        once = self.lower(history_source())
        self.assertEqual(self.lower(once), once)


@unittest.skipUnless(MLIR_OPT and MLIR_RUNNER and C_RUNNER_UTILS,
                     "MLIR tools and the C runner runtime are required")
class ExecutionTests(unittest.TestCase):
    def execute(self, source, *, invalid=False, optimize=False):
        flags = ["--canonicalize", "--cse"] if optimize else []
        lowered = subprocess.run([REWIND_OPT, "--rewind-lower-history", "--verify-each", *flags],
                                 input=source, text=True, capture_output=True, timeout=30)
        self.assertEqual(lowered.returncode, 0, lowered.stderr)
        compiled = subprocess.run([MLIR_OPT, *CPU_PIPELINE], input=lowered.stdout,
                                  text=True, capture_output=True, timeout=30)
        self.assertEqual(compiled.returncode, 0, compiled.stderr)
        self.assertNotRegex(compiled.stdout,
                            r"\b(?:bufferization|linalg|scf|tensor|memref|arith|cf)\.")
        result = subprocess.run([MLIR_RUNNER, "-e", "main", "--entry-point-result=i32",
                                 f"--shared-libs={C_RUNNER_UTILS}",
                                 "--O2" if optimize else "--O0"],
                                input=compiled.stdout, text=True, capture_output=True, timeout=30)
        if invalid:
            self.assertEqual(result.returncode, -signal.SIGABRT, result.stderr)
        else:
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "0")

    def test_every_boundary_including_empty_and_full(self):
        rng = random.Random(426)
        for element in ("f32", "f64"):
            for n in (0, 1, 3):
                initial = [[rng.uniform(-5, 5) for _ in range(3)] for _ in range(2)]
                updates = [(rng.uniform(-2, 2), [rng.uniform(-3, 3) for _ in range(2)],
                            [rng.uniform(-3, 3) for _ in range(3)]) for _ in range(n)]
                with self.subTest(element=element, n=n):
                    self.execute(execution_source(initial, updates, list(range(n + 1)), element=element))

    def test_retrieval_before_and_after_appends_and_repeated_indices(self):
        initial = [[1., 2., 3.], [4., 5., 6.]]
        updates = [(2., [1., 3.], [2., -1., 4.]),
                   (-0.5, [5., 2.], [-3., 1., 2.])]
        self.execute(execution_source(initial, updates, [0, 1, 0, 2, 1, 2],
                                      when=[0, 1, 2, 2, 2, 2]))

    def test_invalid_indices_fail_at_runtime(self):
        for n, index in ((0, -1), (0, 1), (1, -1), (1, 2), (2, 3)):
            with self.subTest(n=n, index=index):
                updates = [(1., [1., 2.], [3., 4., 5.])] * n
                self.execute(execution_source([[0., 0., 0.], [0., 0., 0.]],
                                              updates, [index]), invalid=True)

    def test_future_append_does_not_make_an_early_index_valid(self):
        updates = [(1., [1., 2.], [3., 4., 5.])] * 2
        self.execute(execution_source([[0., 0., 0.], [0., 0., 0.]],
                                      updates, [2], when=[1]), invalid=True)

    def test_unused_invalid_retrieval_still_fails_before_consumer(self):
        source = execution_source([[0., 0., 0.], [0., 0., 0.]],
                                  [(1., [1., 2.], [3., 4., 5.])], [2])
        source = "func.func private @exit(i32)\n" + source
        source = source.replace("  return %selected0", "  %code = arith.constant 99 : i32\n  func.call @exit(%code) : (i32) -> ()\n  return %initial")
        self.execute(source, invalid=True, optimize=True)

    def test_strict_rounding_survives_optimization(self):
        for element, epsilon in (("f32", 2.**-23), ("f64", 2.**-52)):
            for term in ("scaled", "outer"):
                with self.subTest(element=element, term=term):
                    # Each rounded product is exactly 1; a contracted FMA
                    # would instead leave the nonzero residual -epsilon**2.
                    if term == "scaled":
                        initial = [[1. - epsilon]]
                        update = (1. + epsilon, [-1.], [1.])
                    else:
                        initial = [[-1.]]
                        update = (1., [1. + epsilon], [1. - epsilon])
                    self.execute(execution_source(initial, [update], [1], element=element), optimize=True)

    def test_signed_zero_infinity_and_nan(self):
        initial = [[-0., 0., math.inf], [-math.inf, math.nan, 1.]]
        updates = [(1., [0., 0.], [1., -1., 1.])]
        for element in ("f32", "f64"):
            with self.subTest(element=element):
                self.execute(execution_source(initial, updates, [0, 1], element=element), optimize=True)

    def test_independent_stores_with_same_checkpoint(self):
        matrix, scalar, vector, key = types()
        store = f"!rewind.store<{matrix}>"
        source = history_source(n=2, when=[1, 2])
        source = source.replace("  %h =", f"  %other = rewind.initialize_store %initial : {matrix} -> {store}\n  %h =")
        source = source.replace("  %selected1 =", f"  rewind.append %other, %a1, %v1, %k1 : {store}, {scalar}, {vector}, {key}\n  %selected1 =")
        source = source.replace("%selected1 = rewind.retrieve %h", "%selected1 = rewind.retrieve %other")
        initial = [[1., 2., 3.], [4., 5., 6.]]
        updates = [(2., [1., 2.], [3., 4., 5.]), (-1., [4., 2.], [0., 1., -2.])]
        expected = [replay(initial, [update], 1, "f64") for update in updates]
        self.execute(execution_source(initial, updates, [1, 1], when=[1, 2],
                                      source=source, expected=expected))

    def test_retrieved_result_can_initialize_another_store(self):
        matrix, scalar, vector, key = types()
        store = f"!rewind.store<{matrix}>"
        source = history_source(n=2, when=[1, 2])
        source = source.replace("  rewind.append %h, %a1", f"  %other = rewind.initialize_store %selected0 : {matrix} -> {store}\n  rewind.append %other, %a1")
        source = source.replace("%selected1 = rewind.retrieve %h", "%selected1 = rewind.retrieve %other")
        initial = [[1., 2., 3.], [4., 5., 6.]]
        updates = [(2., [1., 2.], [3., 4., 5.]), (-1., [4., 2.], [0., 1., -2.])]
        expected = [replay(initial, updates, n, "f64") for n in (1, 2)]
        self.execute(execution_source(initial, updates, [1, 1], when=[1, 2],
                                      source=source, expected=expected))

    def test_checkpoint_value_survives_a_new_tensor_version(self):
        source = history_source(n=1, when=[0, 1])
        source = source.replace("  %selected1 =", """  %zero = arith.constant 0 : index
  %new = arith.constant 99. : f64
  %changed = tensor.insert %new into %initial[%zero, %zero] : tensor<2x3xf64>
  %selected1 =""")
        source = source.replace("return %selected0, %selected1",
                                "return %changed, %selected1")
        initial = [[1., 2., 3.], [4., 5., 6.]]
        updates = [(2., [1., 2.], [3., 4., 5.])]
        expected = [[[99., 2., 3.], [4., 5., 6.]],
                    replay(initial, updates, 1, "f64")]
        self.execute(execution_source(initial, updates, [0, 1], when=[0, 1],
                                      source=source, expected=expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
