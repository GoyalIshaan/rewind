"""Exercise the compiled dialect through its actual parser and verifier."""

from pathlib import Path
import subprocess
import sys
import unittest


REWIND_OPT = str(Path(sys.argv.pop(1)).resolve())
EXAMPLE = Path(__file__).with_name("history.mlir").read_text()


class DialectTests(unittest.TestCase):
    def run_opt(self, source, *flags):
        return subprocess.run(
            [REWIND_OPT, *flags], input=source, text=True, capture_output=True
        )

    def accept(self, source, *flags):
        result = self.run_opt(source, *flags)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def reject(self, source, diagnostic):
        result = self.run_opt(source)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(diagnostic, result.stderr)

    def test_supported_floating_types(self):
        for element in ("f16", "bf16", "f32", "f64"):
            with self.subTest(element=element):
                self.accept(EXAMPLE.replace("f64", element))

    def test_text_round_trip(self):
        printed = self.accept(EXAMPLE)
        self.assertIn("!rewind.store<tensor<4x3xf64>>", printed)
        self.assertEqual(self.accept(printed), printed)

    def test_generic_syntax_round_trip(self):
        generic = self.accept(EXAMPLE, "--mlir-print-op-generic")
        self.assertIn('"rewind.append"', generic)
        self.assertEqual(self.accept(generic), self.accept(EXAMPLE))

    def test_zero_updates_and_boundary_zero(self):
        source = "\n".join(line for line in EXAMPLE.splitlines()
                           if "rewind.append" not in line)
        source = source.replace(
            "  %selected =", "  %zero = arith.constant 0 : index\n  %selected ="
        ).replace("%h[%accepted]", "%h[%zero]")
        self.accept(source)

    def test_invalid_store_types(self):
        for state_type in ("tensor<4xf64>", "tensor<?x3xf64>",
                           "tensor<4x3xi64>", "f64", "tensor<*xf64>"):
            with self.subTest(state_type=state_type):
                self.reject(
                    EXAMPLE.replace("!rewind.store<tensor<4x3xf64>>",
                                    f"!rewind.store<{state_type}>"),
                    "store requires an unencoded, static rank-2 floating-point tensor",
                )

    def test_initial_type_mismatch(self):
        # Store/result change together; only the captured input disagrees.
        source = EXAMPLE.replace("!rewind.store<tensor<4x3xf64>>",
                                 "!rewind.store<tensor<5x3xf64>>")
        self.reject(source, "initial tensor must match the store's state type")

    def test_update_rank_and_length_mismatches(self):
        for before, after, diagnostic in (
            ("tensor<f64>", "tensor<1xf64>", "scale must be a rank-0 tensor"),
            ("tensor<4xf64>", "tensor<5xf64>", "value must be a vector"),
            ("tensor<3xf64>", "tensor<2xf64>", "key must be a vector"),
            ("tensor<4xf64>", "tensor<2x2xf64>", "value must be a vector"),
            ("tensor<3xf64>", "tensor<1x3xf64>", "key must be a vector"),
        ):
            with self.subTest(after=after):
                self.reject(EXAMPLE.replace(before, after), diagnostic)

    def test_mixed_and_integer_update_elements(self):
        for before, after in (
            ("tensor<f64>", "tensor<f32>"),
            ("tensor<f64>", "tensor<i64>"),
            ("tensor<4xf64>", "tensor<4xf32>"),
            ("tensor<3xf64>", "tensor<3xf32>"),
        ):
            with self.subTest(after=after):
                self.reject(EXAMPLE.replace(before, after),
                            "must share the store's floating element type")

    def test_dynamic_update_shapes(self):
        self.reject(EXAMPLE.replace("tensor<4xf64>", "tensor<?xf64>"),
                    "update operands must have static shapes and no encoding")

    def test_encoded_state_rejected(self):
        self.reject(EXAMPLE.replace("!rewind.store<tensor<4x3xf64>>",
                                    '!rewind.store<tensor<4x3xf64, "layout">>'),
                    "store requires an unencoded, static rank-2 floating-point tensor")

    def test_encoded_update_rejected(self):
        self.reject(EXAMPLE.replace("tensor<4xf64>", 'tensor<4xf64, "layout">'),
                    "update operands must have static shapes and no encoding")

    def test_retrieved_type_mismatch(self):
        source = EXAMPLE.replace(
            "%accepted: index) -> tensor<4x3xf64>",
            "%accepted: index) -> tensor<3x4xf64>",
        ).replace(
            "-> tensor<4x3xf64>\n", "-> tensor<3x4xf64>\n"
        ).replace("return %selected : tensor<4x3xf64>",
                  "return %selected : tensor<3x4xf64>")
        self.reject(source, "result tensor must match the store's state type")

    def test_index_requires_index_type(self):
        self.reject(EXAMPLE.replace("%accepted: index", "%accepted: i64"),
                    "index")

    def test_static_verification_does_not_claim_dynamic_bounds(self):
        # This MUST parse: bounds failure belongs to future runtime lowering.
        source = EXAMPLE.replace(
            "  %selected =", "  %bad = arith.constant -1 : index\n  %selected ="
        ).replace("%h[%accepted]", "%h[%bad]")
        self.accept(source)

    def test_canonicalization_preserves_unused_fallible_retrieval(self):
        extra = ("  %unused = rewind.retrieve %h[%accepted] : "
                 "!rewind.store<tensor<4x3xf64>> -> tensor<4x3xf64>\n")
        source = EXAMPLE.replace("  return %selected", extra + "  return %selected")
        printed = self.accept(source, "--canonicalize", "--cse")
        self.assertEqual(printed.count("rewind.retrieve"), 2)
        self.assertIn("rewind.append", printed)

    def test_equal_initial_values_still_create_independent_stores(self):
        extra = ("  %other = rewind.initialize_store %initial : "
                 "tensor<4x3xf64> -> !rewind.store<tensor<4x3xf64>>\n"
                 "  rewind.append %other, %a, %v, %k : "
                 "!rewind.store<tensor<4x3xf64>>, tensor<f64>, "
                 "tensor<4xf64>, tensor<3xf64>\n")
        source = EXAMPLE.replace("  return %selected", extra + "  return %selected")
        printed = self.accept(source, "--canonicalize", "--cse")
        self.assertEqual(printed.count("rewind.initialize_store"), 2)
        self.assertEqual(printed.count("rewind.append"), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
