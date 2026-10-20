"""Run with python3 -m rewind_mvp.export_hybrid; write the actual captured graph."""

from pathlib import Path

import torch

from rewind_mvp.hybrid import TinyHybridWindow, example_inputs


def main():
    exported = torch.export.export(TinyHybridWindow(), example_inputs())
    destination = Path(__file__).with_name("hybrid_graph.txt")
    destination.write_text(
        f"PyTorch {torch.__version__}; T=3, D=4, K=3, context=2, vocab=7\n"
        "ATen graph; fixed-length Python loop is unrolled; no MLIR conversion yet.\n"
        + exported.graph_module.code)
    print(destination)
    print(f"Graph nodes: {len(list(exported.graph_module.graph.nodes))}")


if __name__ == "__main__":
    main()
