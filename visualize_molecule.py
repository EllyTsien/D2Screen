"""Utility to visualize molecules stored in ``train_data_list.pkl``.

The dataset created by :mod:`preprocess` serializes each ligand as a dictionary
containing the ``graph`` information produced by
``mol_to_geognn_graph_data_MMFF3d``. This helper loads one entry from the
pickle, reconstructs a molecular graph with NetworkX, and renders it with
Matplotlib.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from collections.abc import Sequence
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from pahelix.utils.compound_tools import CompoundKit

AtomPosDict = Dict[int, Tuple[float, float]]


def _atomic_symbol_from_id(atom_id: int) -> str:
    """Return a human readable label for an atom feature id.

    The stored ``atomic_num`` feature is an index (offset by 1) into
    :data:`CompoundKit.atom_vocab_dict['atomic_num']`.
    """
    if atom_id <= 0:
        return "UNK"

    vocab = CompoundKit.atom_vocab_dict["atomic_num"]
    index = atom_id - 1
    if index < 0 or index >= len(vocab):
        return "UNK"

    value = vocab[index]
    if isinstance(value, int):
        # Map atomic number to element symbol via RDKit's periodic table helper.
        try:
            return CompoundKit.period_table.GetElementSymbol(int(value))
        except Exception:  # pragma: no cover - defensive fallback
            return str(value)
    return str(value)


def _extract_atom_positions(graph_dict: dict) -> AtomPosDict | None:
    """Pull a 2D layout from the pre-computed ``atom_pos`` if available."""
    atom_pos = graph_dict.get("atom_pos")
    if atom_pos is None or len(atom_pos) == 0:
        return None

    positions: AtomPosDict = {}
    for idx, coord in enumerate(atom_pos):
        # ``coord`` can be an ``np.ndarray`` or a list; we only keep x/y.
        x = float(coord[0])
        y = float(coord[1]) if len(coord) > 1 else 0.0
        positions[idx] = (x, y)
    return positions


def build_networkx_graph(graph_dict: dict) -> nx.Graph:
    """Convert a stored molecule graph dictionary into a :class:`networkx.Graph`."""
    nx_graph = nx.Graph()

    atomic_ids = graph_dict.get("atomic_num", [])
    for idx, atom_id in enumerate(atomic_ids):
        atomic_id_int = int(atom_id)
        nx_graph.add_node(
            idx,
            label=_atomic_symbol_from_id(atomic_id_int),
            atomic_feature_id=atomic_id_int,
        )

    seen_edges = set()
    for src, dst in graph_dict.get("edges", []):
        src_idx = int(src)
        dst_idx = int(dst)
        if src_idx == dst_idx:
            # Skip self-loops that were added during preprocessing.
            continue
        edge_key = tuple(sorted((src_idx, dst_idx)))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        nx_graph.add_edge(*edge_key)

    return nx_graph


def draw_molecule(
    nx_graph: nx.Graph,
    smiles: str,
    label: float | int | None,
    positions: AtomPosDict | None,
    output_path: Path | None,
) -> None:
    """Render the molecular graph using NetworkX/Matplotlib."""
    if not positions:
        positions = nx.spring_layout(nx_graph, seed=0)

    node_labels = {node: data.get("label", "?") for node, data in nx_graph.nodes(data=True)}

    plt.figure(figsize=(6, 6))
    nx.draw_networkx(
        nx_graph,
        pos=positions,
        labels=node_labels,
        with_labels=True,
        node_size=800,
        node_color="#8ecae6",
        font_size=10,
        font_weight="bold",
        edge_color="#023047",
    )
    plt.title(f"SMILES: {smiles}\nLabel: {label}")
    plt.axis("off")

    if output_path is not None:
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).resolve().parent / "work" / "train_data_list.pkl",
        help="Path to the pickled train_data_list.pkl file.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the molecule to visualize within the pickled list.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the figure instead of opening a window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path: Path = args.data_path

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file: {data_path}")

    with data_path.open("rb") as handle:
        data_list = pickle.load(handle)

    if not isinstance(data_list, Sequence):
        data_list = list(data_list)

    dataset_size = len(data_list)
    if args.index < 0 or args.index >= dataset_size:
        raise IndexError(
            f"Index {args.index} is out of range for dataset of size {dataset_size}"
        )

    molecule = data_list[args.index]

    smiles = molecule.get("smiles", "<unknown>")
    label = molecule.get("label")
    graph_dict = molecule.get("graph")
    if graph_dict is None:
        raise KeyError("Selected molecule does not contain a 'graph' entry.")

    nx_graph = build_networkx_graph(graph_dict)
    positions = _extract_atom_positions(graph_dict)

    draw_molecule(nx_graph, smiles, label, positions, args.output)


if __name__ == "__main__":
    main()
