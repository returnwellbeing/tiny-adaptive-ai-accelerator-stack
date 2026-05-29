#!/usr/bin/env python3

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path


OP_PATTERN = re.compile(r'"?((?:stablehlo|chlo)\.[a-zA-Z0-9_]+)"?')
TENSOR_PATTERN = re.compile(r'tensor<([^>]+)>')
ARG_PATTERN = re.compile(r'%arg\d+: tensor<([^>]+)>')
DOT_CONTRACTING_PATTERN = re.compile(r'contracting_dims\s*=\s*\[([^\]]*)\]\s*x\s*\[([^\]]*)\]')


def classify_op(op: str) -> str:
    if op in {
        "chlo.square",
        "stablehlo.add",
        "stablehlo.subtract",
        "stablehlo.negate",
        "stablehlo.multiply",
        "stablehlo.divide",
        "stablehlo.rsqrt",
        "stablehlo.sqrt",
        "stablehlo.exponential",
        "stablehlo.log",
        "stablehlo.maximum",
        "stablehlo.minimum",
    }:
        return "elementwise"

    if op in {
        "stablehlo.reduce",
        "stablehlo.reduce_window",
    }:
        return "reduction"

    if op in {
        "stablehlo.broadcast_in_dim",
        "stablehlo.reshape",
        "stablehlo.transpose",
        "stablehlo.slice",
        "stablehlo.dynamic_slice",
        "stablehlo.concatenate",
    }:
        return "shape/layout"

    if op in {
        "stablehlo.dot_general",
        "stablehlo.convolution",
    }:
        return "compute-heavy"

    if op in {
        "stablehlo.constant",
        "stablehlo.convert",
    }:
        return "utility"

    return "other"


def summarize_stablehlo(path: Path) -> None:
    text = path.read_text()

    ops = OP_PATTERN.findall(text)
    op_counts = Counter(ops)

    class_counts = Counter()
    op_classes = defaultdict(list)

    for op, count in op_counts.items():
        category = classify_op(op)
        class_counts[category] += count
        op_classes[category].append((op, count))

    tensors = TENSOR_PATTERN.findall(text)
    tensor_counts = Counter(tensors)
    estimates = estimate_costs(text, op_counts)
    dot_generals = extract_dot_generals(text)
    tags = detect_workload_tags(class_counts, op_counts)

    print("=" * 80)
    print("StableHLO Summary")
    print("=" * 80)
    print(f"File: {path}")
    print()

    print("[Operation Counts]")
    for op, count in op_counts.most_common():
        print(f"  {op:<32} {count}")

    print()
    print("[Operation Classes]")
    for category, count in class_counts.most_common():
        print(f"  {category:<16} {count}")

    print()
    print("[Ops by Class]")
    for category, items in sorted(op_classes.items()):
        print(f"  {category}")
        for op, count in sorted(items):
            print(f"    - {op:<30} {count}")

    print()
    print("[Tensor Type Hints]")
    for tensor_type, count in tensor_counts.most_common():
        print(f"  tensor<{tensor_type}>  x{count}")

    if dot_generals:
        print()
        print("[dot_general Shapes]")
        for index, dot in enumerate(dot_generals):
            print(f"  dot_general #{index}")
            print(f"    lhs    tensor<{dot['lhs']}>")
            print(f"    rhs    tensor<{dot['rhs']}>")
            print(f"    output tensor<{dot['output']}>")
            print(f"    GEMM interpretation: M={dot['m']} K={dot['k']} N={dot['n']}")

    print()
    print("[Detected Workload Tags]")
    for tag in tags:
        print(f"  - {tag}")

    print()
    print("[Static Cost Estimates]")
    print(f"  FLOP-equivalent ops       {estimates['flops']}")
    print(f"  Minimum DRAM bytes        {estimates['min_bytes']}")
    print(f"  Arithmetic intensity      {estimates['arithmetic_intensity']:.4f} FLOP/byte")
    print(f"  Systolic array utilization {estimates['systolic_utilization']}")
    for note in estimates["notes"]:
        print(f"  - {note}")

    print()
    print("[Workload Interpretation]")
    interpret_workload(class_counts, op_counts)


def estimate_costs(text: str, op_counts: Counter) -> dict:
    flops = 0

    for line in text.splitlines():
        op_match = OP_PATTERN.search(line)
        if op_match is None:
            continue

        op = op_match.group(1)
        tensor_types = TENSOR_PATTERN.findall(line)
        if not tensor_types:
            continue

        output_elements = tensor_num_elements(tensor_types[-1])
        if output_elements is None:
            continue

        if classify_op(op) == "elementwise":
            flops += output_elements
        elif classify_op(op) == "reduction":
            input_elements = tensor_num_elements(tensor_types[0])
            if input_elements is not None:
                flops += max(input_elements - output_elements, 0)
        elif op == "stablehlo.dot_general":
            dot = parse_dot_general_line(line)
            if dot is not None:
                flops += 2 * dot["m"] * dot["k"] * dot["n"]

    input_tensors = ARG_PATTERN.findall(text)
    output_tensors = find_function_outputs(text)
    min_bytes = sum(tensor_num_bytes(t) or 0 for t in input_tensors + output_tensors)

    arithmetic_intensity = flops / min_bytes if min_bytes else 0.0
    dot_generals = extract_dot_generals(text)
    systolic_utilization = estimate_systolic_utilization(dot_generals)

    notes = [
        "FLOP-equivalent treats rsqrt/divide as one operation each; real latency can be higher.",
        "Minimum DRAM bytes counts function inputs plus outputs once, assuming ideal fusion.",
        "Actual traffic can be higher if intermediates are materialized.",
    ]

    return {
        "flops": flops,
        "min_bytes": min_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "systolic_utilization": systolic_utilization,
        "notes": notes,
    }


def extract_dot_generals(text: str) -> list[dict]:
    dots = []
    for line in text.splitlines():
        if "stablehlo.dot_general" not in line:
            continue
        dot = parse_dot_general_line(line)
        if dot is not None:
            dots.append(dot)
    return dots


def parse_dot_general_line(line: str) -> dict | None:
    tensor_types = TENSOR_PATTERN.findall(line)
    if len(tensor_types) < 3:
        return None

    lhs = tensor_types[0]
    rhs = tensor_types[1]
    output = tensor_types[-1]

    lhs_shape = parse_tensor_shape(lhs)
    rhs_shape = parse_tensor_shape(rhs)
    output_shape = parse_tensor_shape(output)
    if lhs_shape is None or rhs_shape is None or output_shape is None:
        return None

    lhs_contract_dim = len(lhs_shape) - 1
    rhs_contract_dim = 0
    contracting_match = DOT_CONTRACTING_PATTERN.search(line)
    if contracting_match is not None:
        lhs_dims = parse_dim_list(contracting_match.group(1))
        rhs_dims = parse_dim_list(contracting_match.group(2))
        if lhs_dims and rhs_dims:
            lhs_contract_dim = lhs_dims[0]
            rhs_contract_dim = rhs_dims[0]

    k = lhs_shape[lhs_contract_dim]
    if rhs_shape[rhs_contract_dim] != k:
        return None

    n = output_shape[-1]
    m = product(output_shape[:-1])

    return {
        "lhs": lhs,
        "rhs": rhs,
        "output": output,
        "m": m,
        "k": k,
        "n": n,
    }


def parse_dim_list(text: str) -> list[int]:
    if not text.strip():
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def detect_workload_tags(class_counts: Counter, op_counts: Counter) -> list[str]:
    tags = []
    if op_counts["stablehlo.dot_general"] > 0:
        tags.append("GEMM-heavy")
    if class_counts["reduction"] > 0:
        tags.append("reduction-bound")
    if class_counts["elementwise"] >= max(class_counts["compute-heavy"], class_counts["reduction"], 1):
        tags.append("elementwise-heavy")
    if not tags:
        tags.append("unclassified")
    return tags


def estimate_systolic_utilization(dot_generals: list[dict]) -> str:
    if not dot_generals:
        return "not applicable / ~0% (no dot_general)"

    dot = max(dot_generals, key=lambda item: item["m"] * item["n"] * item["k"])
    if dot["m"] < 16:
        return f"shape-limited: M={dot['m']} may underutilize tall systolic arrays"
    return f"GEMM-centric: M={dot['m']} N={dot['n']} K={dot['k']} can use systolic arrays"


def find_function_outputs(text: str) -> list[str]:
    for line in text.splitlines():
        if line.strip().startswith("func.func"):
            output_match = re.search(r'->\s*\((.*?)\)\s*\{', line)
            if output_match is None:
                output_match = re.search(r'->\s*\((.*?)\)\s*\{?', line)
            if output_match is not None:
                return TENSOR_PATTERN.findall(output_match.group(1))
    return []


def tensor_num_bytes(tensor_type: str) -> int | None:
    elements = tensor_num_elements(tensor_type)
    dtype_bytes = tensor_dtype_bytes(tensor_type)
    if elements is None or dtype_bytes is None:
        return None
    return elements * dtype_bytes


def tensor_num_elements(tensor_type: str) -> int | None:
    shape = parse_tensor_shape(tensor_type)
    if shape is None:
        return None
    if not shape:
        return 1
    return product(shape)


def parse_tensor_shape(tensor_type: str) -> list[int] | None:
    shape_dtype = tensor_type.split("x")
    if len(shape_dtype) < 2:
        return []

    dims = shape_dtype[:-1]
    shape = []
    for dim in dims:
        if not dim.isdigit():
            return None
        shape.append(int(dim))
    return shape


def product(values: list[int]) -> int:
    elements = 1
    for dim in values:
        elements *= dim
    return elements


def tensor_dtype_bytes(tensor_type: str) -> int | None:
    dtype = tensor_type.split("x")[-1]
    if dtype in {"f32", "i32"}:
        return 4
    if dtype in {"f16", "bf16", "i16"}:
        return 2
    if dtype in {"i8", "ui8"}:
        return 1
    if dtype in {"f64", "i64"}:
        return 8
    return None


def interpret_workload(class_counts: Counter, op_counts: Counter) -> None:
    has_reduction = class_counts["reduction"] > 0
    has_dot = op_counts["stablehlo.dot_general"] > 0
    has_broadcast = op_counts["stablehlo.broadcast_in_dim"] > 0
    elementwise_count = class_counts["elementwise"]

    if has_dot:
        print("  - This workload contains GEMM-like compute-heavy operations.")
    else:
        print("  - No dot_general found: this is not a GEMM-heavy workload.")

    if has_reduction:
        print("  - Reduction is present: expect latency and memory-access sensitivity.")
        print("  - For RMSNorm, this usually corresponds to mean/sum over hidden dimension.")

    if elementwise_count > 0:
        print(f"  - Elementwise ops are present: count={elementwise_count}.")
        print("  - Fusion opportunities matter for reducing memory traffic.")

    if has_broadcast:
        print("  - Broadcast is present: scalar/vector values are expanded across tensor dimensions.")

    print()
    print("  Hardware/runtime note:")
    if has_dot:
        print("  - Linear layers are typically GEMM-centric and map naturally to systolic arrays.")
        print("  - Useful accelerator questions: tile sizes, array utilization, data reuse, memory bandwidth.")
    elif has_reduction:
        print("  - RMSNorm is typically bandwidth/reduction sensitive, not peak-GEMM limited.")
        print("  - A good backend should fuse reduction-adjacent elementwise ops where possible.")
        print("  - Useful accelerator questions: reduction latency, vector width, memory bandwidth, fusion.")
    else:
        print("  - Elementwise-only workloads are usually bandwidth and fusion sensitive.")
        print("  - Useful accelerator questions: vector width, fusion, memory traffic, activation latency.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mlir_file",
        type=Path,
        help="Path to StableHLO MLIR file",
    )
    args = parser.parse_args()

    summarize_stablehlo(args.mlir_file)


if __name__ == "__main__":
    main()
