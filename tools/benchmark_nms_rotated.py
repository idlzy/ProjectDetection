#!/usr/bin/env python3
"""Benchmark the project CUDA rotated NMS across candidate counts."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from project_detection.ops import nms_rotated


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[100, 300, 1000])
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_inputs(count, generator):
    boxes = torch.empty((count, 5), device="cuda")
    boxes[:, :2].normal_(generator=generator).mul_(20.0)
    boxes[:, 2:4].uniform_(0.2, 8.0, generator=generator)
    boxes[:, 4].uniform_(-torch.pi, torch.pi, generator=generator)
    scores = torch.rand(count, device="cuda", generator=generator)
    return boxes, scores


def benchmark(count, args, generator):
    boxes, scores = make_inputs(count, generator)
    for _ in range(args.warmup):
        nms_rotated(
            boxes,
            scores,
            args.threshold,
            pairwise_chunk_size=args.chunk_size,
        )
    torch.cuda.synchronize()

    elapsed_ms = []
    kept = 0
    for _ in range(args.repeats):
        started = time.perf_counter()
        _, keep = nms_rotated(
            boxes,
            scores,
            args.threshold,
            pairwise_chunk_size=args.chunk_size,
        )
        torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
        kept = keep.numel()
    return {
        "candidate_count": count,
        "kept_count": kept,
        "mean_ms": statistics.mean(elapsed_ms),
        "median_ms": statistics.median(elapsed_ms),
        "min_ms": min(elapsed_ms),
        "max_ms": max(elapsed_ms),
        "samples_ms": elapsed_ms,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    report = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "chunk_size": args.chunk_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "threshold": args.threshold,
        "results": [benchmark(count, args, generator) for count in args.counts],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
