"""Generate three fresh evaluation runs and score exported RELAY checkpoints with GPT-5.1."""
import argparse
import asyncio
import json
import os
from pathlib import Path

from roll.evaluation.final_eval import (
    DOMAIN_BENCHMARKS, REPEATS, VLLMGenerator, combine_domains, file_digest,
    generate_records, judge_directory, protocol_settings, validate_rows, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Generate 3 fresh runs and grade them; needs candidate GPUs and judge API")
    run.add_argument("--model-path", required=True, help="Exported Hugging Face checkpoint directory")
    run.add_argument("--model-size", required=True, choices=["1.7b", "4b"])
    run.add_argument("--domain", required=True, choices=list(DOMAIN_BENCHMARKS))
    for name in {n for values in DOMAIN_BENCHMARKS.values() for n in values}:
        run.add_argument("--" + name.replace("_", "-"), dest=name, help="External prepared test parquet")
    run.add_argument("--output-dir", required=True, help="A new directory for responses, grades and summaries")
    run.add_argument("--tensor-parallel-size", type=int, default=4)
    run.add_argument("--max-model-len", type=int, default=32768)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--print-config", action="store_true", help="Show protocol without reading datasets or loading models")
    judge = commands.add_parser("judge", help="Resume grading saved responses; needs no candidate GPU")
    judge.add_argument("--run-dir", required=True)
    for command in (run, judge):
        command.add_argument("--concurrency", type=int, default=16)
        command.add_argument("--attempts", type=int, default=3)
    aggregate = commands.add_parser("aggregate", help="Combine medicine/science results for one model size")
    aggregate.add_argument("summaries", nargs=2)
    aggregate.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "aggregate":
        destination = Path(args.output).resolve()
        if destination.exists():
            parser.error("Aggregate output already exists; choose a new path.")
        summary = combine_domains([json.loads(Path(p).read_text()) for p in args.summaries])
        write_json(destination, summary)
        print(json.dumps({"output": str(destination), "aggregate": summary["aggregate"]}))
        return
    if args.concurrency < 1 or args.attempts < 1:
        parser.error("Concurrency and attempts must be positive.")
    if args.command == "run" and args.print_config:
        print(json.dumps(protocol_settings(), indent=2))
        return
    if not os.environ.get("GPT_EVAL_API_KEY"):
        parser.error("Set GPT_EVAL_API_KEY privately in the environment.")
    if args.command == "judge":
        summary = asyncio.run(judge_directory(args.run_dir, args.concurrency, args.attempts))
    else:
        if args.batch_size < 1 or args.tensor_parallel_size < 1 or args.max_model_len <= 5120:
            parser.error("Invalid generation capacity.")
        if not Path(args.model_path).is_dir():
            parser.error("--model-path must be an exported Hugging Face checkpoint directory.")
        paths = {}
        for name in DOMAIN_BENCHMARKS[args.domain]:
            value = getattr(args, name)
            if not value or not Path(value).is_file():
                parser.error(f"Provide --{name.replace('_', '-')} with the external evaluation parquet.")
            paths[name] = Path(value).resolve()
        for domain, names in DOMAIN_BENCHMARKS.items():
            if domain != args.domain and any(getattr(args, name) for name in names):
                parser.error("Use separate domain checkpoints for medicine and science.")
        if len(set(paths.values())) != len(paths):
            parser.error("Each benchmark must have a separate input file.")
        directory = Path(args.output_dir).resolve()
        if directory.exists():
            parser.error("Output directory exists; use 'judge' to retry grading or choose a new directory.")
        from datasets import Dataset
        datasets = {name: validate_rows(name, list(Dataset.from_parquet(str(path)))) for name, path in paths.items()}
        directory.mkdir(parents=True)
        manifest = {"domain": args.domain, "model_size": args.model_size,
                    "model_path": str(Path(args.model_path).resolve()), "seed": args.seed,
                    "tensor_parallel_size": args.tensor_parallel_size, "max_model_len": args.max_model_len,
                    "protocol": protocol_settings(), "inputs": {
                        name: {"file": path.name, "sha256": file_digest(path), "rows": len(datasets[name])}
                        for name, path in paths.items()}}
        write_json(directory / "manifest.json", manifest)
        generator = VLLMGenerator(args.model_path, args.tensor_parallel_size, args.max_model_len)
        for repeat in range(REPEATS):
            for name, rows in datasets.items():
                path = directory / f"repeat-{repeat + 1}-{name}.responses.jsonl"
                with path.open("x") as target:
                    for record in generate_records(name, rows, generator, repeat, args.seed, args.batch_size):
                        target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                print(f"Generated repeat {repeat + 1}/{REPEATS}: {name}", flush=True)
        summary = asyncio.run(judge_directory(directory, args.concurrency, args.attempts))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
