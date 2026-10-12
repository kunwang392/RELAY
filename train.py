"""Launch RELAY, or inspect its configuration without loading data or starting Ray."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent


def validate_method(cfg):
    required = {
        "use_grpo_opsd": True, "use_opsd": True, "use_opd": True,
        "is_pure_opd": False, "opsd_teacher_update_mode": "fixed",
        "opsd_loss_type": "forward_kl", "opsd_feedback_mode": "none",
        "opsd_score_ablation": "unit", "opsd_distill_auto_switch": True,
        "rubric_train_positive_only": False,
    }
    for key, expected in required.items():
        if cfg.get(key) != expected:
            raise ValueError(f"The RELAY entry point requires {key}={expected!r}.")
    for key in ("use_cripo", "use_ruscarl", "use_rubric_rlsd", "opsd_probe_enabled"):
        if cfg.get(key, False):
            raise ValueError(f"{key} is not included in this RELAY-only package.")
    if cfg.get("saf_opd_topk_percent", 100) != 100 or cfg.get("saf_opd_tanh_scale") is not None:
        raise ValueError("SAF weighting is not part of RELAY.")
    if cfg.get("opsd_teacher_refresh_step") is not None:
        raise ValueError("Teacher-refresh ablations are not included.")
    if cfg.response_length != 5120 or cfg.validation.generating_args.max_new_tokens != 5120:
        raise ValueError("RELAY training and validation require a 5120-token response limit.")
    if cfg.val_sequence_length - cfg.val_prompt_length != 5120:
        raise ValueError("Validation sequence length minus prompt length must equal 5120 response tokens.")


def build_config(args, overrides):
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt))
    model_size = args.model or "1.7b"
    model = args.model_path or (os.environ.get("PRETRAIN_MODEL") if args.model is None else None) or {
        "1.7b": "Qwen/Qwen3-1.7B", "4b": "Qwen/Qwen3-4B",
    }[model_size]
    os.environ["PRETRAIN_MODEL"] = model
    variables = {
        "TRAIN_FILE": args.train_file, "VALIDATION_FILE": args.validation_file,
        "OUTPUT_DIR": args.output_dir,
    }
    for key, value in variables.items():
        if value is not None:
            os.environ[key] = str(Path(value).expanduser().resolve())
    if args.print_config:
        # Display-only placeholders: never supplied to an actual training run.
        defaults = {
            "TRAIN_FILE": "/path/to/train.parquet",
            "VALIDATION_FILE": "/path/to/validation.parquet",
            "RUBRIC_JUDGE_API_BASE": "http://localhost:8000/v1",
            "RUBRIC_JUDGE_MODEL": "your-judge-model",
        }
        for key, value in defaults.items():
            os.environ.setdefault(key, value)
    else:
        for key in ("TRAIN_FILE", "VALIDATION_FILE", "RUBRIC_JUDGE_API_BASE", "RUBRIC_JUDGE_MODEL"):
            if not os.environ.get(key):
                raise ValueError(f"Set {key} before training; see README.md.")
        train_path = Path(os.environ["TRAIN_FILE"]).expanduser().resolve()
        val_path = Path(os.environ["VALIDATION_FILE"]).expanduser().resolve()
        for key, path in (("TRAIN_FILE", train_path), ("VALIDATION_FILE", val_path)):
            if not path.is_file():
                raise ValueError(f"{key} must point to an existing external parquet file.")
        if train_path == val_path:
            raise ValueError("Training and validation files must be separate.")
        os.environ["TRAIN_FILE"], os.environ["VALIDATION_FILE"] = str(train_path), str(val_path)
    cfg = OmegaConf.load(ROOT / "configs/relay.yaml")
    cfg.exp_name = f"relay-{model_size}-{args.domain}-${{now:%Y%m%d-%H%M%S}}"
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    cfg.pop("hydra", None)
    OmegaConf.resolve(cfg)
    validate_method(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["1.7b", "4b"], help="Default: 1.7b; an explicit choice overrides PRETRAIN_MODEL")
    parser.add_argument("--model-path", help="Override the model hub ID with local weights or another ID")
    parser.add_argument("--domain", choices=["medicine", "science"], default="medicine")
    parser.add_argument("--train-file")
    parser.add_argument("--validation-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--print-config", action="store_true", help="Resolve configuration and exit; no data or GPU access")
    parser.add_argument("--check-runtime", action="store_true", help="Construct runtime config and check GPUs, without training")
    args, overrides = parser.parse_known_args()
    cfg = build_config(args, overrides)
    if args.print_config:
        # Authentication comes exclusively from the environment, never from YAML.
        print(OmegaConf.to_yaml(cfg))
        return
    for key, value in cfg.system_envs.items():
        os.environ[key] = str(value)
    os.environ.setdefault("RUBRIC_JUDGE_API_KEY", "EMPTY")
    from dacite import Config, from_dict
    from roll.pipeline.rlvr.rlvr_config import RLVRConfig
    runtime = from_dict(RLVRConfig, OmegaConf.to_container(cfg), config=Config(check_types=False))
    import torch
    needed = int(cfg.num_gpus_per_node)
    if torch.cuda.device_count() < needed:
        raise RuntimeError(f"This configuration needs {needed} visible NVIDIA GPUs.")
    if args.check_runtime:
        print(json.dumps({"configuration": "valid", "visible_gpus": torch.cuda.device_count()}))
        return
    from roll.distributed.scheduler.initialize import init
    from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline
    init()
    RLVRPipeline(pipeline_config=runtime).run()


if __name__ == "__main__":
    main()
