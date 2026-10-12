"""Strategies used by the RELAY release (Megatron training/teacher, vLLM rollout)."""
from roll.utils.asyncio_decorator import create_sync_class


def create_strategy(worker, sync_wrapper=False):
    name = worker.worker_config.strategy_args.strategy_name
    if name == "vllm":
        from roll.distributed.strategy.vllm_strategy import VllmStrategy as cls
    elif name == "megatron_train":
        from roll.distributed.strategy.megatron_strategy import MegatronTrainStrategy as cls
    elif name == "megatron_infer":
        from roll.distributed.strategy.megatron_strategy import MegatronInferStrategy as cls
    else:
        raise ValueError("This RELAY package supports only megatron_train, megatron_infer, and vllm.")
    return (create_sync_class(cls) if sync_wrapper else cls)(worker)
