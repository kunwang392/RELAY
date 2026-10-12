import time

import ray

from roll.configs.base_config import PPOConfig
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import reduce_metrics_list


class ModelUpdateGroup:
    def __init__(self, src_cluster: Cluster, tgt_cluster: Cluster, pipeline_config: PPOConfig, frequency=1):
        self.src_cluster = src_cluster
        self.tgt_cluster = tgt_cluster
        self.frequency = frequency
        self.pipeline_config = pipeline_config
        self.model_update_name = f"model_update/{self.src_cluster.cluster_name}_2_{self.tgt_cluster.cluster_name}"
        self.megatron_snapshot = (
            src_cluster.worker_config.strategy_args.strategy_name == "megatron_train"
            and tgt_cluster.worker_config.strategy_args.strategy_name == "megatron_infer"
        )
        if self.megatron_snapshot:
            source = src_cluster.worker_config.strategy_args.strategy_config
            target = tgt_cluster.worker_config.strategy_args.strategy_config
            for key in ("tensor_model_parallel_size", "pipeline_model_parallel_size",
                        "context_parallel_size", "expert_model_parallel_size",
                        "virtual_pipeline_model_parallel_size"):
                if source.get(key, 1) != target.get(key, 1):
                    raise ValueError(f"Megatron teacher snapshots require matching {key}")
            if len(src_cluster.workers) != len(tgt_cluster.workers):
                raise ValueError("Megatron teacher snapshots require matching worker counts")
            return
        train_devices = set(src_cluster.worker_config.device_mapping or [])
        infer_devices = set(tgt_cluster.worker_config.device_mapping or [])

        assert (max(train_devices) - min(train_devices)) == (len(train_devices) - 1), f"{train_devices=} must be continuous"
        assert (max(infer_devices) - min(infer_devices)) == (len(infer_devices) - 1), f"{infer_devices=} must be continuous"

        ray.get(
            [
                train_worker.setup_model_update.remote(
                    infer_cluster=self.tgt_cluster, model_update_name=self.model_update_name
                )
                for train_worker in self.src_cluster.workers
            ]
        )

    def model_update(self, step=None):
        if step % self.frequency != 0:
            return {}

        if self.megatron_snapshot:
            started = time.monotonic()
            snapshots = [worker.export_teacher_snapshot.remote() for worker in self.src_cluster.workers]
            # Finish and offload every source rank before loading colocated teachers.
            ray.get([worker.import_teacher_snapshot.remote(snapshot)
                     for worker, snapshot in zip(self.tgt_cluster.workers, self._ready_snapshots(snapshots))])
            return {f"time/{self.model_update_name}": time.monotonic() - started,
                    f"{self.model_update_name}/source_step": step}

        dataprotos: list[DataProto] = ray.get(
            [
                train_worker.start_model_update.remote(model_update_name=self.model_update_name)
                for train_worker in self.src_cluster.workers
            ]
        )
        return reduce_metrics_list([dataproto.meta_info["metrics"] for dataproto in dataprotos])

    @staticmethod
    def _ready_snapshots(snapshots):
        ray.wait(snapshots, num_returns=len(snapshots))
        return snapshots
