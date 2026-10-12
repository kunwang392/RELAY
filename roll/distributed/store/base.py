"""
Offload store abstraction.

Offload backends are equal-level implementations selected by config —
no sequential layering.

Data flow:
    offload: backend.put_tensors(key, params) → GPU memory freed
    reload:  backend.get_tensors(key, params, device=gpu) → params rebuilt on GPU


"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor

from roll.utils.logging import get_logger

logger = get_logger()


def as_flats(tensor: Union[Tensor, Sequence[Tensor]]) -> List[Tensor]:
    """Normalize a put() source into a list of flat views (logical concatenation)."""
    if isinstance(tensor, Tensor):
        return [tensor.detach().reshape(-1)]
    return [t.detach().reshape(-1) for t in tensor]


def copy_flat_slice(dst: Tensor, flats: Sequence[Tensor], start: int, numel: int) -> bool:
    """Copy the logical range [start, start+numel) of the concatenation of
    `flats` into dst[0:numel] (non-blocking). Returns True if any source
    slice was on GPU (caller decides whether/how to synchronize)."""
    end = start + numel
    offset = 0
    from_gpu = False
    for flat in flats:
        n = flat.numel()
        lo, hi = max(start, offset), min(end, offset + n)
        if lo < hi:
            dst.narrow(0, lo - start, hi - lo).copy_(
                flat.narrow(0, lo - offset, hi - lo), non_blocking=True)
            from_gpu |= flat.device.type != "cpu"
        offset += n
    return from_gpu


def slice_flats(flats: Sequence[Tensor], start: int, numel: int) -> List[Tensor]:
    """Views covering the logical range [start, start+numel) of the concatenation of `flats`."""
    end = start + numel
    out: List[Tensor] = []
    offset = 0
    for flat in flats:
        n = flat.numel()
        lo, hi = max(start, offset), min(end, offset + n)
        if lo < hi:
            out.append(flat.narrow(0, lo - offset, hi - lo))
        offset += n
    return out


def shrink_cuda_storage(data: Tensor) -> None:
    """Free the device memory backing `data` even if untracked views still alias it.

    Rebinding .data only drops one reference; mcore keeps several untracked
    views of the DDP/optimizer buffers alive (shard groups, cached param/grad
    buffer shards, bucket views), which pin the allocation. Shrinking the
    storage itself — the same mechanism as mcore's own
    ParamAndGradBuffer.offload_to_cpu — returns the memory regardless.
    """
    if data.device.type != "cpu" and data.untyped_storage().size() > 0:
        data.untyped_storage().resize_(0)


class OffloadBackend(ABC):
    """Abstract offload backend. Implementations are equal-level and selected by config."""

    def __init__(self):
        # key -> per-tensor (device_mesh, placements, local_shape) recorded by
        # put_tensors; mesh/placements are None for plain (non-DTensor) tensors.
        self._tensor_meta: Dict[str, List[Tuple]] = {}

    @abstractmethod
    def _put(self, key: str, tensor: Union[Tensor, List[Tensor]],
             replicated_over: Union[bool, "torch.distributed.ProcessGroup"] = True) -> None:
        """Storage hook: store a tensor, or a list of tensors treated as one flat
        concatenation. replicated_over answers "who else holds this exact copy":
        False = nobody (store a full per-rank copy), True = the backend's
        default replication group, a ProcessGroup = that group (e.g. Megatron
        dense params over DP×CP vs expert params over expert-DP). The group is
        remembered per key, so get/delete never need it.
        All source copies must be complete when this returns."""
        ...

    @abstractmethod
    def _get(self, key: str, device: Optional[Union[torch.device, str]] = None) -> Tensor:
        """Storage hook: retrieve the flat tensor on the target device (default: current GPU)."""
        ...

    @abstractmethod
    def delete(self, key: str, replicated_over: bool = True) -> None:
        """Delete a stored tensor. When replicated_over=True, only dp_rank=0 deletes."""
        ...

    @abstractmethod
    def has_key(self, key: str) -> bool:
        """Check if key exists."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored tensors."""
        ...

    def barrier(self) -> None:
        """Synchronize across DP ranks. No-op for local backends."""
        pass

    def get_stats(self) -> Optional[dict]:
        """Return backend stats for logging. None for backends without stats."""
        return None

    def put_tensors(self, key: str, tensors: List[Tensor],
                    replicated_over: Union[bool, "torch.distributed.ProcessGroup"] = True) -> Optional[str]:
        """Offload a group of tensors under one key and free their GPU memory.

        Thin wrapper over the _put storage hook: record per-tensor meta (DTensor
        mesh/placements or plain shape), store the flat concatenation of local
        shards, rebind each .data to an empty CPU tensor. Not meant to be
        overridden — backend-specific layout lives in _put.
        replicated_over: False = per-rank copy, True = dedup over the backend's
        default replication group, a ProcessGroup = dedup over that group. The
        group is recorded per key, so get_tensors/delete_tensors never take it.
        Returns the key (None if empty).

        Barriers after put when deduplicating (cross-rank write visibility).
        """
        if not tensors:
            return None
        if self.has_key(key):
            raise KeyError(
                f"put_tensors: key '{key}' already exists; delete stale keys before "
                f"re-putting (silent overwrite would leak the previous buffer)"
            )

        metas, locals_ = [], []
        for t in tensors:
            d = t.data
            if isinstance(d, DTensor):
                local = d._local_tensor
                metas.append((d.device_mesh, d.placements, local.shape))
                locals_.append(local)
            else:
                metas.append((None, None, d.shape))
                locals_.append(d)
        self._tensor_meta[key] = metas
        self._put(key, locals_, replicated_over=replicated_over)

        for t, local in zip(tensors, locals_):
            t.data = torch.empty(0, dtype=t.dtype, device="cpu")
            shrink_cuda_storage(local)

        if replicated_over:
            self.barrier()

        return key

    def get_tensors(self, key: str, tensors: List[Tensor],
                    device: torch.device, replicated_over: bool = True) -> None:
        """Retrieve flat buffer from backend, scatter to individual tensors, rebind .data.
        Barriers only when replicated_over=True (cross-rank read-before-delete sync).
        """
        flat = self._get(key, device=device)

        offset = 0
        for t, (mesh, placements, local_shape) in zip(tensors, self._tensor_meta[key]):
            numel = local_shape.numel()
            local = flat.narrow(0, offset, numel).view(local_shape)
            t.data = DTensor.from_local(local, mesh, placements) if mesh is not None else local
            offset += numel
        del flat

        if replicated_over:
            self.barrier()

    def delete_tensors(self, key: str, replicated_over: bool = True) -> None:
        """Delete tensors by key. Barriers only when replicated_over=True."""
        self.delete(key, replicated_over=replicated_over)
        self._tensor_meta.pop(key, None)
        if replicated_over:
            self.barrier()
