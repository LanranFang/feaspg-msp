"""Isolated coupled-availability adapter for the Aspinall development case.

The frozen primary suite binds the shared projection, simulator, and SDDP
modules byte-for-byte.  Aspinall therefore opts into its richer cascade
polytope through this explicit runtime adapter rather than changing those
modules.  Every non-Aspinall call is delegated to the original implementation.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import Any

import numpy as np
import torch

from .config import ReservoirSystem


ADAPTER_VERSION = "aspinall_coupled_active_faces_v1"
AVAILABILITY_METADATA_KEY = "availability_semantics"
AVAILABILITY_METADATA_VALUE = "same_stage_upstream_release"


def is_aspinall_coupled(system: ReservoirSystem) -> bool:
    metadata = dict(system.metadata)
    return (
        system.benchmark == "rise_uc_hdb_v1"
        and system.case_id == "aspinall_3"
        and metadata.get(AVAILABILITY_METADATA_KEY)
        == AVAILABILITY_METADATA_VALUE
    )


def validate_coupled_system(system: ReservoirSystem) -> None:
    if not is_aspinall_coupled(system):
        raise ValueError("System does not opt into the Aspinall coupled polytope.")
    routing = np.asarray(system.routing, dtype=np.float64)
    reservoirs = system.n_reservoirs
    indegree = np.count_nonzero(routing > 0.0, axis=0).astype(int)
    frontier = [index for index, value in enumerate(indegree) if value == 0]
    visited = 0
    while frontier:
        source = frontier.pop()
        visited += 1
        for target in np.flatnonzero(routing[source] > 0.0):
            indegree[target] -= 1
            if indegree[target] == 0:
                frontier.append(int(target))
    if visited != reservoirs:
        raise ValueError("Same-stage availability requires an acyclic network.")


@lru_cache(maxsize=16)
def _projection_operators(
    routing_rows: tuple[tuple[float, ...], ...],
    has_shared_limit: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute affine projections onto all independent polytope faces."""

    routing = np.asarray(routing_rows, dtype=np.float64)
    reservoirs = routing.shape[0]
    identity = np.eye(reservoirs, dtype=np.float64)
    constraints = [-identity, identity, identity - routing.T]
    if has_shared_limit:
        constraints.append(np.ones((1, reservoirs), dtype=np.float64))
    matrix = np.concatenate(constraints, axis=0)
    count = matrix.shape[0]

    proposal_maps: list[np.ndarray] = [identity]
    rhs_maps: list[np.ndarray] = [np.zeros((reservoirs, count))]
    for size in range(1, reservoirs + 1):
        for active in combinations(range(count), size):
            active_indices = np.asarray(active, dtype=int)
            face = matrix[active_indices]
            gram = face @ face.T
            if np.linalg.matrix_rank(gram, tol=1.0e-11) != size:
                continue
            lift = face.T @ np.linalg.inv(gram)
            proposal_maps.append(identity - lift @ face)
            rhs_map = np.zeros((reservoirs, count), dtype=np.float64)
            rhs_map[:, active_indices] = lift
            rhs_maps.append(rhs_map)
    return matrix, np.stack(proposal_maps), np.stack(rhs_maps)


def _operators(system: ReservoirSystem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_coupled_system(system)
    routing = tuple(
        tuple(float(value) for value in row) for row in system.routing
    )
    return _projection_operators(
        routing, system.shared_release_capacity is not None
    )


def _rhs_torch(storage: torch.Tensor, system: ReservoirSystem) -> torch.Tensor:
    capacity = torch.as_tensor(
        system.release_capacity, dtype=storage.dtype, device=storage.device
    ).expand_as(storage)
    rhs = torch.cat(
        (torch.zeros_like(storage), capacity, torch.clamp(storage, min=0.0)),
        dim=-1,
    )
    if system.shared_release_capacity is not None:
        rhs = torch.cat(
            (
                rhs,
                torch.full_like(
                    storage[..., :1], float(system.shared_release_capacity)
                ),
            ),
            dim=-1,
        )
    return rhs


def _rhs_np(storage: np.ndarray, system: ReservoirSystem) -> np.ndarray:
    capacity = np.broadcast_to(
        np.asarray(system.release_capacity, dtype=np.float64), storage.shape
    )
    rhs = np.concatenate(
        (np.zeros_like(storage), capacity, np.maximum(storage, 0.0)), axis=-1
    )
    if system.shared_release_capacity is not None:
        rhs = np.concatenate(
            (
                rhs,
                np.full(
                    (*storage.shape[:-1], 1),
                    float(system.shared_release_capacity),
                    dtype=np.float64,
                ),
            ),
            axis=-1,
        )
    return rhs


def project_release_coupled(
    proposal: torch.Tensor,
    storage: torch.Tensor,
    system: ReservoirSystem,
    bisection_steps: int = 50,
) -> torch.Tensor:
    """Exact Euclidean projection onto same-stage cascade availability."""

    del bisection_steps
    if proposal.shape != storage.shape or proposal.shape[-1] != system.n_reservoirs:
        raise ValueError("Projection inputs must share shape [..., reservoir].")
    matrix_np, proposal_np, rhs_np = _operators(system)
    matrix = torch.as_tensor(matrix_np, dtype=proposal.dtype, device=proposal.device)
    proposal_map = torch.as_tensor(
        proposal_np, dtype=proposal.dtype, device=proposal.device
    )
    rhs_map = torch.as_tensor(rhs_np, dtype=proposal.dtype, device=proposal.device)
    rhs = _rhs_torch(storage, system)
    candidates = torch.einsum("...j,sij->...si", proposal, proposal_map)
    candidates = candidates + torch.einsum("...k,sik->...si", rhs, rhs_map)
    residual = (
        torch.einsum("...si,ki->...sk", candidates, matrix)
        - rhs.unsqueeze(-2)
    )
    scale = torch.maximum(
        torch.ones_like(rhs[..., :1]), rhs.abs().amax(dim=-1, keepdim=True)
    )
    tolerance = 128.0 * torch.finfo(proposal.dtype).eps * scale
    feasible = (residual <= tolerance.unsqueeze(-2)).all(dim=-1)
    distance = ((candidates - proposal.unsqueeze(-2)) ** 2).sum(dim=-1)
    distance = torch.where(
        feasible, distance, torch.full_like(distance, torch.inf)
    )
    selected = distance.argmin(dim=-1)
    index = selected[..., None, None].expand(
        *selected.shape, 1, system.n_reservoirs
    )
    projected = candidates.gather(-2, index).squeeze(-2)
    if not torch.isfinite(distance.amin(dim=-1)).all():
        raise RuntimeError("Coupled projection found no feasible active face.")
    return projected


def project_release_coupled_np(
    proposal: np.ndarray,
    storage: np.ndarray,
    system: ReservoirSystem,
    bisection_steps: int = 70,
) -> np.ndarray:
    del bisection_steps
    proposal = np.asarray(proposal, dtype=np.float64)
    storage = np.asarray(storage, dtype=np.float64)
    if proposal.shape != storage.shape or proposal.shape[-1] != system.n_reservoirs:
        raise ValueError("Projection inputs must share shape [..., reservoir].")
    matrix, proposal_map, rhs_map = _operators(system)
    rhs = _rhs_np(storage, system)
    candidates = np.einsum("...j,sij->...si", proposal, proposal_map)
    candidates += np.einsum("...k,sik->...si", rhs, rhs_map)
    residual = np.einsum("...si,ki->...sk", candidates, matrix)
    residual -= rhs[..., None, :]
    scale = np.maximum(1.0, np.max(np.abs(rhs), axis=-1, keepdims=True))
    tolerance = 128.0 * np.finfo(np.float64).eps * scale
    feasible = np.all(residual <= tolerance[..., None, :], axis=-1)
    distance = np.sum((candidates - proposal[..., None, :]) ** 2, axis=-1)
    distance = np.where(feasible, distance, np.inf)
    selected = np.argmin(distance, axis=-1)
    index = np.broadcast_to(
        selected[..., None, None],
        (*selected.shape, 1, system.n_reservoirs),
    )
    projected = np.take_along_axis(candidates, index, axis=-2)[..., 0, :]
    if not np.all(np.isfinite(np.min(distance, axis=-1))):
        raise RuntimeError("Coupled projection found no feasible active face.")
    return projected


def proposal_violation_coupled(
    proposal: torch.Tensor,
    storage: torch.Tensor,
    system: ReservoirSystem,
) -> torch.Tensor:
    capacity = torch.as_tensor(
        system.release_capacity, dtype=proposal.dtype, device=proposal.device
    )
    routing = torch.as_tensor(
        system.routing, dtype=proposal.dtype, device=proposal.device
    )
    available = storage + proposal @ routing
    violations = torch.stack(
        (
            torch.relu(-proposal).amax(dim=-1),
            torch.relu(proposal - capacity).amax(dim=-1),
            torch.relu(proposal - available).amax(dim=-1),
        ),
        dim=-1,
    ).amax(dim=-1)
    if system.shared_release_capacity is not None:
        violations = torch.maximum(
            violations,
            torch.relu(proposal.sum(dim=-1) - float(system.shared_release_capacity)),
        )
    return violations


_INSTALLED = False
_ORIGINALS: dict[str, Any] = {}


def install_aspinall_runtime() -> None:
    """Install idempotent dispatch hooks without editing frozen source files."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import online as online_module
    from . import projection as projection_module
    from . import rl as rl_module
    from . import sddp as sddp_module
    from . import simulator as simulator_module

    _ORIGINALS.update(
        {
            "project_release": projection_module.project_release,
            "project_release_np": projection_module.project_release_np,
            "proposal_violation": projection_module.proposal_violation,
            "trace_max_action_violation": (
                simulator_module.SimulationTrace.max_action_violation.fget
            ),
            "sddp_stage_init": sddp_module._DecisionHazardStage.__init__,
        }
    )

    def project_dispatch(proposal, storage, system, bisection_steps=50):
        if is_aspinall_coupled(system):
            return project_release_coupled(
                proposal, storage, system, bisection_steps=bisection_steps
            )
        return _ORIGINALS["project_release"](
            proposal, storage, system, bisection_steps=bisection_steps
        )

    def project_np_dispatch(proposal, storage, system, bisection_steps=70):
        if is_aspinall_coupled(system):
            return project_release_coupled_np(
                proposal, storage, system, bisection_steps=bisection_steps
            )
        return _ORIGINALS["project_release_np"](
            proposal, storage, system, bisection_steps=bisection_steps
        )

    def violation_dispatch(proposal, storage, system):
        if is_aspinall_coupled(system):
            return proposal_violation_coupled(proposal, storage, system)
        return _ORIGINALS["proposal_violation"](proposal, storage, system)

    def max_action_violation(trace):
        if not is_aspinall_coupled(trace.system):
            return _ORIGINALS["trace_max_action_violation"](trace)
        capacity = torch.as_tensor(
            trace.system.release_capacity,
            dtype=trace.actions.dtype,
            device=trace.actions.device,
        )
        routing = torch.as_tensor(
            trace.system.routing,
            dtype=trace.actions.dtype,
            device=trace.actions.device,
        )
        available = trace.storage_before_action + trace.actions @ routing
        violations = torch.stack(
            (
                torch.relu(-trace.actions).amax(),
                torch.relu(trace.actions - capacity).amax(),
                torch.relu(trace.actions - available).amax(),
            )
        )
        if trace.system.shared_release_capacity is not None:
            violations = torch.cat(
                (
                    violations,
                    torch.relu(
                        trace.actions.sum(dim=-1)
                        - float(trace.system.shared_release_capacity)
                    ).amax().view(1),
                )
            )
        return violations.amax()

    def sddp_stage_init(stage_self, *args, **kwargs):
        _ORIGINALS["sddp_stage_init"](stage_self, *args, **kwargs)
        if not is_aspinall_coupled(stage_self.system):
            return
        routing = np.asarray(stage_self.system.routing, dtype=np.float64)
        stage_self.model.update()
        for reservoir in range(stage_self.system.n_reservoirs):
            constraint = stage_self.model.getConstrByName(
                f"available_water[{reservoir}]"
            )
            if constraint is None:
                raise RuntimeError("SDDP availability constraint is missing.")
            stage_self.model.remove(constraint)
        stage_self.model.update()
        for reservoir in range(stage_self.system.n_reservoirs):
            routed = sddp_module.gp.quicksum(
                float(routing[source, reservoir]) * stage_self.action[source]
                for source in range(stage_self.system.n_reservoirs)
            )
            stage_self.model.addConstr(
                stage_self.action[reservoir]
                <= stage_self.incoming_storage[reservoir] + routed,
                name=f"available_water[{reservoir}]",
            )
        stage_self.model.update()

    projection_module.project_release = project_dispatch
    projection_module.project_release_np = project_np_dispatch
    projection_module.proposal_violation = violation_dispatch
    simulator_module.project_release = project_dispatch
    simulator_module.proposal_violation = violation_dispatch
    simulator_module.SimulationTrace.max_action_violation = property(
        max_action_violation
    )
    rl_module.project_release_np = project_np_dispatch
    online_module.project_release = project_dispatch
    online_module.proposal_violation = violation_dispatch
    sddp_module._DecisionHazardStage.__init__ = sddp_stage_init
    _INSTALLED = True


__all__ = [
    "ADAPTER_VERSION",
    "AVAILABILITY_METADATA_KEY",
    "AVAILABILITY_METADATA_VALUE",
    "install_aspinall_runtime",
    "is_aspinall_coupled",
    "project_release_coupled",
    "project_release_coupled_np",
    "proposal_violation_coupled",
    "validate_coupled_system",
]
