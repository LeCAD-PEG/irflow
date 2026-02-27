# src/irflow/models/resources.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional


@dataclass(frozen=True)
class Resources:
    """
    Execution resource request for a step.

    This is a scheduler-agnostic description. LocalExecutor will only use a subset
    (e.g., OMP threads + env). A SlurmExecutor can later translate this into SBATCH.
    """

    # CPU / MPI
    nodes: int = 1
    tasks_per_node: int = 1
    cpus_per_task: int = 1
    mpi: bool = False
    omp_threads: Optional[int] = None  # sets OMP_NUM_THREADS if provided

    # Memory / time
    mem_gb: Optional[float] = None
    mem_per_cpu_gb: Optional[float] = None
    walltime: Optional[str] = None  # "HH:MM:SS"

    # GPU (optional)
    gpus: int = 0
    gpu_type: Optional[str] = None

    # Scheduler hints (optional)
    partition: Optional[str] = None
    qos: Optional[str] = None
    account: Optional[str] = None
    constraint: Optional[str] = None
    exclusive: bool = False

    # Runtime environment
    modules: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    container: Optional[str] = None  # path to container image (apptainer/singularity)

    def total_tasks(self) -> int:
        return int(self.nodes) * int(self.tasks_per_node)

    def validate(self) -> None:
        if self.nodes < 1:
            raise ValueError("Resources.nodes must be >= 1")
        if self.tasks_per_node < 1:
            raise ValueError("Resources.tasks_per_node must be >= 1")
        if self.cpus_per_task < 1:
            raise ValueError("Resources.cpus_per_task must be >= 1")
        if self.omp_threads is not None and self.omp_threads < 1:
            raise ValueError("Resources.omp_threads must be >= 1 if provided")
        if self.gpus < 0:
            raise ValueError("Resources.gpus must be >= 0")
        if self.mem_gb is not None and self.mem_gb <= 0:
            raise ValueError("Resources.mem_gb must be > 0 if provided")
        if self.mem_per_cpu_gb is not None and self.mem_per_cpu_gb <= 0:
            raise ValueError("Resources.mem_per_cpu_gb must be > 0 if provided")
