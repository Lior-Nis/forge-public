"""
Downstream Task Callback for Pretraining Pipelines.

Runs downstream task evaluation (linear probing and finetuning) during pretraining
by launching classification training via subprocess or SBATCH jobs. This ensures
reproducible evaluation identical to manual classification runs while maintaining
proper wandb artifact lineage.

Features:
- Dual execution modes: subprocess (local) or SBATCH (cluster)
- WandB artifact management for checkpoint storage and lineage
- Independent wandb runs for each downstream task
- Proper linear probing (frozen backbone) and finetuning (trainable backbone)
- No result parsing needed - tasks handle their own logging
- Resource isolation with SBATCH mode (no GPU contention)
"""

import logging
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Set

import pytorch_lightning as pl
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DownstreamTaskCallback(pl.Callback):
    """
    Downstream task callback using subprocess or SBATCH execution with wandb artifacts.

    Runs linear probing and/or finetuning evaluation during pretraining by:
    1. Saving current pretraining weights as wandb artifacts
    2. Launching tasks via subprocess (local) or SBATCH (cluster)
    3. Each task creates its own wandb run with proper tagging
    4. Automatic artifact lineage tracking in wandb
    """

    def __init__(self, downstream_config: DictConfig):
        """
        Initialize downstream task callback.

        Args:
            downstream_config: Configuration containing:
                - run_every_n_epochs: Frequency of downstream task evaluation
                - downstream_task_type: 'linear_probe', 'finetune', or 'both'
                - execution_mode: 'subprocess' (local) or 'sbatch' (cluster)
                - sbatch_config: SBATCH resource configuration (only for sbatch mode)
                    - partition: GPU partition name
                    - time: Job time limit (e.g., '0-02:00:00' for 2 hours)
                    - gpu_type: GPU type (e.g., 'rtx_4090' or 'a6000')
                    - script_dir: Directory to save SBATCH scripts (default: /tmp/downstream_tasks)
                    - project_dir: FORGE checkout used by the job (default: current directory)
        """
        super().__init__()

        self.run_every_n_epochs = downstream_config.get("run_every_n_epochs", 5)
        self.downstream_task_type = downstream_config.get("downstream_task_type", "both")
        self.execution_mode = downstream_config.get("execution_mode", "subprocess")

        if self.execution_mode == "sbatch" and platform.system() == "Windows":
            raise NotImplementedError(
                "SBATCH execution mode requires Linux/Unix (Slurm scheduler is not available on Windows). "
                "Use execution_mode='subprocess' instead."
            )

        # SBATCH configuration (only used in sbatch mode)
        self.sbatch_config = downstream_config.get("sbatch_config", {})
        if self.execution_mode == "sbatch":
            # Set defaults for sbatch config (matching cluster conventions)
            self.sbatch_config.setdefault("partition", "main")
            self.sbatch_config.setdefault("time", "0-02:00:00")
            self.sbatch_config.setdefault("gpu_type", "rtx_4090")
            self.sbatch_config.setdefault("script_dir", "/tmp/downstream_tasks")
            self.sbatch_config.setdefault("project_dir", str(Path.cwd()))

            # Create script directory if it doesn't exist
            os.makedirs(self.sbatch_config["script_dir"], exist_ok=True)
            logger.info(f"SBATCH mode enabled - scripts will be saved to {self.sbatch_config['script_dir']}")

        # Track running tasks for cleanup
        if self.execution_mode == "subprocess":
            self.running_processes: Set[subprocess.Popen] = set()
        else:  # sbatch mode
            self.running_job_ids: Set[str] = set()

        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="downstream_task")

        logger.info(f"Downstream task callback initialized - running every {self.run_every_n_epochs} epochs")
        logger.info(f"Execution mode: {self.execution_mode}")
        task_type_to_log = ['Probing', 'Finetuning'] if self.downstream_task_type == 'both'\
                            else [self.downstream_task_type]
        logger.info(f"Task types: {task_type_to_log}") # TODO: Support dynamic downstream tasks incl. regression


    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the end of each training epoch."""
        current_epoch = trainer.current_epoch

        # Check if this is an evaluation epoch
        if current_epoch > 0 and current_epoch % self.run_every_n_epochs == 0:
            logger.info(f"Running downstream task evaluation at epoch {current_epoch}")

            try:
                # Save current checkpoint as wandb artifact
                artifact_name = self._save_checkpoint_to_wandb_artifact(trainer, pl_module, current_epoch)

                # Run downstream tasks based on configuration and execution mode
                if self.execution_mode == "subprocess":
                    if self.downstream_task_type in ['linear_probe', 'both']:
                        logger.info("Launching linear probe downstream task (async subprocess)")
                        self.executor.submit(self._run_linear_probe_subprocess, artifact_name, current_epoch)

                    if self.downstream_task_type in ['finetune', 'both']:
                        logger.info("Launching finetuning downstream task (async subprocess)")
                        self.executor.submit(self._run_finetuning_subprocess, artifact_name, current_epoch)

                elif self.execution_mode == "sbatch":
                    if self.downstream_task_type in ['linear_probe', 'both']:
                        logger.info("Launching linear probe downstream task (SBATCH)")
                        self.executor.submit(self._run_linear_probe_sbatch, artifact_name, current_epoch)

                    if self.downstream_task_type in ['finetune', 'both']:
                        logger.info("Launching finetuning downstream task (SBATCH)")
                        self.executor.submit(self._run_finetuning_sbatch, artifact_name, current_epoch)

                logger.info(f"Downstream task evaluation launched at epoch {current_epoch} (mode: {self.execution_mode})")

            except Exception as e:
                logger.error(f"Downstream task evaluation failed at epoch {current_epoch}: {e}")
                logger.exception("Full traceback:")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when training ends - cleanup running tasks."""
        self._cleanup_tasks()

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        """Called when an exception occurs - cleanup running tasks."""
        self._cleanup_tasks()

    def _cleanup_tasks(self) -> None:
        """Clean up any running downstream tasks (processes or SBATCH jobs)."""
        if self.execution_mode == "subprocess":
            if hasattr(self, 'running_processes') and self.running_processes:
                logger.info(f"Cleaning up {len(self.running_processes)} running downstream processes...")

                for process in self.running_processes.copy():
                    try:
                        if process.poll() is None:  # Process is still running
                            logger.info(f"Terminating downstream process PID {process.pid}")
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                logger.warning(f"Process PID {process.pid} didn't terminate, killing...")
                                process.kill()
                                process.wait()
                    except Exception as e:
                        logger.error(f"Error cleaning up process: {e}")

                self.running_processes.clear()

        elif self.execution_mode == "sbatch":
            if hasattr(self, 'running_job_ids') and self.running_job_ids:
                logger.info(f"Cleaning up {len(self.running_job_ids)} running SBATCH jobs...")

                for job_id in self.running_job_ids.copy():
                    try:
                        logger.info(f"Canceling SBATCH job {job_id}")
                        subprocess.run(['scancel', job_id], check=False, capture_output=True)
                    except Exception as e:
                        logger.error(f"Error canceling job {job_id}: {e}")

                self.running_job_ids.clear()

        # Shutdown executor
        if hasattr(self, 'executor'):
            logger.info("Shutting down downstream task executor...")
            self.executor.shutdown(wait=True, cancel_futures=True)

    def __del__(self):
        """Destructor - ensure cleanup on garbage collection."""
        try:
            self._cleanup_tasks()
        except Exception:
            pass  # Ignore errors during cleanup in destructor

    def _save_checkpoint_to_wandb_artifact(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        current_epoch: int
    ) -> str:
        """
        Save current pretraining checkpoint as wandb artifact.

        Args:
            trainer: Current trainer
            pl_module: Pretraining module
            current_epoch: Current epoch

        Returns:
            Artifact reference string for loading
        """
        if not trainer.logger or not hasattr(trainer.logger, 'experiment'):
            raise ValueError("WandB logger not found - cannot save checkpoint artifact")

        try:
            import wandb

            # Create temporary checkpoint file
            with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as temp_file:
                temp_path = temp_file.name

            # Save checkpoint
            trainer.save_checkpoint(temp_path)

            # Create wandb artifact
            artifact_name = f"pretraining_checkpoint_epoch_{current_epoch}"
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"Pretraining checkpoint at epoch {current_epoch}",
                metadata={
                    "epoch": current_epoch,
                    "global_step": trainer.global_step,
                    "model_type": pl_module.__class__.__name__
                }
            )

            # Add checkpoint file to artifact
            artifact.add_file(temp_path, name="checkpoint.ckpt")

            # Log artifact to wandb
            run = trainer.logger.experiment
            run.log_artifact(artifact)

            # Clean up temporary file
            os.unlink(temp_path)

            # Return artifact reference for downstream loading
            artifact_ref = f"{run.entity}/{run.project}/{artifact_name}:latest"
            logger.info(f"Saved checkpoint artifact: {artifact_ref}")

            return artifact_ref

        except Exception as e:
            logger.error(f"Failed to save checkpoint artifact: {e}")
            # Clean up temp file on error
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _run_linear_probe_subprocess(self, artifact_ref: str, current_epoch: int) -> None:
        """
        Run linear probe downstream task via subprocess with verbose real-time output.

        Args:
            artifact_ref: WandB artifact reference for checkpoint loading
            current_epoch: Current pretraining epoch
        """
        try:
            # Prepare subprocess command
            script_path = PROJECT_ROOT / "scripts" / "train.py"

            cmd = [
                sys.executable, str(script_path),
                "experiment=downstream/linear_probe_binary",
                f"weight_manager.load_from_registry={artifact_ref}",
                f"train.logger.segmentation.name=linear_probe_epoch_{current_epoch}"
            ]

            logger.info(f"[LINEAR_PROBE] Starting subprocess: {' '.join(cmd)}")

            # Execute subprocess with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=PROJECT_ROOT,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )

            # Track process for cleanup
            self.running_processes.add(process)

            try:
                # Stream output in real-time
                for line in process.stdout:
                    if line.strip():
                        logger.info(f"[LINEAR_PROBE] {line.strip()}")

                # Wait for completion
                return_code = process.wait(timeout=1800)  # 30 minute timeout

                if return_code == 0:
                    logger.info("[LINEAR_PROBE] Subprocess completed successfully")
                else:
                    logger.error(f"[LINEAR_PROBE] Subprocess failed with return code {return_code}")

            except subprocess.TimeoutExpired:
                logger.error("[LINEAR_PROBE] Subprocess timed out, terminating...")
                process.kill()
                process.wait()
            finally:
                # Remove from tracking set
                self.running_processes.discard(process)

        except Exception as e:
            logger.error(f"[LINEAR_PROBE] Failed to run subprocess: {e}")

    def _run_finetuning_subprocess(self, artifact_ref: str, current_epoch: int) -> None:
        """
        Run finetuning downstream task via subprocess with verbose real-time output.

        Args:
            artifact_ref: WandB artifact reference for checkpoint loading
            current_epoch: Current pretraining epoch
        """
        try:
            # Prepare subprocess command
            script_path = PROJECT_ROOT / "scripts" / "train.py"

            cmd = [
                sys.executable, str(script_path),
                "experiment=downstream/finetune_binary",
                f"weight_manager.load_from_registry={artifact_ref}",
                f"train.logger.wandb.name=finetune_epoch_{current_epoch}"
            ]

            logger.info(f"[FINETUNE] Starting subprocess: {' '.join(cmd)}")

            # Execute subprocess with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=PROJECT_ROOT,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )

            # Track process for cleanup
            self.running_processes.add(process)

            try:
                # Stream output in real-time
                for line in process.stdout:
                    if line.strip():
                        logger.info(f"[FINETUNE] {line.strip()}")

                # Wait for completion
                return_code = process.wait(timeout=3600)  # 60 minute timeout

                if return_code == 0:
                    logger.info("[FINETUNE] Subprocess completed successfully")
                else:
                    logger.error(f"[FINETUNE] Subprocess failed with return code {return_code}")

            except subprocess.TimeoutExpired:
                logger.error("[FINETUNE] Subprocess timed out, terminating...")
                process.kill()
                process.wait()
            finally:
                # Remove from tracking set
                self.running_processes.discard(process)

        except Exception as e:
            logger.error(f"[FINETUNE] Failed to run subprocess: {e}")

    def _generate_sbatch_script(
        self,
        task_name: str,
        experiment_config: str,
        artifact_ref: str,
        current_epoch: int
    ) -> Path:
        """
        Generate SBATCH script for downstream task.

        Args:
            task_name: Task name ('linear_probe' or 'finetune')
            experiment_config: Hydra experiment config path
            artifact_ref: WandB artifact reference
            current_epoch: Current pretraining epoch

        Returns:
            Path to generated SBATCH script
        """
        script_dir = Path(self.sbatch_config["script_dir"])
        script_path = script_dir / f"{task_name}_epoch_{current_epoch}.sh"
        project_path = Path(self.sbatch_config["project_dir"]).resolve()
        project_dir = shlex.quote(str(project_path))
        source_dir = shlex.quote(str(project_path / "src"))

        # Generate SBATCH script content (matching cluster conventions)
        script_content = f"""#!/bin/bash
#SBATCH --partition {self.sbatch_config["partition"]}
#SBATCH --time {self.sbatch_config["time"]}
#SBATCH --job-name {task_name}_ep{current_epoch}
#SBATCH --output {script_dir}/{task_name}_epoch_{current_epoch}_%J.out
#SBATCH --gpus={self.sbatch_config["gpu_type"]}:1

# Downstream task: {task_name} at pretraining epoch {current_epoch}
# WandB artifact: {artifact_ref}

export PYTHONPATH={source_dir}:"${{PYTHONPATH:-}}"
cd {project_dir}

echo "Starting {task_name} evaluation at pretraining epoch {current_epoch}"
echo "Using checkpoint: {artifact_ref}"

HYDRA_FULL_ERROR=1 python scripts/train.py \\
    experiment={experiment_config} \\
    weight_manager.load_from_registry={artifact_ref} \\
    train.logger.wandb.name={task_name}_epoch_{current_epoch}

echo "{task_name} evaluation complete!"
"""

        # Write script to file
        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make script executable (no-op on Windows)
        if platform.system() != "Windows":
            os.chmod(script_path, 0o755)

        logger.info(f"Generated SBATCH script: {script_path}")
        return script_path

    def _submit_sbatch_job(self, script_path: Path) -> Optional[str]:
        """
        Submit SBATCH job and return job ID.

        Args:
            script_path: Path to SBATCH script

        Returns:
            Job ID string if successful, None otherwise
        """
        try:
            result = subprocess.run(
                ['sbatch', str(script_path)],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse job ID from sbatch output (format: "Submitted batch job 12345")
            match = re.search(r'Submitted batch job (\d+)', result.stdout)
            if match:
                job_id = match.group(1)
                logger.info(f"Submitted SBATCH job {job_id}")
                return job_id
            else:
                logger.error(f"Failed to parse job ID from sbatch output: {result.stdout}")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit SBATCH job: {e}")
            logger.error(f"stderr: {e.stderr}")
            return None

    def _run_linear_probe_sbatch(self, artifact_ref: str, current_epoch: int) -> None:
        """
        Run linear probe downstream task via SBATCH job.

        Args:
            artifact_ref: WandB artifact reference for checkpoint loading
            current_epoch: Current pretraining epoch
        """
        try:
            logger.info(f"[LINEAR_PROBE] Generating SBATCH script for epoch {current_epoch}")

            # Generate SBATCH script
            script_path = self._generate_sbatch_script(
                task_name="linear_probe",
                experiment_config="downstream/linear_probe_binary",
                artifact_ref=artifact_ref,
                current_epoch=current_epoch
            )

            # Submit job
            job_id = self._submit_sbatch_job(script_path)

            if job_id:
                self.running_job_ids.add(job_id)
                logger.info(f"[LINEAR_PROBE] SBATCH job {job_id} submitted successfully")
            else:
                logger.error("[LINEAR_PROBE] Failed to submit SBATCH job")

        except Exception as e:
            logger.error(f"[LINEAR_PROBE] Failed to launch SBATCH job: {e}")

    def _run_finetuning_sbatch(self, artifact_ref: str, current_epoch: int) -> None:
        """
        Run finetuning downstream task via SBATCH job.

        Args:
            artifact_ref: WandB artifact reference for checkpoint loading
            current_epoch: Current pretraining epoch
        """
        try:
            logger.info(f"[FINETUNE] Generating SBATCH script for epoch {current_epoch}")

            # Generate SBATCH script
            script_path = self._generate_sbatch_script(
                task_name="finetune",
                experiment_config="downstream/finetune_binary",
                artifact_ref=artifact_ref,
                current_epoch=current_epoch
            )

            # Submit job
            job_id = self._submit_sbatch_job(script_path)

            if job_id:
                self.running_job_ids.add(job_id)
                logger.info(f"[FINETUNE] SBATCH job {job_id} submitted successfully")
            else:
                logger.error("[FINETUNE] Failed to submit SBATCH job")

        except Exception as e:
            logger.error(f"[FINETUNE] Failed to launch SBATCH job: {e}")
