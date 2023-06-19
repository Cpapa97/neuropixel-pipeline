from pydantic import BaseModel
from typing import Optional
from enum import Enum
from pathlib import Path


class ClusteringTaskMode(str, Enum):
    LOAD = "load"
    TRIGGER = "trigger"


class ClusteringTaskRunner(BaseModel):
    task_mode: ClusteringTaskMode
    data_dir: Path = None
    results_dir: Path = None
    filename: Path = None
    clustering_params: Optional[dict] = None

    def trigger_clustering(self):
        # Locally or using an HTTP request to a REST server
        if self.task_mode is ClusteringTaskMode.TRIGGER:
            try:
                from kilosort_runner.run import KilosortRunner, KilosortParams

                params = KilosortParams.model_validate(self.clustering_params)
                runner = KilosortRunner.model_validate(
                    data_dir=self.data_dir,
                    results_dir=self.results_dir,
                    filename=self.filename,
                    params=params,
                )
                runner.run_kilosort()

            except Exception as e:
                print(f"Caught exception when trying to trigger Kilosort:\n{e}")
        elif self.task_mode is ClusteringTaskMode.LOAD:
            print("task mode set to 'load', not doing anything")
        else:
            raise NotImplementedError(
                f"This task mode '{self.task_mode}' is not supported"
            )