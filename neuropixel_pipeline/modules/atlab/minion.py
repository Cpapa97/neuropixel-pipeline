"""
TODO: Will move out of the atlab module once PipelineMode is generalized as well.
"""

import datajoint as dj

from .modes import PipelineInput

schema = dj.schema("neuropixel_minion")

# ------------ Tasks --------------

@schema
class IngestionTask(dj.Manual):
    definition = """
    # Task that should be triggered or data to be ingested
    request_start=CURRENT_TIMESTAMP: timestamp # timestamp when ingestion is requested
    ---
    params: longblob # parameters to be passed to a minion
    """

    # TODO: Change to accomodate PipelineInput/PipelineMode?
    @classmethod
    def add_curation_task(cls, pipeline_params: PipelineInput):
        cls.insert1(
            dict(
                params=pipeline_params,
            )
        )


@schema
class IngestionTaskFinished(dj.Computed):
    definition = """
    # Ingestion task finished
    -> IngestionTask
    ---
    request_end=CURRENT_TIMESTAMP: timestamp # timestamp when ingestion is finished
    """

    def make(self, key):
        pass