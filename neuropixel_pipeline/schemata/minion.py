import datajoint as dj

from . import SCHEMA_PREFIX

schema = dj.schema(SCHEMA_PREFIX + "minion")


@schema
class MinionVersion(dj.Lookup):
    definition = """
    # Minion versioning that can be used to differentiate no-longer runnable tasks
    version: varchar(255) # Might want it as an int instead for easier filtering?
    ---
    description: varchar(1000)
    """


@schema
class MinionInput(dj.Manual):
    definition = """
    # Stores the input to the minion pipeline for task queuing
    task_id: bigint unsigned auto_increment # auto-incrementing task id
    ---
    -> MinionVersion
    inserted=CURRENT_TIMESTAMP: timestamp
    params: longblob # all parameters related to running the task
    """


@schema
class MinionOutput(dj.Computed):
    definition = """
    # Stores the finished tasks, as well as optional output from the pipeline minion
    -> MinionInput
    ---
    output=NULL: longblob # Optional output from minion
    """

    def make(self, key):
        raise NotImplementedError("Override the make method when running the minion")
