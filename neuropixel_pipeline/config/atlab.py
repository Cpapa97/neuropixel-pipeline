from . import PipelineConfig

atlab_mouse_pipeline_config = PipelineConfig(
    use_global_config=True,
    generic_directory_suffix=dict(generic="raw", replacement=None),
    session_source="MouseSource",
)

atlab_monkey_pipeline_config = PipelineConfig(
    use_global_config=True,
    generic_directory_suffix=dict(generic="raw", replacement=None),
    session_source="MonkeySource",
)
