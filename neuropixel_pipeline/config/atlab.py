from . import PipelineConfig

atlab_pipeline_config = PipelineConfig(
    use_global_config=True,
    generic_directory_suffix=dict(generic="raw", replacement=None),
)
