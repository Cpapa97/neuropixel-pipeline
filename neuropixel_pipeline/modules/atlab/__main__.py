import logging

from .modes import PipelineInput


def setup_logging(log_level=logging.INFO, handler_mode=True):
    import sys

    if handler_mode:
        root = logging.getLogger()
        root.setLevel(log_level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s: %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        root.addHandler(handler)
    else:
        root = logging.basicConfig(stream=sys.stdout, level=log_level)
    return root


def main(args: PipelineInput):
    setup_logging(handler_mode=False)

    args = PipelineInput.model_validate(args)
    args.run()


if __name__ == "__main__":
    main()
