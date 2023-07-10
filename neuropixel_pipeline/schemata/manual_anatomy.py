from . import ephys  # noqa: F401
import datajoint as dj

schema = dj.schema("manual_anatomy", "neuropixel_manual_anatomy")


@schema
class Area(dj.Lookup):
    definition = """
    # area of the brain
    brain_area: varchar(100)
    """

    contents = zip(
        [
            "A",
            "AL",
            "AM",
            "LGN",
            "LI",
            "LLA",
            "LM",
            "MAP",
            "nonvisual",
            "P",
            "PM",
            "POR",
            "RL",
            "SC",
            "unknown",
            "V1",
            "visual",
        ]
    )


@schema
class SegmentationMethod(dj.Lookup):
    definition = """
    # area segmentation method
    segmentation: varchar(100)
    """

    contents = [["manual"]]


@schema
class UnitArea(dj.Manual):
    definition = """
    # brain area membership per unit
    -> ephys.CuratedClustering
    -> SegmentationMethod
    ---
    -> Area
    """
