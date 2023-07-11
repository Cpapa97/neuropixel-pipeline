from . import ephys
import datajoint as dj

schema = dj.schema("neuropixel_manual_anatomy")


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

    @classmethod
    def restrict_by_scan(cls, scan_key: dict):
        return cls & (ephys.Session & scan_key)

    @classmethod
    def fill(
        cls,
        scan_key: dict,
        insertion_number: int,
        paramset_idx: int,
        curation_id: int,
        segmentation_method: str,
        brain_area: str,
    ):
        cls.insert1(
            dict(
                session_id=ephys.Session.get_session_id(scan_key),
                insertion_number=insertion_number,
                paramset_idx=paramset_idx,
                curation_id=curation_id,
                segmentation_method=segmentation_method,
                brain_area=brain_area,
            )
        )
