from __future__ import annotations

import time
from pathlib import Path
from pydantic import BaseModel, Field

from ..utils import extract_data_from_bin

MEAN_WAVEFORMS_FILE = "mean_waveforms.npy"
WAVEFORM_METRICS_FILE = "waveform_metrics.csv"
QUALITY_METRICS_FILE = "metrics.csv"


# i.e. Waveforms and QualityMetrics
# runs ecephys_spike_sorting to produce the waveform analysis and quality metrics files


# https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/mean_waveforms/_schemas.py
class WaveformMetricsRunner(BaseModel):
    generic_params: WaveformMetricsRunner.GenericParams = Field(
        serialization_alias="ephys_params",
        default_factory=lambda: WaveformMetricsRunner.GenericParams(),
    )
    params: WaveformMetricsRunner.Params = Field(
        serialization_alias="mean_waveform_params",
        default_factory=lambda: WaveformMetricsRunner.Params(),
    )

    class GenericParams(BaseModel):
        bit_volts: float = Field(
            description="Scalar required to convert int16 values into microvolts",
        )
        sample_rate: float = Field(
            default=30000.0,
            description="Sample rate of Neuropixels AP band continuous data",
        )
        num_channels: int = Field(
            default=384, description="Total number of channels in binary data files"
        )
        vertical_site_spacing: float = Field(
            default=20e-6, description="Vertical site spacing in meters"
        )

    class Params(BaseModel):
        samples_per_spike: int = Field(
            default=82, description="Number of samples to extract for each spike"
        )
        pre_samples: int = Field(
            default=20,
            description="Number of samples between start of spike and the peak",
        )
        num_epochs: int = Field(
            default=1, description="Number of epochs to compute mean waveforms"
        )
        spikes_per_epoch: int = Field(
            default=100, description="Max number of spikes per epoch"
        )
        upsampling_factor: float = Field(
            default=200 / 82,
            description="Upsampling factor for calculating waveform metrics",
        )
        spread_threshold: float = Field(
            default=0.12,
            description="Threshold for computing channel spread of 2D waveform",
        )
        site_range: int = Field(
            default=16, description="Number of sites to use for 2D waveform metrics"
        )

    def calculate(self, kilosort_output_dir: Path, bin_file: Path, has_sync_channel=False):
        ### Calculating waveform metrics required reimplementing calculate_mean_waveforms
        ### This is because it directly accesses data (which requires a channel count of 384)
        ### But we have an extra sync channel, so that needs to be handled on our side instead.

        from ecephys_spike_sorting.modules.mean_waveforms.__main__ import (
            load_kilosort_data,
            extract_waveforms,
            writeDataAsNpy,
        )

        kilosort_output_dir = Path(kilosort_output_dir)
        mean_waveforms_file = kilosort_output_dir / MEAN_WAVEFORMS_FILE
        waveform_metrics_file = kilosort_output_dir / WAVEFORM_METRICS_FILE


        print("ecephys spike sorting: mean waveforms module")

        start = time.time()

        print("Loading data...")

        data = extract_data_from_bin(
            bin_file=bin_file,
            num_channels=self.generic_params.num_channels,
            has_sync_channel=has_sync_channel,
        )

        (
            spike_times,
            spike_clusters,
            spike_templates,
            amplitudes,
            templates,
            channel_map,
            clusterIDs,
            cluster_quality,
        ) = load_kilosort_data(
            kilosort_output_dir,
            self.generic_params.sample_rate,
            convert_to_seconds=False,
        )

        print("Calculating mean waveforms...")

        waveforms, spike_counts, coords, labels, metrics = extract_waveforms(
            data,
            spike_times,
            spike_clusters,
            templates,
            channel_map,
            self.generic_params.bit_volts,
            self.generic_params.sample_rate,
            self.generic_params.vertical_site_spacing,
            self.params.model_dump(by_alias=True),
        )

        writeDataAsNpy(waveforms, mean_waveforms_file)
        metrics.to_csv(waveform_metrics_file)

        execution_time = time.time() - start

        print(f"total time: {round(execution_time, 2)} seconds")

        return {"execution_time": execution_time}  # output manifest


# https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/_schemas.py
class QualityMetricsRunner(BaseModel):
    generic_params: QualityMetricsRunner.GenericParams = Field(
        serialization_alias="ephys_params",
        default_factory=lambda: QualityMetricsRunner.GenericParams(),
    )
    params: QualityMetricsRunner.Params = Field(
        serialization_alias="quality_metrics_params",
        default_factory=lambda: QualityMetricsRunner.Params(),
    )

    class GenericParams(BaseModel):
        sample_rate: float = Field(
            default=30000.0,
            description="Sample rate of Neuropixels AP band continuous data",
        )
        num_channels: int = Field(
            default=384, description="Total number of channels in binary data files"
        )

    class Params(BaseModel):
        isi_threshold: float = Field(
            default=0.0015, description="Maximum time (in seconds) for ISI violation"
        )
        min_isi: float = Field(
            default=0.00, description="Minimum time (in seconds) for ISI violation"
        )
        num_channels_to_compare: int = Field(
            default=13,
            description="Number of channels to use for computing PC metrics; must be odd",
        )
        max_spikes_for_unit: int = Field(
            default=500,
            description="Number of spikes to subsample for computing PC metrics",
        )
        max_spikes_for_nn: int = Field(
            default=10000,
            description="Further subsampling for NearestNeighbor calculation",
        )
        n_neighbors: int = Field(
            default=4,
            description="Number of neighbors to use for NearestNeighbor calculation",
        )
        n_silhouette: int = Field(
            default=10000,
            description="Number of spikes to use for calculating silhouette score",
        )
        drift_metrics_min_spikes_per_interval: int = Field(
            default=10, description="Minimum number of spikes for computing depth"
        )
        drift_metrics_interval_s: float = Field(
            default=100.0,
            description="Interval length is seconds for computing spike depth",
        )
        include_pc_metrics: bool = Field(
            default=False,
            description="Compute features that require principal components",
        )

    def calculate(self, kilosort_output_dir: Path):
        from ecephys_spike_sorting.modules.quality_metrics.__main__ import (
            calculate_quality_metrics,
        )

        kilosort_output_dir = Path(kilosort_output_dir)

        args = self.model_dump(by_alias=True)
        args["quality_metrics_params"]["quality_metrics_output_file"] = (
            kilosort_output_dir / QUALITY_METRICS_FILE
        )
        args["directories"] = {"kilosort_output_directory": kilosort_output_dir}
        args["waveform_metrics"] = {
            "waveform_metrics_file": kilosort_output_dir / WAVEFORM_METRICS_FILE
        }
        return calculate_quality_metrics(args)
