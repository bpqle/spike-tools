import os
import yaml
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import spikeinterface.full as si
from probeinterface import get_probe
from probeinterface.plotting import plot_probe

import warnings
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("-r", "recording_folder", help="OpenEphys Recording Folder", type=Path, required=True)
parser.add_argument("-o", "output_folder", help="Folder to write sorting results to", type=Path, required=True)
parser.add_argument("-p", "probe", help="Probe Config File", type=Path, required=True)
parser.add_argument("-s", "sorter", help="Sorter Config File", type=Path, required=True)
p = parser.parse_args()

abs_path = os.path.abspath(os.path.dirname(__file__))
with open(Path(abs_path / 'configs/general_cfg.yml')) as cfg_f:
    cfg = yaml.safe_load(cfg_f)
with open(Path(abs_path / 'configs/metrics.yml')) as mtrc_f:
    metrics = yaml.safe_load(mtrc_f)

recording_path = p.recording_folder
try:
    assert recording_path.exists()
except:
    raise ValueError("Recording Folder not found.")    
    
if p.probe.suffix == '.yml': # download saved probe
    with open(p.probe) as probe_file:
        probe_config = yaml.safe_load(probe_file)
    probe = get_probe(probe_config['manufacturer'], probe_config['model'])
    probe.wiring_to_device(probe_config['wiring'])
elif p.probe.suffix == '.csv':
    from .utils import create_probe
    probe = create_probe(p.probe)
else:
    raise ValueError("Undefined probe config format file.")

try:
    assert p.sorter.is_file()
    sorter = p.sorter.stem
    docker_img = sorter.pop('docker')
except:
    raise ValueError('Sorter config is not a file.')
with open(p.sorter) as sorter_file: # import config for sorting algorithm
    sparams = yaml.safe_load(sorter_file)
if sorter == "kilosort2_5":
    si.Kilosort2_5Sorter.set_kilosort2_5_path(cfg['sorters'][sorter])
elif sorter == "waveclus":
    si.WaveClusSorter.set_waveclus_path(cfg['sorters'][sorter])
elif sorter == "ironclus":
    si.IronClustSorter.set_ironclust_path(cfg['sorters'][sorter])
elif sorter == "kilosort3":
    si.Kilosort3Sorter. set_kilosort3_path(cfg['sorters'][sorter])
assert sorter in si.installed_sorters()

base_output = p.output_folder
base_output.mkdir(parents=True, exist_ok=True)
sorter_folder = Path(base_output / f'results-{sorter}')
waveform_folder = Path(base_output / f'waveforms-{sorter}')
phy_folder = Path(base_output / f'phy-{sorter}')

# n_cpus = os.cpu_count()
# n_jobs = n_cpus - 4
# job_kwargs = dict(n_jobs=n_jobs, progress_bar=True)
# si.set_global_job_kwargs(**job_kwargs)

full_raw_rec = si.read_openephys(recording_path)
raw_rec = full_raw_rec.set_probe(probe)
recording_f = si.bandpass_filter(raw_rec, freq_min=cfg['filter']['freq_min'], freq_max=cfg['filter']['freq_max'])
recording_cmr = si.common_reference(recording_f, reference='global', operator='median')

print(f'Begin Sorting. Sorting algorithm: {sorter}. Use docker image: {docker_img}. Manual Parameters: {sparams}')
sorter_params = si.get_default_sorter_params(sorter)
sorter_params.update(sparams)
sorting_data = si.run_sorter(sorter_name=sorter,
                             recording=recording_cmr,
                             remove_existing_folder=True,
                             output_folder=sorter_folder,
                             docker_image=docker_img,
                             verbose=True,
                             **sorter_params)

print("Sorting completed. Extracting waveforms.")
we = si.extract_waveforms(recording_cmr,
                          sorting_data,
                          folder = waveform_folder,
                          sparse = False, overwrite=True)
sparsity = si.compute_sparsity(we, method=cfg['sparsity']['method'], radius_um=cfg['sparsity']['radius_um'])
we = si.extract_waveforms(recording_cmr,
                          sorting_data,
                          folder = waveform_folder,
                          sparsity = sparsity,
                          overwrite=True)
print("Calculating pca.")
_ = si.compute_principal_components(we,
                                     mode=cfg['pca']['mode'],
                                     n_components=cfg['pca']['n_components'],
                                     load_if_exists=False)
print("Calculating metrics")
metric_names = list(metrics.keys())
default_metrics =  si.get_default_qm_params()
metric_params = {k: v for k,v in default_metrics.items() if k not in metric_names}
for k in metric_names:
    metric_params[k].update(metrics[k])
_ = si.compute_quality_metrics(we, metric_names=metric_names, verbose=True,  qm_params=metric_params)

print("Saving to phy.")
si.export_to_phy(we,
                 output_folder=phy_folder,
                 compute_amplitudes=True,
                 # compute_pc_features=True,
                 remove_if_exists=True,
                 copy_binary=True)

si.plot_rasters(sorting_data, time_range=(0, 60), backend="matplotlib")