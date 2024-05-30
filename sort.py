import os
import yaml
import argparse
import numpy as np
from pathlib import Path

import spikeinterface.full as si
from probeinterface import get_probe
from probeinterface.plotting import plot_probe

import warnings
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("recording_folder", help="OpenEphys Recording Folder", type=Path)
parser.add_argument("output_folder", help="Folder to write sorting results to", type=Path)
parser.add_argument("probe", help="Probe Config File", type=Path)
parser.add_argument("sorter", help="Sorter Config File", type=Path)
p = parser.parse_args()

with open('./configs/general_cfg.yml') as cfg_f:
    cfg = yaml.safe_load(cfg_f)

recording_path = p.recording_folder
try:
    assert recording_path.exists()
except:
    raise ValueError("Recording Folder not found.")    
    
with open(p.probe) as probe_file:
    probe_config = yaml.safe_load(probe_file)
if probe_config['mode'] == 'import':
    probe = get_probe(probe_config['manufacturer'], probe_config['model'])
    probe.wiring_to_device(probe_config['wiring'])

try:
    assert p.sorter.is_file()
    algorithm = p.sorter.stem
except:
    raise ValueError('Sorter config is not a file.')
with open(p.sorter) as sorter_file:
    sparams = yaml.safe_load(sorter_file)

base_output = p.output_folder
base_output.mkdir(parents=True, exist_ok=True)
preprocess_folder = Path(base_output / 'preprocess')
sorter_folder = Path(base_output / f'results-{algorithm}')
waveform_folder = Path(base_output / f'waveforms-{algorithm}')
phy_folder = Path(base_output / f'phy-{algorithm}')

n_cpus = os.cpu_count()
n_jobs = n_cpus - 4
job_kwargs = dict(n_jobs=n_jobs, progress_bar=True)
si.set_global_job_kwargs(**job_kwargs)

full_raw_rec = si.read_openephys(recording_path)
raw_rec = full_raw_rec.set_probe(probe)
recording_f = si.bandpass_filter(raw_rec, freq_min=cfg['filter']['freq_min'], freq_max=cfg['filter']['freq_max'])
recording_cmr = si.common_reference(recording_f, reference='global', operator='median')
if preprocess_folder.is_dir():
    recording_saved = si.load_extractor(preprocess_folder)
else:
    recording_saved = recording_f.save(folder=preprocess_folder, overwrite=True)
    print(f"Frequency Filter & Common Reference Applied. Preprocessed Recording saved to {preprocess_folder} ")
    print(f'Saved channels ids:\n{recording_saved.get_channel_ids()}')

print(f'Begin Sorting. Sorting algorithm: {algorithm}. Use docker image: {cfg["docker"]}. Manual Parameters: {sparams}')
sorter_params = si.get_default_sorter_params(algorithm)
sorter_params.update(sparams)
sorting_data = si.run_sorter(sorter_name=algorithm,
                             recording=recording_saved,
                             remove_existing_folder=True,
                             output_folder=sorter_folder,
                             docker_image=cfg['docker'],
                             verbose=True,
                             **sorter_params)

print("Sorting completed. Extracting waveforms.")
we = si.extract_waveforms(recording_saved,
                          sorting_data,
                          folder = waveform_folder,
                          sparse = False, overwrite=True)
sparsity = si.compute_sparsity(we, method=cfg['sparsity']['method'], radius_um=cfg['sparsity']['radius_um'])
we = si.extract_waveforms(recording_saved,
                          sorting_data,
                          folder = waveform_folder,
                          sparsity = sparsity,
                          overwrite=True)
print("Calculating pca.")
pc = si.compute_principal_components(we,
                                     mode=cfg['pca']['mode'],
                                     n_components=cfg['pca']['n_components'],
                                     load_if_exists=False)
print("Saving to phy.")
si.export_to_phy(we,
                 output_folder=phy_folder,
                 compute_amplitudes=True,
                 compute_pc_features=True,
                 remove_if_exists=True,
                 copy_binary=False)