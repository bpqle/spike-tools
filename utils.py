import numpy as np
import pandas as pd
from probeinterface import Probe, combine_probes

def create_probe(csv_file):
    ProbeCFG = pd.read_csv(csv_file)
    probe = []
    for s in ProbeCFG.shaft.unique():
        shankcfg = ProbeCFG[ProbeCFG.shaft==s].sort_values('channel')
        shank = Probe(ndim=2)
        x_coords = shankcfg.X.values
        y_coords = shankcfg.Z.values
        shank.set_contacts(
            positions = list(zip(x_coords, y_coords)),
            shapes = 'circle',
            shape_params = {'radius': 5},
        )
        shank.create_auto_shape()
        probe.append(shank)
    
    probe = combine_probes(probe)
    probe.set_contact_ids(np.arange(len(ProbeCFG)))
    probe.set_device_channel_indices(np.arange(len(ProbeCFG)))
    return probe