from dataclasses import dataclass
from typing import Tuple, Type
import os
import numpy as np

@dataclass
class Ice_output_config:
    name: str
    path: str
    nslices: int
    nfiles: int
    nechoes: int
    nchannels: int
    dtype: Type
    size: Tuple[int, int]

def Ice_output_config_factory(name: str, path: str, nslices: int, nfiles: int, nechoes: int = 1, nchannels: int = 1, dtype: Type = np.int16):
    """
    Ice_output_config factory function
    
    `name` can be a unique part of the full file name
    `nfiles` is the number of .ima files in the folder
    
    Example:
    cfg = Ice_output_config_factory("Aspire_P", "/path/to/ima_folder", 120, 720)
    volume = read_volume(cfg)
    """
    #TODO automatically detect number of files in folder
    return Ice_output_config(name, path, nslices, nfiles, nechoes, nchannels, dtype, getsize(path))

def get_setting(T, lines, setting, offset=3, default=0):
    for i, line in enumerate(lines):
        if setting in line:
            try:
                return T(lines[i + offset])
            except:
                return default
    return default

def read_volume(cfg):
    """
    read_volume(cfg)
    
    Example:
    cfg = Ice_output_config_factory("Aspire_P", "/path/to/ima_folder", 120, 720)
    volume = read_volume(cfg)
    """
    volume = create_volume(cfg)

    for i in range(1, cfg.nfiles + 1):
        num = f"{i:05d}"
        imahead = os.path.join(cfg.path, f"MiniHead_ima_{num}.IceHead")
        file = os.path.join(cfg.path, f"WriteToFile_{num}.ima")
        
        with open(imahead, 'r') as f:
            imahead_content = f.read()
        
        if cfg.name in imahead_content:
            vol = np.fromfile(file, dtype=cfg.dtype).reshape(cfg.size)
            with open(imahead, 'r') as f:
                lines = f.readlines()
            eco = get_setting(int, lines, "EchoNumber", default=1)
            slc = getslice(lines)
            rescale_slope = get_setting(float, lines, "RescaleSlope", offset=4, default=1)
            rescale_intercept = get_setting(float, lines, "RescaleIntercept", offset=4, default=0)
            volume[:,:,slc,eco] = vol * rescale_slope + rescale_intercept

    return volume

def getslice(lines):
    slc = get_setting(int, lines, "Actual3DImaPartNumber", default=None)
    if slc is None:
        slc = get_setting(int, lines, "AnatomicalSliceNo")
    return slc

def getsize(path):
    with open(os.path.join(path, "MiniHead_ima_00001.IceHead"), 'r') as f:
        lines = f.readlines()
    return (get_setting(int, lines, "NoOfCols"), get_setting(int, lines, "NoOfRows"))

def create_volume(cfg):
    return np.zeros((cfg.size[0], cfg.size[1], cfg.nslices, cfg.nechoes), dtype=np.float32)