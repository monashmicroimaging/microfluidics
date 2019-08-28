from collections import defaultdict
from bidict import bidict
import pathlib
import re

def get_channel_dict_from_name(exp_df, filename):
    """Generates a channel dict from a file name string or path
    
    Parameters
    ----------
    exp_df : pandas Dataframe
        This is the pandas dataframe generated from Jens Excel table decribing her experiments
    filename : tif file that is being analyzed
        From the tif file name or path the basename of the Experiment will be extracted and used as
        a key in the Filename column of the table

    For the given experiment returns a default dictionary that maps channel names such 
    as "dapi", "brightfield", "red", "green" to their respective channel numbers or -1 (through defaultdict 
    value) if a channel is not present for a given experiment.
    """
    
    translate_names = defaultdict(lambda: None, {"WL":"brightfield", "B":"dapi", "R":"red", "G":"green"})

    name = str(pathlib.Path(filename).name)
    regex = r'(?P<name>.*)_s\d+.tif'
    try:
        m = re.match(regex, name)
        basename=m.groupdict()["name"]
    except:
        print(f"error extracting basename from {filename}")
        return None
    try:
        tmp = exp_df[exp_df.Filename == basename].iloc[0]
    except:
        print(f"no entry {basename} found in table")
        return None
    chdict = bidict()
    for i in range(4):
        col = "Channel "+str(i)
        chname = translate_names[tmp[col]]
        if chname:
            chdict[i] = chname 
    dd = defaultdict(lambda: -1, chdict.inverse)
    return dd
    