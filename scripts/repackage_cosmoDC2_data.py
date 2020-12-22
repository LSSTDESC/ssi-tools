"""This script repackages a subset of the DC2 data into a file that
corresponds to 4 square degrees of objects brighter than i = 28.

The resulting file is ~500 MB which is a useful sizxe in that it is not too big
but not too small either.
"""
import os
import tqdm
import numpy as np
import fitsio

import GCRCatalogs

# numpy structured array dtype for the outputs
dtype = [
    ("raJ2000", "f8"),
    ("decJ2000", "f8"),
    ("redshift", "f4"),
    ("sourceType", "U6"),
    ("umagVar", "f4"),
    ("gmagVar", "f4"),
    ("rmagVar", "f4"),
    ("imagVar", "f4"),
    ("zmagVar", "f4"),
    ("ymagVar", "f4"),
    ("DiskHalfLightRadius", "f4"),
    ("disk_n", "f4"),
    ("a_d", "f4"),
    ("b_d", "f4"),
    ("pa_disk", "f4"),
    ("BulgeHalfLightRadius", "f4"),
    ("bulge_n", "f4"),
    ("a_b", "f4"),
    ("b_b", "f4"),
    ("pa_bulge", "f4"),
]

# these are mappings from the column names expected by the fsi code and
# the DC2 column names
dc2_name_map = {
    "raJ2000": "ra",
    "decJ2000": "dec",
    "redshift": "redshift",
    "umagVar": "mag_true_u_lsst",
    "gmagVar": "mag_true_g_lsst",
    "rmagVar": "mag_true_r_lsst",
    "imagVar": "mag_true_i_lsst",
    "zmagVar": "mag_true_z_lsst",
    "ymagVar": "mag_true_Y_lsst",
    "DiskHalfLightRadius": "morphology/diskHalfLightRadiusArcsec",
    "a_d": 'morphology/diskMajorAxisArcsec',
    "b_d": 'morphology/diskMinorAxisArcsec',
    "disk_n": 'morphology/diskSersicIndex',
    "pa_disk": 'morphology/positionAngle',
    "BulgeHalfLightRadius": "morphology/spheroidHalfLightRadiusArcsec",
    "a_b": 'morphology/spheroidMajorAxisArcsec',
    "b_b": 'morphology/spheroidMinorAxisArcsec',
    "bulge_n": 'morphology/spheroidSersicIndex',
}

cat = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_small")

# we keep a catalog corresponding to objects from 4 square degrees
# of everything brighter than i = 28
rng = np.random.RandomState(seed=3463)
frac = 4.0/cat.sky_area

# we have to make the random selection after the i mag cut, so we do that column
# first
col_d = cat.get_quantities(dc2_name_map["imagVar"])[dc2_name_map["imagVar"]]
imsk = (col_d <= 28.0)
tot = int(np.sum(imsk))
num = int(tot * frac)
inds = rng.choice(tot, size=num)
d = np.zeros(num, dtype=dtype)

for col, dc2_col in tqdm.tqdm(dc2_name_map.items()):
    d[col] = cat.get_quantities(dc2_col)[dc2_col][imsk][inds]
d["sourceType"] = "galaxy"

os.makedirs(
    "/global/cfs/cdirs/lsst/groups/fake-source-injection/DC2/catalogs/",
    exist_ok=True,
)
fitsio.write(
    "/global/cfs/cdirs/lsst/groups/fake-source-injection/DC2/catalogs/"
    "cosmoDC2_v1.1.4_small_fsi_catalog.fits",
    d,
    clobber=True,
)
