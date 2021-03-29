import numpy as np
from smatch.matcher import Matcher


def do_balrogesque_matching(
    fsi_det_cat, orig_det_cat, fsi_truth_cat, flux_column,
    fsi_match_radius=0.5, blended_fsi_match_radius=1.5,
):
    """Do balrog-esque matching similar to https://arxiv.org/abs/2012.12825
    and return the match flag.

    To replicate the cuts from the paper above, cut on `match_flag < 2` with
    `fsi_match_radius=0.5` and `blended_fsi_match_radius=1.5`.

    Parameters
    ----------
    fsi_det_cat : structured array-like
        Catalog of all detections in the FSI-processed images. Should have columns
        'ra' (degrees), 'dec' (degrees), and `mag_column`.
    orig_det_cat : structured array-like
        Catalog of all original detections in the image used for FSI processing.
        Should have columns 'ra' (degrees), 'dec' (degrees), and `mag_column`.
    fsi_truth_cat : structured array-like
        Truth catalog of all injections. Should have columns 'ra' (degrees)
        and 'dec' (degrees).
    flux_column : str
        The column of a flux measure to be used to disambiguate mixtures of
        injected and original sources.
    fsi_match_radius : float, optional
        The radius in arcseconds for matching the truth objects in `fsi_truth_cat`
        to the objects detected in `fsi_det_cat`. Default is 0.5 arcseconds.
    blended_fsi_match_radius : float, optional
        The radius in arcseconds for matching any object in the `fsi_det_cat` that
        matches an object in `fsi_truth_cat` to the original detections in
        `orig_det_cat`. Default is 1.5 arcseconds.

    Returns
    -------
    match_flag : array-like of int, size of `fsi_det_cat`
        The match flag. It has possible values
            0: indicates the object in `fsi_det_cat` matched cleanly to one
               truth object from `fsi_truth_cat` and no previous detections in
               `orig_det_cat`.
            1: indicates the object in `fsi_det_cat` matched to one truth object
               from `fsi_truth_cat` and matched to a dimmer previous detection
               in `orig_det_cat`.
            2: indicates the object in `fsi_det_cat` matched to one truth object
               from `fsi_truth_cat` and matched to a brighter previous detection
               in `orig_det_cat`.
            3: indicates the object in `fsi_det_cat` did not match to any injected
               object in `fsi_det_cat`
    match_index : array-like of int
        Index of the match in the `fsi_truth_cat`. A value of -1 indicates no match.
    """
    match_flag = np.zeros(fsi_det_cat.shape[0], dtype=np.int32)
    match_index = np.zeros(fsi_det_cat.shape[0], dtype=np.int64) + -1

    with Matcher(fsi_truth_cat["ra"], fsi_truth_cat["dec"]) as mch:
        idx = mch.query_knn(
            fsi_det_cat["ra"],
            fsi_det_cat["dec"],
            k=1,
            distance_upper_bound=fsi_match_radius/3600,
        )
        msk = idx >= fsi_truth_cat["ra"].shape[0]
        match_flag[msk] = 3
        match_index[~msk] = idx[~msk]

    with Matcher(fsi_det_cat["ra"], fsi_det_cat["dec"]) as mch:
        oidx = mch.query_radius(
            orig_det_cat["ra"],
            orig_det_cat["dec"],
            blended_fsi_match_radius/3600,
        )
        for i, idx in enumerate(oidx):
            if len(idx) == 0 or match_flag[i] != 0:
                continue
            else:
                omag = orig_det_cat[flux_column][idx]
                cmag = fsi_det_cat[flux_column][i]
                if np.any(omag > cmag):
                    match_flag[i] = 2
                else:
                    match_flag[i] = 1

    return match_flag, match_index
