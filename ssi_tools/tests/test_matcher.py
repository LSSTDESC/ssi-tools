import numpy as np
from lsst import geom

import tqdm

from ..matching import do_balrogesque_matching


def _make_balrogesque_cat(n, seed):
    rng = np.random.RandomState(seed=seed)
    data = np.zeros(n, dtype=[("ra", "f8"), ("dec", "f8"), ("flux", "f8")])
    data["ra"] = rng.uniform(size=n) * 1/60
    data["dec"] = np.arcsin(rng.uniform(size=n, low=-1/60, high=1/60)) / np.pi * 180.0
    data["flux"] = rng.uniform(size=n, low=1, high=10)
    return data


def test_do_balrogesque_matching():
    fsi_det_cat = _make_balrogesque_cat(100, 34489)
    fsi_truth_cat = _make_balrogesque_cat(100000, 3448)
    orig_det_cat = _make_balrogesque_cat(10000, 43)

    match_flag, match_index = do_balrogesque_matching(
        fsi_det_cat, orig_det_cat, fsi_truth_cat, "flux",
    )

    # make sure we get all types of matches in our test
    assert set(np.unique(match_flag)) == set([0, 1, 2, 3])
    assert match_flag.shape == (100,)
    assert match_index.shape == (100,)

    # check that unmatched have the right index
    msk = match_flag == 3
    assert np.all(match_index[msk] == -1)
    assert np.all(match_index[~msk] != -1)

    fsi_det_sps = [
        geom.SpherePoint(fsi_det_cat["ra"][i], fsi_det_cat["dec"][i], geom.degrees)
        for i in range(fsi_det_cat.shape[0])
    ]
    fsi_truth_sps = [
        geom.SpherePoint(fsi_truth_cat["ra"][i], fsi_truth_cat["dec"][i], geom.degrees)
        for i in range(fsi_truth_cat.shape[0])
    ]
    orig_sps = [
        geom.SpherePoint(orig_det_cat["ra"][i], orig_det_cat["dec"][i], geom.degrees)
        for i in range(orig_det_cat.shape[0])
    ]

    print("", flush=True)
    for idet in tqdm.trange(fsi_det_cat.shape[0], leave=False):
        dt = np.array([
            fsi_det_sps[idet].separation(fsi_truth_sps[j]).asArcseconds()
            for j in range(fsi_truth_cat.shape[0])
        ])
        do = np.array([
            fsi_det_sps[idet].separation(orig_sps[j]).asArcseconds()
            for j in range(orig_det_cat.shape[0])
        ])
        if match_flag[idet] == 3:
            # did match anything in the truth catalog so all > 0.5
            assert np.all(dt > 0.5)
        elif match_flag[idet] == 0:
            # matched to truth catalog and no orig objects
            assert np.any(dt <= 0.5)
            assert np.argmin(dt) == match_index[idet]
            assert np.all(do > 1.5)
        else:
            # match to truth and orig catalogs
            assert np.any(dt <= 0.5)
            assert np.argmin(dt) == match_index[idet]
            msk = do <= 1.5
            assert np.any(msk)
            if match_flag[idet] == 1:
                # for flag == 1, it is the brightest of the orig things
                assert np.all(
                    orig_det_cat["flux"][msk] < fsi_det_cat["flux"][idet]
                )
            elif match_flag[idet] == 2:
                # for flag == 2, it is dimmer than at least one orig thing
                assert np.any(
                    orig_det_cat["flux"][msk] >= fsi_det_cat["flux"][idet]
                )
