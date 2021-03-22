import numpy as np
from lsst import geom

import tqdm
import pytest

from ..matching import Matcher, do_balrogesque_matching


def _gen_sphere_pts(n, seed):
    rng = np.random.RandomState(seed=seed)
    ra = rng.uniform(size=n) * 360
    dec = np.arcsin(rng.uniform(size=n, low=-1, high=1)) / np.pi * 180.0
    return ra, dec


@pytest.mark.parametrize('k', [1, 2, 3])
def test_matcher_knn(k):
    ra, dec = _gen_sphere_pts(50, 4543)
    mch = Matcher(ra, dec)
    d, idx = mch.query_knn(ra, dec, k=k)

    # test via brute force
    sps = [geom.SpherePoint(ra[i], dec[i], geom.degrees) for i in range(ra.shape[0])]
    idxs = np.arange(ra.shape[0])
    for i in range(ra.shape[0]):
        ds = np.array([
            sps[i].separation(sps[j]).asArcseconds() for j in range(ra.shape[0])
        ])
        inds = np.argsort(ds)

        if k != 1:
            assert np.allclose(d[i, :], ds[inds[:k]])
            assert np.array_equal(idx[i, :], idxs[inds[:k]])
        else:
            assert np.allclose(d[i], 0)
            assert np.array_equal(idx[i], i)


@pytest.mark.parametrize('k', [1, 2, 3])
def test_matcher_knn_maxrad(k):
    ra, dec = _gen_sphere_pts(50, 4543)
    mch = Matcher(ra, dec)
    d, idx = mch.query_knn(ra, dec, distance_upper_bound=5e4, k=k)

    # test via brute force
    sps = [geom.SpherePoint(ra[i], dec[i], geom.degrees) for i in range(ra.shape[0])]
    idxs = np.arange(ra.shape[0])
    for i in range(ra.shape[0]):
        ds = np.array([
            sps[i].separation(sps[j]).asArcseconds() for j in range(ra.shape[0])
        ])
        inds = np.argsort(ds)
        msk = (ds[inds] < 5e4) & (np.arange(ra.shape[0]) < k)
        s = np.sum(msk)

        if k != 1:
            assert np.allclose(d[i, :s], ds[inds[msk]])
            assert np.array_equal(idx[i, :s], idxs[inds[msk]])
        else:
            assert np.allclose(d[i], 0)
            assert np.array_equal(idx[i], i)


@pytest.mark.parametrize('k', [1, 2, 3])
def test_matcher_knn_maxrad_inf(k):
    ra, dec = _gen_sphere_pts(50, 4543)
    mch = Matcher(ra, dec)
    rap, decp = _gen_sphere_pts(50, 443)
    d, idx = mch.query_knn(rap, decp, distance_upper_bound=1, k=k)
    assert not np.any(np.isfinite(d))
    assert np.all(idx == 50)


def test_matcher_radius():
    ra, dec = _gen_sphere_pts(50, 4543)
    mch = Matcher(ra, dec)

    rap, decp = _gen_sphere_pts(100, 454)
    idx = mch.query_radius(rap, decp, 6e4)

    sps = [geom.SpherePoint(ra[i], dec[i], geom.degrees) for i in range(ra.shape[0])]
    spsp = [
        geom.SpherePoint(rap[i], decp[i], geom.degrees)
        for i in range(rap.shape[0])
    ]

    for ic in range(ra.shape[0]):
        idxc = []
        for ip in range(rap.shape[0]):
            sep = sps[ic].separation(spsp[ip]).asArcseconds()
            if sep < 6e4:
                idxc.append(ip)
        assert set(idxc) == set(idx[ic])


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
