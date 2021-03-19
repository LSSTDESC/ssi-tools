from scipy.spatial import cKDTree
import healpy as hp
import numpy as np

ARCSEC2RAD = 1/60/60/180*np.pi


class Matcher(object):
    """A class to find the nearest nbrs of a set of points.

    Idea and bits of code using cKDTree from Alex Drlica-Wagner w/ help and tips
    from Eli Rykoff.

    Parameters
    ----------
    ra : array-like
        The right ascension in degrees.
    dec : array-like
        The declination in degrees.
    **kwargs : extra keyword arguments
        Any exta keyword arguments are passed to `cKDTree`
    """
    def __init__(self, ra, dec, **kwargs):
        self.ra = ra
        self.dec = dec
        self.coords = hp.rotator.dir2vec(ra, dec, lonlat=True).T
        self.tree = cKDTree(self.coords, compact_nodes=False, balanced_tree=False)

    def query_knn(self, ra, dec, k=1, distance_upper_bound=None, **kwargs):
        """Find the `k` nearest nbrs of each point in (ra, dec) in the points
        held by the matcher.

        Parameters
        ----------
        ra : array-like
            The right ascension in degrees.
        dec : array-like
            The declination in degrees.
        k : int
            The number of nearest nbrs to find.
        distance_upper_bound : float, optional
            The maximum allowed distance in arcseconds for a nearest nbr. Default of
            None results in no upper bound on the distance.
        **kwargs : extra keyword arguments
            These are any keyword arguments accepted by `cKDTree.query`. Ones of
            interest may be `workers` for parallel processing and `eps` for approximate
            nearest nbr searches.

        Returns
        -------
        d : array-like, float
            Array of distances in arcseconds. Same shape as input array with axis
            of dimension `k` added to the end. If `k=1`, then this last dimension
            is squeezed out.
        idx : array-like, int
            Array of indices. Same shape as input array with axis
            of dimension `k` added to the end. If `k=1`, then this last dimension
            is squeezed out.
        """
        if "p" in kwargs:
            raise RuntimeError("You must use a Euclidean metric with the `Matcher`!")

        if distance_upper_bound is not None:
            maxd = 2 * np.sin(ARCSEC2RAD * distance_upper_bound / 2)
        else:
            maxd = np.inf

        x = hp.rotator.dir2vec(ra, dec, lonlat=True).T
        d, idx = self.tree.query(x, k=k, p=2, distance_upper_bound=maxd, **kwargs)
        d /= 2
        np.arcsin(d, where=np.isfinite(d), out=d)
        d *= (2/ARCSEC2RAD)

        return d, idx

    def query_radius(self, ra, dec, radius, eps=0):
        """Find all points in (ra, dec) that are within `radius` of a point in the
        matcher.

        Parameters
        ----------
        ra : array-like
            The right ascension in degrees.
        dec : array-like
            The declination in degrees.
        radius : float
            Maximum radius in arcseconds.
        eps : float, optional
            If non-zero, the set of returned points are correct to within
            a fractional precision of `eps` of being closer than `radius`.

        Returns
        -------
        idx : list of list of ints
            For each point in the matcher, a list of indices into `ra` and `dec`
            of points within `radius`.
        """
        # The second tree in the match does not need to be balanced, and
        # turning this off yields significantly faster runtime.
        # - Eli Rykoff
        coords = hp.rotator.dir2vec(ra, dec, lonlat=True).T
        tree = cKDTree(coords, compact_nodes=False, balanced_tree=False)
        d = 2.0*np.sin(ARCSEC2RAD * radius/2.0)
        idx = self.tree.query_ball_tree(tree, d, eps=eps)
        return idx

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # clear refs to data so that GC can do its thing
        del self.tree
        del self.ra
        del self.dec
        del self.coords


def do_balrogesque_matching(
    fsi_det_cat, orig_det_cat, fsi_truth_cat, mag_column,
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
    """
    pass
