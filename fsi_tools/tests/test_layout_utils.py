import pickle

import numpy as np

import lsst.geom as geom

from ..layout_utils import make_hexgrid_for_tract

# MRB I pickled this and am stuffing the bytes here
# not the best but fine
TRACTINFO_BYTES = b'\x80\x03clsst.skymap.tractInfo\nExplicitTractInfo\nq\x00)\x81q\x01}q\x02(X\x07\x00\x00\x00_radiusq\x03clsst.geom\nAngle\nq\x04G?\x8a\x962\x115\x143\x85q\x05Rq\x06X\x03\x00\x00\x00_idq\x07M\xa4\nX\x15\x00\x00\x00_patchInnerDimensionsq\x08clsst.geom\nExtent2I\nq\tM\xa0\x0fM\xa0\x0f\x86q\nRq\x0bX\x0c\x00\x00\x00_patchBorderq\x0cKdX\t\x00\x00\x00_ctrCoordq\rclsst.geom\nSpherePoint\nq\x0eh\x04G?\xed\x0e!\x12\xfdm\xd8\x85q\x0fRq\x10h\x04G\xbf\xe9W\'\xb8f\x97@\x85q\x11Rq\x12\x86q\x13Rq\x14X\x10\x00\x00\x00_vertexCoordListq\x15]q\x16(h\x0eh\x04G?\xed\xae\x9b\xa8\xad\x18\x15\x85q\x17Rq\x18h\x04G\xbf\xe9\xc5\x90IBb|\x85q\x19Rq\x1a\x86q\x1bRq\x1ch\x0eh\x04G?\xecm\xa3\x8e>\xa2\x92\x85q\x1dRq\x1eh\x04G\xbf\xe9\xc5\x90A\xe8O\xae\x85q\x1fRq \x86q!Rq"h\x0eh\x04G?\xecq\xfd\xfb/\\.\x85q#Rq$h\x04G\xbf\xe8\xe75\x9e\x8eLB\x85q%Rq&\x86q\'Rq(h\x0eh\x04G?\xed\xaaAP\x19\x12\x91\x85q)Rq*h\x04G\xbf\xe8\xe75\xa5\x84\xe7\xd2\x85q+Rq,\x86q-Rq.eX\r\x00\x00\x00_tractOverlapq/h\x04G?3\x10KW\xcf\x96\xa3\x85q0Rq1X\x0b\x00\x00\x00_numPatchesq2h\tK\x07K\x07\x86q3Rq4X\x05\x00\x00\x00_bboxq5clsst.geom\nBox2I\nq6clsst.geom\nPoint2I\nq7K\x00K\x00\x86q8Rq9h7M_mM_m\x86q:Rq;\x86q<Rq=X\x04\x00\x00\x00_wcsq>clsst.afw.geom.python.transform\nunpickleTransform\nq?clsst.afw.geom.skyWcs.skyWcs\nSkyWcs\nq@X\xc8\x16\x00\x001 SkyWcs 0  Begin FrameSet\n#   Title = "ICRS coordinates; gnomonic projection"\n#   Naxes = 2\n#   Domain = "SKY"\n#   Epoch = 2000\n#   Lbl1 = "Right ascension"\n#   Lbl2 = "Declination"\n#   System = "ICRS"\n#   Uni1 = "hh:mm:ss.s"\n#   Uni2 = "ddd:mm:ss"\n#   Dir1 = 0\n#   Bot2 = -1.5707963267948966\n#   Top2 = 1.5707963267948966\n IsA Frame\n    Nframe = 3\n#   Base = 1\n    Currnt = 3\n    Lnk2 = 1\n    Lnk3 = 2\n    Frm1 =\n       Begin Frame\n#         Title = "2-d coordinate system"\n          Naxes = 2\n          Domain = "PIXELS"\n#         Lbl1 = "Axis 1"\n#         Lbl2 = "Axis 2"\n          Ax1 =\n             Begin Axis\n             End Axis\n          Ax2 =\n             Begin Axis\n             End Axis\n       End Frame\n    Frm2 =\n       Begin Frame\n          Title = "FITS Intermediate World Coordinates"\n          Naxes = 2\n          Domain = "IWC"\n#         Lbl1 = "Axis 1"\n#         Lbl2 = "Axis 2"\n#         Uni1 = "deg"\n#         Uni2 = "deg"\n          Ax1 =\n             Begin Axis\n                Unit = "deg"\n             End Axis\n          Ax2 =\n             Begin Axis\n                Unit = "deg"\n             End Axis\n       End Frame\n    Frm3 =\n       Begin SkyFrame\n          Ident = " "\n       IsA Object\n#         Title = "ICRS coordinates; gnomonic projection"\n          Naxes = 2\n#         Domain = "SKY"\n#         Epoch = 2000\n#         Lbl1 = "Right ascension"\n#         Lbl2 = "Declination"\n          System = "ICRS"\n          AlSys = "ICRS"\n#         Uni1 = "hh:mm:ss.s"\n#         Uni2 = "ddd:mm:ss"\n#         Dir1 = 0\n#         Bot2 = -1.5707963267948966\n#         Top2 = 1.5707963267948966\n          Ax1 =\n             Begin SkyAxis\n             End SkyAxis\n          Ax2 =\n             Begin SkyAxis\n             End SkyAxis\n       IsA Frame\n          Proj = "gnomonic"\n#         SkyTol = 0.001\n          SRefIs = "Ignored"\n          SRef1 = 0.9079747553727725\n          SRef2 = -0.79188905730982384\n       End SkyFrame\n    Map2 =\n       Begin WinMap\n          Nin = 2\n          IsSimp = 1\n       IsA Mapping\n          Sft1 = 0.77772222222222231\n          Scl1 = -5.5555555555586444e-05\n          Sft2 = -0.77772222222222231\n          Scl2 = 5.5555555555586444e-05\n       End WinMap\n    Map3 =\n       Begin CmpMap\n          Nin = 2\n       IsA Mapping\n          InvA = 1\n          MapA =\n             Begin CmpMap\n                Nin = 2\n                Invert = 1\n             IsA Mapping\n                InvA = 1\n                MapA =\n                   Begin SphMap\n                      Nin = 3\n                      Nout = 2\n                      Invert = 0\n                   IsA Mapping\n                      UntRd = 1\n                      PlrLg = 0.90797475537277261\n                   End SphMap\n                MapB =\n                   Begin CmpMap\n                      Nin = 3\n                      Nout = 2\n                   IsA Mapping\n                      InvA = 1\n                      MapA =\n                         Begin MatrixMap\n                            Nin = 3\n                            Invert = 0\n                         IsA Mapping\n                            M0 = -0.43792860044783516\n                            M1 = -0.78825913613806919\n                            M2 = 0.43228008883669983\n                            M3 = -0.56098952976750427\n                            M4 = 0.61534342792855501\n                            M5 = 0.5537537477944231\n                            M6 = -0.70250216255968545\n                            M7 = 0\n                            M8 = -0.71168160830456006\n                            IM0 = -0.43792860044783516\n                            IM1 = -0.56098952976750427\n                            IM2 = -0.70250216255968545\n                            IM3 = -0.78825913613806919\n                            IM4 = 0.61534342792855501\n                            IM5 = 0\n                            IM6 = 0.43228008883669983\n                            IM7 = 0.5537537477944231\n                            IM8 = -0.71168160830456006\n                            Form = "Full"\n                         End MatrixMap\n                      MapB =\n                         Begin CmpMap\n                            Nin = 3\n                            Nout = 2\n                         IsA Mapping\n                            MapA =\n                               Begin SphMap\n                                  Nin = 3\n                                  Nout = 2\n                                  Invert = 1\n                               IsA Mapping\n                                  UntRd = 1\n                                  PlrLg = 0\n                               End SphMap\n                            MapB =\n                               Begin CmpMap\n                                  Nin = 2\n                               IsA Mapping\n                                  MapA =\n                                     Begin WcsMap\n                                        Nin = 2\n                                        Invert = 1\n                                     IsA Mapping\n                                        Type = "TAN"\n                                     End WcsMap\n                                  MapB =\n                                     Begin ZoomMap\n                                        Nin = 2\n                                        Invert = 0\n                                     IsA Mapping\n                                        Zoom = 57.295779513082323\n                                     End ZoomMap\n                               End CmpMap\n                         End CmpMap\n                   End CmpMap\n             End CmpMap\n          MapB =\n             Begin UnitMap\n                Nin = 2\n                IsSimp = 1\n             IsA Mapping\n             End UnitMap\n       End CmpMap\n End FrameSet\nqA\x86qBRqCub.'  # noqa


def test_make_hexgrid_for_tract_rng():
    ti = pickle.loads(TRACTINFO_BYTES)
    grid1 = make_hexgrid_for_tract(ti, rng=np.random.RandomState(seed=100))
    grid2 = make_hexgrid_for_tract(ti, rng=np.random.RandomState(seed=100))
    grid3 = make_hexgrid_for_tract(ti, rng=np.random.RandomState(seed=10))

    for key in grid1.dtype.names:
        assert np.array_equal(grid1[key], grid2[key])
        assert not np.array_equal(grid1[key], grid3[key])


def test_make_hexgrid_for_tract_spacing():
    ti = pickle.loads(TRACTINFO_BYTES)
    grid = make_hexgrid_for_tract(
        ti,
        spacing_pixels=128,
        rng=np.random.RandomState(seed=100)
    )

    dist = np.sqrt(
        (grid["x"][1] - grid["x"][0])**2
        + (grid["y"][1] - grid["y"][0])**2
    )
    assert np.allclose(dist, 128)


def test_make_hexgrid_for_tract_contains_points():
    ti = pickle.loads(TRACTINFO_BYTES)
    grid = make_hexgrid_for_tract(
        ti,
        spacing_pixels=128,
        rng=np.random.RandomState(seed=100)
    )

    cvx = ti.getOuterSkyPolygon()
    points = [
        geom.SpherePoint(grid["ra"][i], grid["dec"][i], geom.degrees)
        for i in range(len(grid["ra"]))
    ]
    assert np.all(np.array([cvx.contains(point.getVector()) for point in points]))

    bb = ti.getBBox()
    assert np.all(bb.contains(grid["x"], grid["y"]))
