import numpy as np
import trimesh
import alphashape
import pyvista
import pyacvd
import pymeshfix

def pyvista_to_mesh(mesh):
    v = mesh.points
    f = mesh.faces.reshape(-1, 4)[:, 1:4]
    return v, f


def mesh_to_pyvista(v, f):
    n, m = f.shape
    threes = np.full((n, 1), 3)
    face_arr = np.hstack((threes, f)).flatten()
    return pyvista.PolyData(v, face_arr)

def mesh_cleanup(mesh_raw, target_verts=2500):

    v, f = mesh_raw.vertices, mesh_raw.faces
    # this can give a depreciation warning but it is fine
    mesh = mesh_to_pyvista(v, f)

    # target mesh resolution
    # target_verts = 2500

    clus = pyacvd.Clustering(mesh)
    clus.subdivide(2)
    clus.cluster(target_verts)

    remesh = clus.create_mesh()

    v2, f2 = pyvista_to_mesh(remesh)

    # pymeshfix is often necessary here to get rid of non-manifold vertices
    v2, f2 = pymeshfix.clean_from_arrays(v2, f2)

    mesh_out = trimesh.Trimesh(vertices=v2, faces=f2)

    return mesh_out

def fit_cell_mesh(xyz_fin, alpha=5, n_faces=500, smoothing_strength=3):

    # normalize for alphshape fitting
    mp = np.min(xyz_fin)
    points = xyz_fin - mp
    mmp = np.max(points)
    points = points / mmp

    meshing_error_flag = False
    try:
        raw_hull = alphashape.alphashape(points, alpha)
    except:
        meshing_error_flag = True

    if not meshing_error_flag:
        # copy
        hull02_cc = raw_hull.copy()

        # keep only largest component
        hull02_cc = hull02_cc.split(only_watertight=False)
        hull02_sm = max(hull02_cc, key=lambda m: m.area)

        hull02_sm = trimesh.smoothing.filter_laplacian(hull02_sm, iterations=2)

        # fill holes
        hull02_sm = mesh_cleanup(hull02_sm)

        # smooth
        hull02_sm = trimesh.smoothing.filter_laplacian(hull02_sm, iterations=smoothing_strength)

        # resample
        n_faces = np.min([n_faces, hull02_sm.faces.shape[0]-1])
        hull02_rs = hull02_sm.simplify_quadric_decimation(face_count=n_faces)
        hull02_rs = hull02_rs.split(only_watertight=False)
        hull02_rs = max(hull02_rs, key=lambda m: m.area)
        hull02_rs.fill_holes()
        hull02_rs.fix_normals()

        vt = hull02_rs.vertices
        vt = vt * mmp
        vt = vt + mp
        hull02_rs.vertices = vt

        # check
        wt_flag = hull02_rs.is_watertight

        return hull02_rs, raw_hull, wt_flag

    else:
        return None, None, False