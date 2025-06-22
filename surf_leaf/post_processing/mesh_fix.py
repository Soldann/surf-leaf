import argparse
import os
import pymeshlab
import open3d as o3d
import numpy as np
import time


def process_mesh(o3d_imput_mesh, alpha_fraction=0.001, stepsmoothnum=1, targetperc=0.6, print_progress=True):
    """
    Fixes a mesh by removing isolated pieces, smoothing, applying alpha wrapping,
    and simplifying the mesh.
    
    Parameters:
    o3d_imput_mesh (open3d.geometry.TriangleMesh): Open3D mesh to be processed.
    alpha_fraction (float): The size of the ball (fraction) for alpha wrapping. 
    stepsmoothnum (int): The number of times that the HC Laplacian smoothing algorithm is iterated.
    targetperc (float, 0...1): Target percentage reduction for mesh simplification.
    Returns:
    ms (pymeshlab.MeshSet): The processed MeshSet object containing the fixed mesh.
    """

    # Convert Open3D mesh to PyMeshLab mesh
    vertices = np.asarray(o3d_imput_mesh.vertices).astype(np.float64)
    faces = np.asarray(o3d_imput_mesh.triangles).astype(np.int32)
    input_mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)

    # Create a MeshSet object
    ms = pymeshlab.MeshSet()

    # load the input mesh
    start_time = time.time()
    ms.add_mesh(input_mesh)
    if print_progress:
        print(f"Input mesh loaded. Face count: {ms.current_mesh().face_number()}")

    # remove isolated pieces
    start_time = time.time()
    ms.meshing_remove_connected_component_by_diameter()
    if print_progress:
        print(f"✅ Removing isolated pieces. Elapsed time: {time.time() - start_time:.2f} seconds.")

    # smooth the output mesh
    start_time = time.time()
    ms.apply_coord_two_steps_smoothing(normalthr=20.0, stepnormalnum=6, stepfitnum=6)
    if print_progress:
        print(f"✅ Smoothing mesh. Elapsed time: {time.time() - start_time:.2f} seconds.")

    # create a new mesh object using alpha wrapping
    start_time = time.time()
    ms.generate_alpha_wrap(alpha_fraction=alpha_fraction, offset_fraction=0.000200)
    if print_progress:
        print(f"✅ Alpha wrapping. Elapsed time: {time.time() - start_time:.2f} seconds.")

    # set the second mesh (alpha wrapping mesh) as current mesh
    ms.set_current_mesh(1)

    # HC Laplacian smoothing
    start_time = time.time()
    for _ in range(stepsmoothnum):
        ms.apply_coord_hc_laplacian_smoothing()
    if print_progress:
        print(f"✅ HC Laplacian smoothing. Elapsed time: {time.time() - start_time:.2f} seconds.")

    # simplify the mesh
    start_time = time.time()
    ms.meshing_decimation_quadric_edge_collapse(targetperc=targetperc, preservetopology=False, planarquadric=True)
    if print_progress:
        print(f"✅ Mesh simplification. Elapsed time: {time.time() - start_time:.2f} seconds.")

    # save the fixed mesh to outfile
    start_time = time.time()
    return ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh fixing utility using pymeshlab.")
    parser.add_argument('--infile', type=str, required=True, help='Input mesh file (required)')
    parser.add_argument('--outfile', type=str, default=None, help='Output mesh file (default: infile + _fixed)')
    parser.add_argument('--alpha_fraction', type=float, default=0.001, help='Alpha wrapping ball size fraction (default: 0.001)')
    parser.add_argument('--stepsmoothnum', type=int, default=1, help='HC Laplacian smoothing steps (default: 1)')
    parser.add_argument('--targetperc', type=float, default=0.6, help='Target percentage for mesh simplification (default: 0.6)')
    args = parser.parse_args()

    infile = args.infile
    if args.outfile is not None:
        outfile = args.outfile
    else:
        base, ext = os.path.splitext(infile)
        outfile = f"{base}_fixed{ext}"

    start_time = time.time()
    # 1. Create or load your Open3D mesh
    o3d_mesh = o3d.io.read_triangle_mesh(infile)  # or generate it
    o3d_mesh.compute_vertex_normals()

    ms = process_mesh(
        o3d_mesh,
        alpha_fraction=args.alpha_fraction,
        stepsmoothnum=args.stepsmoothnum,
        targetperc=args.targetperc
    )
    ms.save_current_mesh(outfile)
    print(f"Mesh fixed and saved to {outfile}. Face count: {ms.current_mesh().face_number()}")

    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed:.2f} seconds")
