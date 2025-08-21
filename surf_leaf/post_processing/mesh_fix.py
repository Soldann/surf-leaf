import argparse
import os
import pymeshlab
import open3d as o3d
import numpy as np
import time


def process_mesh(o3d_input_mesh, alpha_fraction=0.001, stepsmoothnum=1, targetperc=0.6, print_progress=True, output_dir="processed_steps"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Convert Open3D mesh to PyMeshLab mesh
    vertices = np.asarray(o3d_input_mesh.vertices).astype(np.float64)
    faces = np.asarray(o3d_input_mesh.triangles).astype(np.int32)
    if o3d_input_mesh.has_vertex_colors():
        vertex_colours = np.asarray(o3d_input_mesh.vertex_colors, dtype=np.float64)
    else:
        vertex_colours = np.zeros((vertices.shape[0], 3))
    vertex_colours = np.hstack((vertex_colours, np.ones((vertex_colours.shape[0], 1))))  # RGBA

    input_mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces, v_color_matrix=vertex_colours)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(input_mesh)
    if print_progress:
        print(f"Input mesh loaded. Face count: {ms.current_mesh().face_number()}")

    ms.save_current_mesh(os.path.join(output_dir, "step0_input.obj"))

    # Step 1: Remove isolated pieces
    ms.meshing_remove_connected_component_by_diameter()
    ms.save_current_mesh(os.path.join(output_dir, "step1_removed_isolated.obj"))
    if print_progress:
        print("✅ Removed isolated pieces.")

    # Step 2: Smoothing
    ms.apply_coord_two_steps_smoothing(normalthr=20.0, stepnormalnum=6, stepfitnum=6)
    ms.save_current_mesh(os.path.join(output_dir, "step2_smoothed.obj"))
    if print_progress:
        print("✅ Smoothed mesh.")

    # Step 3: Alpha wrapping
    ms.generate_alpha_wrap(alpha_fraction=alpha_fraction, offset_fraction=0.000200)
    ms.set_current_mesh(1)
    ms.save_current_mesh(os.path.join(output_dir, "step3_alpha_wrap.obj"))
    if print_progress:
        print("✅ Alpha wrapping applied.")

    # Step 4: HC Laplacian smoothing
    for _ in range(stepsmoothnum):
        ms.apply_coord_hc_laplacian_smoothing()
    ms.save_current_mesh(os.path.join(output_dir, "step4_hc_smoothing.obj"))
    if print_progress:
        print("✅ HC Laplacian smoothing completed.")

    # Step 5: Simplify mesh
    ms.meshing_decimation_quadric_edge_collapse(
        targetperc=targetperc,
        preservetopology=False,
        planarquadric=True
    )
    ms.save_current_mesh(os.path.join(output_dir, "step5_simplified.obj"))
    if print_progress:
        print("✅ Mesh simplification complete.")

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
