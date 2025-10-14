import argparse
import os
import pymeshlab
import open3d as o3d
import numpy as np
import time


def process_mesh(
        o3d_input_mesh,
        alpha_fraction=0.001,
        offset_fraction=0.000200,
        stepsmoothnum=1,
        targetperc=0.6,
        print_progress=True,
        output_dir="processed_steps",
        save_intermediate=True
):
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

    i = 0

    if save_intermediate:
        ms.save_current_mesh(os.path.join(output_dir, f"step{i}_input.ply"))

    i += 1

    # Step 1: Remove isolated pieces
    ms.meshing_remove_connected_component_by_diameter()
    if save_intermediate:
       ms.save_current_mesh(os.path.join(output_dir, f"step{i}_removed_isolated.ply"))
    if print_progress:
       print("✅ Removed isolated pieces.")

    i += 1

    # Step 2: Smoothing
    ms.apply_coord_two_steps_smoothing(normalthr=20.0, stepnormalnum=6, stepfitnum=6)
    if save_intermediate:
       ms.save_current_mesh(os.path.join(output_dir, f"step{i}_smoothed.ply"))
    if print_progress:
       print("✅ Smoothed mesh.")

    i += 1

    # Step 3: Alpha wrapping
    ms.generate_alpha_wrap(alpha_fraction=alpha_fraction, offset_fraction=offset_fraction)
    ms.set_current_mesh(1)
    if save_intermediate:
        ms.save_current_mesh(os.path.join(output_dir, f"step{i}_alpha_wrap.ply"))
    if print_progress:
        print("✅ Alpha wrapping applied.")

    i += 1

    # Step 4: Simplify mesh
    ms.meshing_decimation_quadric_edge_collapse(
        targetperc=targetperc,
        preservetopology=False,
        planarquadric=True
    )
    if save_intermediate:
        ms.save_current_mesh(os.path.join(output_dir, f"step{i}_simplified.ply"))
    if print_progress:
        print("✅ Mesh simplification complete.")

    return ms


if __name__ == "__main__":
    #python3 mesh_fix.py --infile /home/jannick/Projects/GaussianSplatting/nsWorkspace/mesh_pipeline_testing/raw_mesh_0.3_closest_gaussian.ply --outdir /home/jannick/Projects/GaussianSplatting/nsWorkspace/mesh_pipeline_testing/og
    parser = argparse.ArgumentParser(description="Mesh fixing utility using pymeshlab.")
    parser.add_argument('--infile', type=str, required=True, help='Input mesh file (required)')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory for processed mesh files (default: infile + _fixed)')
    parser.add_argument('--alpha_fraction', type=float, default=0.001, help='Alpha wrapping ball size fraction (default: 0.001)')
    parser.add_argument('--stepsmoothnum', type=int, default=1, help='HC Laplacian smoothing steps (default: 1)')
    parser.add_argument('--targetperc', type=float, default=0.6, help='Target percentage for mesh simplification (default: 0.6)')
    args = parser.parse_args()

    infile = args.infile
    if args.outdir is not None:
        outdir = args.outdir
    else:
        base, _ = os.path.splitext(infile)
        outdir = f"{base}_fixed"
    os.makedirs(outdir, exist_ok=True)

    start_time = time.time()
    o3d_mesh = o3d.io.read_triangle_mesh(infile)
    o3d_mesh.compute_vertex_normals()

    ms = process_mesh(
        o3d_mesh,
        alpha_fraction=args.alpha_fraction,
        stepsmoothnum=args.stepsmoothnum,
        targetperc=args.targetperc,
        output_dir=outdir,
        save_intermediate=True
    )
    # Save the final mesh as step5_simplified.ply in the output directory
    final_mesh_path = os.path.join(outdir, "step5_simplified.ply")
    ms.save_current_mesh(final_mesh_path)
    print(f"Mesh fixed and saved to {final_mesh_path}. Face count: {ms.current_mesh().face_number()}")

    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed:.2f} seconds")
