import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import tyro
from tqdm import tqdm
from typing_extensions import Annotated
from pcdmeshing import run_block_meshing

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.scene_box import OrientedBox
from surf_leaf.post_processing.mesh_fix import process_mesh

@dataclass
class SurfLeafMesher:
    """SurfLEAF: Mesh Extraction using SuGaR Surface Level Extraction with Advancing Front Reconstruction"""

    # Input Output parameters
    load_config: Path
    """Path to the trained config YAML file."""
    output_dir: Path = Path("./mesh_exports/")
    """Path to the output directory."""

    # Sampling parameters
    total_points: int = 2_000_000
    """Total target surface samples"""
    use_masks: bool = False
    """If dataset has masks, use these to limit surface sampling regions."""
    surface_levels: List[float] = field(default_factory=lambda: [0.3])
    """Surface levels to extract"""
    return_normal: Literal[
        "analytical", "closest_gaussian", "average"
    ] = "closest_gaussian"
    """Method to return normals for the surface points."""

    # Cropbox parameters
    cropbox_pos: Optional[Tuple[float, float, float]] = None
    """Position of the cropbox center"""
    cropbox_rpy: Optional[Tuple[float, float, float]] = None
    """Roll, pitch, yaw of the cropbox in radians"""
    cropbox_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the cropbox"""

    # Meshing parameters
    voxel_size: float = 0.4
    """Voxel size for meshing"""
    margin_seam: float = 0.04
    """Margin seam for meshing"""
    margin_discard: float = 0.04
    """Margin discard for meshing"""
    max_edge_length: float = 1
    """Maximum edge length for meshing"""

    # Post-processing parameters
    postprocess_alpha_fraction: float = 0.001
    """Alpha wrapping ball size fraction"""
    postprocess_stepsmoothnum: int = 1
    """HC Laplacian smoothing steps"""
    postprocess_targetperc: float = 0.6
    """Target percentage reduction for mesh simplification"""


    def main(self):
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model

        cropbox = None

        if self.cropbox_pos is not None or self.cropbox_rpy is not None or self.cropbox_scale is not None:
            CONSOLE.print("Applying cropbox to the model.")
            if self.cropbox_pos is None:
                self.cropbox_pos = (0.0, 0.0, 0.0)
            if self.cropbox_rpy is None:
                self.cropbox_rpy = (0.0, 0.0, 0.0)
            if self.cropbox_scale is None:
                self.cropbox_scale = (1.0, 1.0, 1.0)

            cropbox = OrientedBox.from_params(
                pos=(0.0, 0.0, 0.0),
                rpy=(0.0, 0.0, 0.0),
                scale=(2.0, 2.0, 2.0),
            )

        with torch.no_grad():
            cameras: Cameras = pipeline.datamanager.train_dataset.cameras  # type: ignore
            num_frames = len(pipeline.datamanager.train_dataset)  # type: ignore
            samples_per_frame = (self.total_points + num_frames) // (num_frames)
            surface_levels_outputs = {}
            for surface_level in self.surface_levels:
                surface_levels_outputs[surface_level] = {
                    "points": torch.zeros(0, 3, device="cuda"),
                    "colors": torch.zeros(0, 3, device="cuda"),
                    "normals": torch.zeros(0, 3, device="cuda"),
                }

            for image_idx, data in tqdm(
                    enumerate(pipeline.datamanager.train_dataset),
                    desc="Computing surface levels for train images",
            ):  # type: ignore
                print(
                    "image:",
                    image_idx,
                    f"out of {len(pipeline.datamanager.train_dataset)}",
                )
                camera = cameras[image_idx: image_idx + 1].to("cuda")
                mask = None
                if "mask" in data and self.use_masks:
                    mask = data["mask"]
                frame_outputs = model.compute_level_surface_points(
                    camera=camera,
                    mask=mask,
                    num_samples=samples_per_frame,
                    surface_levels=self.surface_levels,
                    return_normal=self.return_normal,
                )  # type: ignore

                for surface_level in self.surface_levels:
                    img_surface_points = frame_outputs[surface_level]["points"]
                    img_surface_colors = frame_outputs[surface_level]["colors"]
                    img_surface_normals = frame_outputs[surface_level]["normals"]

                    # Apply oriented box crop if specified
                    if cropbox is not None:
                        inside_crop = cropbox.within(img_surface_points).squeeze()
                        if inside_crop.sum() == 0:
                            continue
                        img_surface_points = img_surface_points[inside_crop]
                        img_surface_colors = img_surface_colors[inside_crop]
                        img_surface_normals = img_surface_normals[inside_crop]

                    surface_levels_outputs[surface_level]["points"] = torch.cat(
                        [
                            surface_levels_outputs[surface_level]["points"],
                            img_surface_points,
                        ],
                        dim=0,
                    )
                    surface_levels_outputs[surface_level]["colors"] = torch.cat(
                        [
                            surface_levels_outputs[surface_level]["colors"],
                            img_surface_colors,
                        ],
                        dim=0,
                    )
                    surface_levels_outputs[surface_level]["normals"] = torch.cat(
                        [
                            surface_levels_outputs[surface_level]["normals"],
                            img_surface_normals,
                        ],
                        dim=0,
                    )

            for surface_level in self.surface_levels:
                outs = surface_levels_outputs[surface_level]
                points = outs["points"].cpu().numpy()
                colors = outs["colors"].cpu().numpy()
                normals = outs["normals"].cpu().numpy()

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                pcd.normals = o3d.utility.Vector3dVector(normals)

                before_clean_path = str(
                    self.output_dir
                    / f"before_clean_points_surface_level_{surface_level}_{self.return_normal}.ply"
                )
                CONSOLE.print(
                    "Saving unclean points to ",
                    before_clean_path,
                )
                o3d.io.write_point_cloud(
                    before_clean_path,
                    pcd,
                )

                cl, ind = pcd.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=20.0
                )
                pcd_clean = pcd.select_by_index(ind)
                clean_path = str(
                    self.output_dir
                    / f"after_clean_points_surface_level_{surface_level}_{self.return_normal}.ply"
                )
                CONSOLE.print(
                    "Saving cleaned points to ",
                    clean_path,
                )
                o3d.io.write_point_cloud(
                    clean_path,
                    pcd_clean,
                )

                CONSOLE.print("Computing Mesh... this may take a while.")

                # Run meshing
                mesh_raw, _ = run_block_meshing(
                    pcd=pcd_clean,
                    voxel_size=self.voxel_size,
                    margin_seam=self.margin_seam,
                    margin_discard=self.margin_discard,
                    num_parallel=10,
                    tmp_dir=None,
                    use_visibility=False,
                    pcd_all_path=None,
                    pcd_obs_path=None,
                    opts=dict(max_edge_length=self.max_edge_length, max_visibility=10),
                    simplify_fn=None,
                    cleanup=True
                )

                raw_mesh_path = str(
                    self.output_dir
                    / f"raw_mesh_{surface_level}_{self.return_normal}.ply"
                )
                CONSOLE.print("Saving mesh to ", raw_mesh_path)
                o3d.io.write_triangle_mesh(raw_mesh_path, mesh_raw)

                clean_mesh_path = str(
                    self.output_dir
                    / f"clean_mesh_{surface_level}_{self.return_normal}.ply"
                )
                CONSOLE.print("Applying Post-Processing to Mesh... this may take a while.")
                cleaned_meshset = process_mesh(
                    mesh_raw,
                    alpha_fraction=self.postprocess_alpha_fraction,
                    stepsmoothnum=self.postprocess_stepsmoothnum,
                    targetperc=self.postprocess_targetperc,
                    print_progress=False,
                )
                CONSOLE.print("Saving cleaned mesh to ", clean_mesh_path)
                cleaned_meshset.save_current_mesh(clean_mesh_path)


def entrypoint():
    """Entry point for the script."""
    tyro.extras.set_accent_color('bright_yellow')
    mesher = tyro.cli(SurfLeafMesher)
    mesher.main()

if __name__ == "__main__":
    entrypoint()
