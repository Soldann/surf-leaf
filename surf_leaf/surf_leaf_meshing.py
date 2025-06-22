import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

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

@dataclass
class SurfLeafMesher:
    load_config: Path
    """Path to the trained config YAML file."""
    output_dir: Path = Path("./mesh_exports/")
    """Path to the output directory."""

    total_points: int = 2_000_000
    """Total target surface samples"""
    use_masks: bool = False
    """If dataset has masks, use these to limit surface sampling regions."""
    surface_levels: Tuple[float, float, float] = (0.3)
    """Surface levels to extract"""
    return_normal: Literal[
        "analytical", "closest_gaussian", "average"
    ] = "closest_gaussian"

    voxel_size: float = 0.4
    """Voxel size for meshing"""
    margin_seam: float = 0.04
    """Margin seam for meshing"""
    margin_discard: float = 0.04
    """Margin discard for meshing"""
    max_edge_length: float = 1

    def main(self):
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model

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


def entrypoint():
    """Entry point for the script."""
    tyro.extras.set_accent_color('bright_blue')
    mesher = tyro.cli(SurfLeafMesher)
    mesher.main()

if __name__ == "__main__":
    entrypoint()
