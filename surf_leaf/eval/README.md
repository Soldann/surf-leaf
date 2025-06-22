# Surf-LEAF/eval

This subfolder contains the evaluation script used to evaluate the final quality of the meshes output by Surf-LEAF.

This tool will compare a given input mesh to the ground-truth in a Nerfstudio format RGB-D dataset. If no `gt-pointcloud.ply` file is found in the Nerfstudio dataset, one will be generated from the GT depth images.
The tool will then perform global registration using RANSAC and ICP to align the pointclouds before calculating the Chamfer Distance between them.


### Installation
```
pip install open3d
```

### Usage
```
python eval_mesh.py --input-mesh <path_to_mesh> --dataset <path_to_nerfstudio_dataset> --scaling_factor <scaling_factor>
```
The scaling factor mentioned here is applied to the ground-truth before calculation to account for NerfStudio normalizing the distances of its output, this has to be manually configured per scene (the default value given is for the Tartan Air Abandoned Factory P001 scene).

For example:
```
python eval_mesh.py --input-mesh output_mesh.ply --dataset /data/tartanair/abandoned_factory/P001 --scaling_factor 0.04
``` 
