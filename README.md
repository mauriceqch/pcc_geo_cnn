# Learning Convolutional Transforms for Point Cloud Geometry Compression

<p align="center">
  <img src="figs/comparison_samples.png?raw=true" alt="Sublime's custom image" width="70%"/>
</p>

## Overview

We present a novel data-driven geometry compression method for static point clouds based on learned convolutional transforms and uniform quantization.

[[Paper]](https://arxiv.org/abs/1903.08548)

## Versions

Python 3.6.7 and Tensorflow v1.13.1.

## Datasets

You need two datasets to reproduce our results:

* ModelNet10 manually aligned dataset: http://modelnet.cs.princeton.edu
* Microsoft Voxelized Upper Bodies dataset (MVUB): https://jpeg.org/plenodb/pc/microsoft/

In our paper, we use resolutions of 512 x 512 x 512 for the MVUB dataset.

## Typical workflow

* Converting the point cloud dataset to voxel grid: `mesh_to_pc.py`
* Training the model(s) with different lambda values on this dataset: `train.py`
* If the test dataset is different from the train dataset, convert it: `mesh_to_pc.py`
* For each model, perform compression and decompression on the test dataset: `compress.py`, `decompress.py`
* For each model, compute distortion metrics and bitrates: `eval.py`
* Compute distortion metrics and bitrates for baseline methods
* For each method, merge the CSVs corresponding to each rate-distortion tradeoff: `merge_csv.py`
* Plot rate-distortion curves and compute BDBR: `plot_results.py`

## Point Cloud to Voxel Grid

We use the `mesh_to_pc.py` script to convert point cloud datasets to voxel grid datasets.
This conversion translates, scales and quantizes the point cloud coordinates so that they become integer coordinates between 0 and the target resolution.

Example:

    ./mesh_to_pc.py ../data/ModelNet40 ../data/ModelNet40_pc_64 --vg_size 64

## Training

To train a model on a particular dataset, we use the `train.py` script.

Example:

    python train.py "../data/ModelNet40_pc_64/**/*.ply" ../models/ModelNet40_pc_64_000001 --resolution 64 --lmbda 0.000001
In this sec
## Compression and Decompression

To produce compress and decompress point clouds, we use the `compress.py` and `decompress.py` scripts.
These scripts use a trained model to compress/decompress a point cloud dataset into another folder.

Example:

    python compress.py ../data/msft/ "**/*.ply" ../data/msft_bin_00001 ../models/ModelNet40_pc_64_lmbda_00001 --resolution 512
    python decompress.py ../data/msft_bin_0001/ "**/*.ply.bin" ../data/msft_dec_00001/ ../models/ModelNet40_pc_64_lmbda_00001

## Evaluation

The evaluation part depends on `geo_dist` which performs geometric distortion computation for point clouds.
This software can be retrieved at https://github.com/mauriceqch/geo_dist.
The evaluation script `eval.py` compares the original dataset, the compressed dataset and the decompressed dataset to compute distortion metrics and bitrates.
The script can also run without the decompressed dataset which means that bitrates won't be computed.

Example with original, compressed and decompressed datasets:

    python eval.py ../data/msft "**/*.ply" ../data/msft_dec_000005 ../../geo_dist/build/pc_error --decompressed_suffix .bin.ply --compressed_dir ../data/msft_bin_000005 --compressed_suffix .bin --output_file ../eval/eval_64_000005.csv

Example with only original and decompressed datasets:

    python eval.py ../data/msft "**/*.ply" ../msft_9 ../../geo_dist/build/pc_error --output_file ../eval/eval_mpeg_9.csv

## Fusing MPEG results

A modified version of the MPEG anchor code can be retrieved at https://github.com/mauriceqch/cwi-pcl-codec.
It features directory structure preservation and adds byte count information to output CSVs.
This allows us to compute the bitrate using our pipeline.
To combine the CSVs produced by the anchor and the CSVs produced by `eval.py`, we provide a `fuse_eval_mpeg.py` script.
The script outputs a CSV file integrating results from both CSVs and computing the bitrate.

Example:

    python fuse_eval_mpeg.py ../eval/eval_mpeg_9.csv ~/code/cwi-pcl-codec/build/apps/evaluate_compression/msft_9_intra.csv ../eval/eval_mpeg_9_fused.csv

## Merging evaluation results

To merge evaluation results obtained with different rate-distortion tradeoffs, we provide a `merge_csv.py` utility script that merges multiple CSV files and adds a column named `csv_file` corresponding to the csv path.

Examples:

    python merge_csv.py ../eval/eval_64_fused.csv -i ../eval/eval_64_00*.csv
    python merge_csv.py ./eval_mpeg_fused.csv -i ../eval/eval_mpeg_*_fused.csv

## Plotting results

To plot rate-distortion curves, we use the `plot_results.py` script.
This script averages rate-distortion points over time for each sequence of the MVUB dataset.
Then, it computes Bjontegaard-delta bitrates (BDBR) for each individual sequence and for all sequences.

Example:
  
    python plot_results.py ../figs/rd -i ../eval/eval_64_fused.csv ../eval/eval_mpeg_fused.csv -t Proposed Anchor

## Mapping color

We provide the `map_color.py` script which maps colors from a point cloud to another on a nearest-neighbor basis.
This is typically used to add colors to decompressed point clouds for visualization.
It can also be used to replace compressed colors by ground truth colors for a point cloud.

**Note**: the MPEG codec adds some metadata at the end of decompressed files which causes some parsing errors. As a result, it may be necessary to manually edit the file to perform color mapping.

Example:

    python map_color.py ../data/msft/phil9/ply/frame0000.ply ./phil9_frame0000_6.ply ./phil9_frame0000_6_color.ply

## Citation

	@misc{quach2019learning,
	    title={Learning Convolutional Transforms for Lossy Point Cloud Geometry Compression},
	    author={Maurice Quach and Giuseppe Valenzise and Frederic Dufaux},
	    year={2019},
	    eprint={1903.08548},
	    archivePrefix={arXiv},
	    primaryClass={cs.CV}
	}

