# Learning Convolutional Transforms for Point Cloud Geometry Compression

<p align="center">
  <img src="figs/comparison_samples.png?raw=true" alt="Comparison samples"/>
</p>

[[Paper]](https://arxiv.org/abs/1903.08548), Supplementary Material: [[Website]](https://mauriceqch.github.io/pcc_geo_cnn_samples/) [[Data]](https://github.com/mauriceqch/pcc_geo_cnn_samples)

Authors:
[Maurice Quach](https://scholar.google.com/citations?user=atvnc2MAAAAJ),
[Giuseppe Valenzise](https://scholar.google.com/citations?user=7ftDv4gAAAAJ) and
[Frederic Dufaux](https://scholar.google.com/citations?user=ziqjbTIAAAAJ)  
Affiliation: L2S, CNRS, CentraleSupelec  
Funding: ANR ReVeRy national fund (REVERY ANR-17-CE23-0020)

## Overview

Efficient point cloud compression is fundamental to enable the deployment of virtual and mixed reality applications, since the number of points to code can range in the order of millions.
In this paper, we present a novel data-driven geometry compression method for static point clouds based on learned convolutional transforms and uniform quantization.
We perform joint optimization of both rate and distortion using a trade-off parameter.
In addition, we cast the decoding process as a binary classification of the point cloud occupancy map.
Our method outperforms the MPEG reference solution in terms of rate-distortion on the Microsoft Voxelized Upper Bodies dataset with 51.5% BDBR savings on average.
Moreover, while octree-based methods face exponential diminution of the number of points at low bitrates, our method still produces high resolution outputs even at low bitrates.

Rate-distortion data is available [here](eval). `eval_64_*` files correspond to our method and `eval_mpeg_*` files correspond to the MPEG anchor.
An analysis of this data is also available [here](src/report.ipynb).

## Versions

Python 3.6.7 and Tensorflow v1.13.1.

## Datasets

You need two datasets to reproduce our results:

* ModelNet10 manually aligned dataset: [http://modelnet.cs.princeton.edu](http://modelnet.cs.princeton.edu)
* Microsoft Voxelized Upper Bodies dataset (MVUB): [https://jpeg.org/plenodb/pc/microsoft](https://jpeg.org/plenodb/pc/microsoft)

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
This software can be retrieved at [https://github.com/mauriceqch/geo_dist](https://github.com/mauriceqch/geo_dist).
The evaluation script `eval.py` compares the original dataset, the compressed dataset and the decompressed dataset to compute distortion metrics and bitrates.
The script can also run without the decompressed dataset which means that bitrates won't be computed.

Example with original, compressed and decompressed datasets:

    python eval.py ../data/msft "**/*.ply" ../data/msft_dec_000005 ../../geo_dist/build/pc_error --decompressed_suffix .bin.ply --compressed_dir ../data/msft_bin_000005 --compressed_suffix .bin --output_file ../eval/eval_64_000005.csv

Example with only original and decompressed datasets:

    python eval.py ../data/msft "**/*.ply" ../msft_9 ../../geo_dist/build/pc_error --output_file ../eval/eval_mpeg_9.csv

## Fusing MPEG results

A modified version of the MPEG anchor code can be retrieved at [https://github.com/mauriceqch/cwi-pcl-codec](https://github.com/mauriceqch/cwi-pcl-codec).
It features directory structure preservation and adds byte count information to output CSVs.
This allows us to compute the bitrate using our pipeline.
To combine the CSVs produced by the anchor and the CSVs produced by `eval.py`, we provide a `fuse_eval_mpeg.py` script.
The script outputs a CSV file integrating results from both CSVs and computing the bitrate.

Example:

    ./evaluate_compression -i ~/data/datasets/msft -o ~/data/datasets/cwi-pcl-codec/msft_9 -q 1 -b 9 -g 1 --intra_frame_quality_csv ~/data/datasets/cwi-pcl-codec/msft_9_intra.csv --predictive_quality_csv ~/data/datasets/cwi-pcl-codec/msft9_pred.csv
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

## Samples and thumbnails

We select a subset of the dataset and provide the point clouds produced by our method and the MPEG anchor.
In addition, we also provide thumbnails for easier visualization.

Get samples with color:

    rsync -a --prune-empty-dirs --include '*/' --include 'frame0000.*' --exclude '*' pcc_geo_cnn/ pcc_geo_cnn_samples/
    rsync -a --prune-empty-dirs --include '*/' --include 'frame0000.*' --exclude '*' msft/ pcc_geo_cnn_samples/msft_samples/
    cd pcc_geo_cnn_samples/
    for i in **/*.ply.bin.ply; do j=${i#*/}; python ~/code/pcc_geo_cnn/src/map_color.py msft/${j%.bin.ply} $i $i; done

Create thumbnails:

    # Generate camera parameters
    for i in msft/**/*.ply; do echo $i; python ~/code/pcc_geo_cnn/src/pc_to_camera_params.py $i ${i}.json; done
    # Generate thumbnails
    for i in msft/**/*.ply; do echo $i; python ~/code/pcc_geo_cnn/src/pc_to_img.py $i ${i}.png ${i}.json --point_size 3; done
    for i in cwi-pcl-codec-samples/msft_5/**/*.ply; do echo $i; python ~/code/pcc_geo_cnn/src/pc_to_img.py $i ${i}.png msft/${i#*/*/}.json --point_size 33; done
    for i in cwi-pcl-codec-samples/msft_6/**/*.ply; do echo $i; python ~/code/pcc_geo_cnn/src/pc_to_img.py $i ${i}.png msft/${i#*/*/}.json --point_size 17; done
    for i in cwi-pcl-codec-samples/msft_7/**/*.ply; do echo $i; python ~/code/pcc_geo_cnn/src/pc_to_img.py $i ${i}.png msft/${i#*/*/}.json --point_size 9; done
    for i in cwi-pcl-codec-samples/msft_8/**/*.ply; do echo $i; python ~/code/pcc_geo_cnn/src/pc_to_img.py $i ${i}.png msft/${i#*/*/}.json --point_size 5; done
    for i in cwi-pcl-codec-samples/msft_9/**/*.ply; do echo $i; python ~/code/pcc_geo_cnn/src/pc_to_img.py $i ${i}.png msft/${i#*/*/}.json --point_size 3; done
    for i in msft_dec_*/**/*.ply.bin.ply; do j=${i%.bin.ply}; echo $i; python ~/code/pcc_geo_cnn/src/pc_to_img.py $i ${i}.png msft/${j#*/}.json --point_size 3; done
    # With ImageMagick, crop the whitespace on the images' sides
    for i in **/*.png; do convert $i -shave 500x0 $i; done

Produce evaluation results:

    for i in 0001 00005 00001 000005 000001; do python ~/code/pcc_geo_cnn/src/eval.py ./msft "**/*.ply" ./msft_dec_${i} ~/code/geo_dist/build/pc_error --decompressed_suffix .bin.ply --compressed_dir ./msft_bin_${i} --compressed_suffix .bin --output_file ./eval_64_${i}.csv; done
    for i in 5 6 7 8 9; do python ~/code/pcc_geo_cnn/src/eval.py ./msft "**/*.ply" ./cwi-pcl-codec-samples/msft_${i} ~/code/geo_dist/build/pc_error --output_file ./eval_mpeg_${i}.csv; done
    for i in 5 6 7 8 9; do python ~/code/pcc_geo_cnn/src/fuse_eval_mpeg.py ./eval_mpeg_${i}.csv ../cwi-pcl-codec/msft_${i}_intra.csv ./eval_mpeg_${i}_fused.csv; done
    python ~/code/pcc_geo_cnn/src/merge_csv.py ./eval_64_fused.csv -i ./eval_64_00*.csv
    python ~/code/pcc_geo_cnn/src/merge_csv.py ./eval_mpeg_fused.csv -i ./eval_mpeg_*_fused.csv


## Citation

	@inproceedings{DBLP:conf/icip/QuachVD19,
	  author    = {Maurice Quach and
		       Giuseppe Valenzise and
		       Fr{\'{e}}d{\'{e}}ric Dufaux},
	  title     = {Learning Convolutional Transforms for Lossy Point Cloud Geometry Compression},
	  booktitle = {2019 {IEEE} International Conference on Image Processing, {ICIP} 2019,
		       Taipei, Taiwan, September 22-25, 2019},
	  pages     = {4320--4324},
	  publisher = {{IEEE}},
	  year      = {2019},
	  url       = {https://doi.org/10.1109/ICIP.2019.8803413},
	  doi       = {10.1109/ICIP.2019.8803413},
	  timestamp = {Wed, 11 Dec 2019 16:30:23 +0100},
	  biburl    = {https://dblp.org/rec/conf/icip/QuachVD19.bib},
	  bibsource = {dblp computer science bibliography, https://dblp.org}
	}

