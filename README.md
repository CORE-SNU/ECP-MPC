# ECP-MPC: Model Predictive Control with Egocentric Conformal Prediction

This repository provides the official Python implementation for the paper [Egocentric Conformal Prediction for Safe and Efficient Navigation in Dynamic Cluttered Environments](https://arxiv.org/abs/2504.00447).

## Installation

Our implementation has minimal dependencies. We have tested the code successfully with the following setup:

* python 3.9
* numpy 2.0.2
* scipy 1.13.1
* scikit-learn 1.6.1
* opencv-python 4.11.0
* matplotlib 3.9.4

## Datasets & Prediction Models

For evaluation and comparison, we use the **ETH** and **UCY** pedestrian datasets. Predictions are generated using [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus), which provides straightforward instructions for training models on these datasets.

To simplify usage and avoid requiring users to train their own models, we provide precomputed predictions obtained from pretrained Trajectron++ models. These predictions are stored as Python dictionaries in `./ecp/predictions`.

## Visualization Setup

To visualize prediction and control results, we transform all relevant data (originally in world coordinates) into image coordinates, then overlay the transformed data onto video frames.

To synchronize pedestrian movements with visualization results, execute the following commands to download and parse raw videos (used for pedestrian annotations) into frames at 2.5Hz:

```bash
bash assets/download_videos.sh
python assets/video_parser.py [directory-to-save-parsed-frames]
```

Parsed frames are saved in `[directory-to-save-parsed-frames]` for visualization. This approach is ideal for detailed frame-by-frame analysis but requires a large storage (\~3GB). Please ensure that sufficient storage space is available before proceeding.

### Homography Matrices

Visualization requires projecting data onto image frames using homography matrices. We provide these matrices in `assets/homographies`, used primarily by `ecp/visualization_utils.py`.

* **ETH** provides official homography matrices. We have permuted the first two columns (i.e., swapped x and y pixel coordinates) for easier visualization.
* **UCY** does not seem to publicly provide homography matrices. Therefore, we include approximate matrices computed via RANSAC. Note that these matrices are imprecise, so users should verify their suitability if they want to use these in other projects.

## Evaluation

We include baseline implementations for comparison:

* [ACI-based MPC](https://proceedings.mlr.press/v211/dixit23a/dixit23a.pdf), labeled as `acp-mpc`,
* [Conformal controller](https://conformal-decision.github.io/), labeled as `cc`.

To evaluate all methods across all datasets, simply run:

```bash
bash ./ecp/run.sh
```

To visualize the results, run:

```bash
bash ./ecp/run.sh --visualize --asset-dir [directory-to-frames]
```

where `[directory-to-frames]` is the path to the parsed video frames generated during the [Visualization Setup](#visualization-setup).

### Running Individual Methods or Datasets

To evaluate specific methods and datasets individually, use:

```bash
python evaluate_controller.py --dataset [dataset-name] --controller [controller-name] --visualize --asset_dir [directory-to-frames]
```

* `[dataset-name]`: `zara1`, `zara2`, `univ`, `eth`, or `hotel`
* `[controller-name]`: `ecp-mpc`, `acp-mpc`, or `cc`

Again, the visualization options (`--visualize` and `--asset_dir`) are optional. 

Running this command generates three directories under `ecp`:

* `ecp/stats`: Contains plots of online statistics.
* `ecp/metric`: Contains evaluation metrics in JSON format.
* `ecp/traj`: Contains robot trajectories stored as NumPy arrays.
