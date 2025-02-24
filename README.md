# Adsorption Energy Prediction for High-Entropy Surfaces
This package (GCN-HEENG) implements a graph neural network for the prediction of adsorption energies of different gas molecules, herein H and CO, onto high-entropy alloys. It constructs graphs based on the chemical environment in vicinity to the adsorbate. The training is performed in a message passing (convolution) scheme.

The framework and results are comprehensively discussed in [my paper](https://pubs.aip.org/aip/aml/article/2/2/026103/3280563).

## Table of Contents

- [Installation](#Installation)
- [Usage](#usage)
  - [Trajectory to Graph](#MD-Data-Processing)
  - [Training](#Graph-neural-network-training)
  - [Analysis using gradient](#Saliency-Map)
  - [Analysis using masking](#masking-Explaination)
  - [Visualize the results](#visualize-the-results)
- [Data](#data)
- [Authors](#authors)
- [License](#license)
- [License and credits](#License-and-credits)

## License and credits
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](./license.txt)
If you use this project in your research, please cite it as follows:
- Hananeh Oliaei, Narayana R. Aluru. "Study of the adsorption sites of high entropy alloys for CO2 reduction using graph convolutional network" APL Machine Learning 2, 026103 (2024).
[![https://doi.org/10.1063/5.0198043]]


# GCN-for-Modeling-CO2RR-catalyst-High-Entropy-Alloys
In the data/input folder the 'HEA_properties.csv' file contains the element intrinsic properties. the rest of the csv files within this folder are brought/downloaded from the work by Pedersen et al. on high entropy alloys. [1]

The src/shared folder includes scripts for 1)constructing graphs from the csv files and their corresponding featurization and 2)defining the architecture of the GCN model.

The src/training folder includes scripts for training the GCN model on the constructed graphs (we trained four models with different initializations).

The src/explanation folder includes scripts for explaining the trained GCN models (and their predictions) in addition to ranking the importance of the node features.

The src/test_training_data folder tests the trained GCN models on the training data which helps to visualize the GCN predictions versus DFT values (figure 2 in our manuscript).

The src/test_testing_data folder tests the trained GCN models on the data not seen by them (it includes all the possible combinations of the elements) which helps to evaluate their robustness.

[1]: https://pubs.acs.org/doi/full/10.1021/acscatal.9b04343
