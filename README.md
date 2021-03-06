# ActiveLearning
Despite the medical imaging capabilities demonstrated by deep learning with respect to automatic tumor segmentation, the 
large amount of data required for these models presents a major limitation. This barrier can be prohibitive given the 
high expertise necessary and the time-consuming nature of segmentation. Here, we provide a framework for active learning 
to minimize the amount of data required for state-of-the-art performance by iteratively identifying batches of data that 
would be most informative for model training.

## Framework Overview
### Initial Dataset Selection
Initial data to be annotated can be selected either randomly or using radiomics features to build a training set with 
the most diverse features. Radiomic feature extraction is performed with PyRadiomics,an open-source Python package for 
extracting radiomic features from medical images. The skull-stripped brain region is used as the mask to focus feature 
extraction. Characteristics extracted are first ordered features, including mean, median, entropy, and energy. An 
initial dataset can be obtained by selecting the subset of images that maximizes the minimum or mean Euclidean distance 
between normalized feature arrays.

### Selecting Subsequent Datasets
Subsequent datasets can be constructed based on uncertainty estimations and redundancy restriction. Uncertainty sampling 
first uses a model trained on the previous iteration of data to predict segmentations on unannotated data. Uncertainty 
is calculated from the predicted segmentations, and images with the highest uncertainty are chosen for annotation. 
Uncertainty scores estimate the model’s confidence on data that was not included in training. A subset of images from 
those that have the greatest uncertainty scores can also be selected for annotation to reduce redundancy. This subset
can be identified as those that are most representative of the uncertain images or those that are most non-similar to
currently annotated data.

## Setup
### Code
Our code was written in Python 3. After cloning the repository, please install the dependencies in *Pipfile* by running
```
pipenv install
```

### Dataset
To configure your institutional dataset for our code, please refer to the *config.yaml* file to set the appropriate
specifications.

## Running the Code
After updating *config.yaml*, run *initial_setup.py* to select the images to be annotated for a given iteration. 
Running *initial_setup.py* to build the next iteration will select the images for the Train and Val sets, indicating 
them in *AL_data.csv*. Running *initial_setup.py* will also allow for building a dataset from a previous iteration
using the log file, creating *Train* and *Val* directories with the image files selected to be annotated. After training 
a model using the selected Train and Val sets for the given iteration, identification of samples to annotate for the 
next iteration may rely on uncertainty and redundancy restriction measures determined from predictions of the model on 
the unannotated dataset.

## Utilizing Pre-trained Models
Pre-trained models are available as separate docker containers with the models loaded inside. There are several
iterations (0-6) for each, with each subsequent iteration trained on more data than the previous. There are also 4
models, each indicated by a model_number from 0-3, for each iteration for bootstrapping.

For random query:
```bash
docker pull dannyhki/al:rq_<iteration_number>
```
For bootstrapping:
```bash
docker pull dannyhki/al:bs_<iteration_number>_<model_number>
```
For dropout traintest:
```bash
docker pull dannyhki/al:do_traintest_<iteration_number>
```
For representative redundancy restriction with dropout traintest:
```bash
docker pull rajatchandra/al:representative_<iteration_number>
```
For non-similar redundancy restriction with dropout traintest:
```bash
docker pull rajatchandra/al:dissimilar_<iteration_number>
```

One of these pre-trained models can be set up onto the preprocessed data. For example, for iteration 0 of random query:
```bash
docker run --runtime=nvidia -it -v <path_to_data>:/home/neural_network_code/Data/Patients/ dannyhki/al:rq_0 bash
```

## References
1. Smailagic, A., Noh, H.Y., Costa, P., Walawalkar, D., Khandelwal, K., Mirshekari, M., Fagert, J., Galdrán, A., Xu, S.: 
   Medal: Deep active learning sampling method for medical image analysis (September 01, 2018 2018), 
   https://ui.adsabs.harvard.edu/abs/2018arXiv180909287S
2. van Griethuysen, J.J.M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R.G.H., 
   Fillion-Robin, J.C., Pieper, S., Aerts, H.: Computational radiomics system to decode the radiographic phenotype. 
   Cancer Res 77(21), e104–e107 (2017). https://doi.org/10.1158/0008-5472.Can-17-0339, 1538-7445 2017/11/03 Cancer 
   Res. 2017 Nov 1;77(21):e104-e107. doi: 10.1158/0008-5472.CAN-17-0339.
