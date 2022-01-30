# TODO the pseudo file code is bugged

import os 
import shutil 
from glob import glob 
from tqdm import tqdm 
from pqdm.processes import pqdm
import numpy as np 
import nibabel as nib 
import pandas as pd
import random 
import time
import yaml
from sklearn.utils import resample
import SimpleITK as sitk
from radiomics import firstorder, glcm, shape, glszm, glrlm, ngtdm, gldm
from sklearn.metrics.pairwise import cosine_similarity


N_BOOTSTRAPPED_MODELS = 4

# IO and file setup
class ActiveLearner():
    def __init__(self, config):
        self.config = config
        self.iteration_path = os.path.join(self.config['model_predictions_path'], 'iteration_'+ str(self.config['active_learning_iteration'] - 1))
        # initialize empty dicts to store patient feature values
        self.patient_feature_values = {}
        self.scaled_patient_feature_values = {}

    def get_all_files(self):
        """
        Gets all of the patient mrns in the data folder

        Returns
        -------
            list: all patient mrns in the data folder
        """
        return os.listdir(self.config["all_files_path"])
    
    def get_annotated_files(self):
        """
        Gets all of the patient mrns that have been annotated based the log
        history. If the iteration number is zero, returns empty list.

        Returns
        -------
            list: all patient mrns that have been annotated
        """
        if self.config["active_learning_iteration"] == 0:
            return []
        else:
            current_iteration = self.config["active_learning_iteration"]
            csvpath_ofinterest = os.path.join(self.config["export_path"], self.config["unique_id"], "iteration_" + str(current_iteration - 1), "AL_groupings.csv") 
            df = pd.read_csv(csvpath_ofinterest, dtype = str)
            return list(df[~df["annotated"].isnull()]["annotated"]) + list(df[~df["to_annotate"].isnull()]["to_annotate"])
    
    def get_unannotated_files(self):
        """
        Gets all of the patient mrns that have not been annotated based on the
        log history. This includes patients who have been previously flagged
        for pseudo labeling. If the iteration number is zero, returns all
        patient mrns

        Returns
        -------
            list: all patient mrns that have not been annotated
        """
        all_patients = self.get_all_files()
        current_iteration = self.config["active_learning_iteration"]
        if current_iteration == 0:
            return all_patients
        else:
            annotated = self.get_annotated_files()
            # changed
            # return [x for x in all_patients if x not in annotated]
            return list(set(all_patients) - set(annotated))

    def get_x_random_unannotated_files(self, x, seed = None):
        """
        Returns array with x randomly selected patient mrns without annotations

        Parameters
        ----------
        x : int
            the number of samples to randomly select
        
        Returns
        -------
            list: randomly selected unannoated patient mrns
        """
        unannotated = self.get_unannotated_files()
        return self.get_x_random_files_from_subset(x, unannotated, seed = seed)

    def get_x_random_files_from_subset(self, x, subset, seed = None):
        """
        Returns array with x randomly selected samples from subset

        Parameters
        ----------
        x : int
            the number of samples to randomly select
        
        Returns
        -------
            list: randomly selected unannoated patient mrns
        """
        np.random.seed(seed)
        random.seed(seed)
        np.random.shuffle(subset)
        return subset[:x]

    def get_images(self, patient):
        """
        Gets images for given patient mrn

        Parameters
        ----------
        patient : str
            the patient mrn

        Returns
        -------
            list of sitk images for patient
        """
        image_paths = [os.path.join(self.config['all_files_path'], patient, image_path)
                       for image_path in self.config['file_names']['image_names']]
        return [sitk.ReadImage(image_path) for image_path in image_paths]

    @staticmethod
    def generate_mask(image):
        """
        Generates mask for given image, where any 0 in the image is a 0 in the mask, all other values are 1 in the mask.

        Parameters
        ----------
        image : sitk object
            the image to generate the mask for

        Returns
        -------
            sitk mask for patient
        """

        # generate mask for patient, 0s if 0 in image, otherwise 1
        image_array = sitk.GetArrayFromImage(image)
        mask_array = np.where(image_array == 0, 0, 1)
        mask = sitk.GetImageFromArray(mask_array)
        return mask

    def get_mask(self, patient):
        """
        Gets mask for given patient mrn, use if there is a path to the mask specified.

        Parameters
        ----------
        patient : str
            the patient mrn

        Returns
        -------
            sitk mask for patient
        """

        # if there is a path to mask file, can modify code below
        mask_path = os.path.join(self.config['all_files_path'], patient, self.config['file_names']['roi_name'])
        if os.path.exists(mask_path):
            mask = sitk.ReadImage(mask_path)
            # binarize mask
            mask_array = sitk.GetArrayFromImage(mask)
            mask_array = np.round(mask_array).astype(int)
            # mask_array = np.ones(mask_array.shape, np.int8)
            mask = sitk.GetImageFromArray(mask_array)
            return mask
        print(f'No mask found for {patient}')

    def get_encoded_feature_map(self, patient):
        """
        Gets encoded feature map for given patient mrn

        Parameters
        ----------
        patient : str
            the patient mrn

        Returns
        -------
            encoded feature map array for patient
        """
        prev_iteration_str = f"iteration_{int(self.config['active_learning_iteration']) - 1}"
        encoded_feature_map_path = os.path.join(self.config['model_predictions_path'], prev_iteration_str, patient,
                                                self.config['representativeness']['encoded_feature_map_name'])
        return np.load(encoded_feature_map_path)

    def get_initial_dataset_from_log(self):
        """
        Gets the initial dataset. The iteration must be > 0 as this method
        pulls from the log history

        Returns
        -------
            list : all patient mrns that were part of the initial dataset
        """
        if self.config["active_learning_iteration"] != 0:
            current_iteration = self.config["active_learning_iteration"]
            csvpath_ofinterest = os.path.join(self.config["export_path"], self.config["unique_id"], "iteration_" + str(current_iteration - 1), "AL_groupings.csv") 
            df = pd.read_csv(csvpath_ofinterest, dtype = str)
            return list(df[~df["initial"].isnull()]["initial"])
        else:
            raise RuntimeError("Need a log history to call get_initial_dataset_from_log")

    def _check_ascending(self, x):
        x_copy = x.copy()
        x_copy.sort()
        return x_copy == x

    # initial dataset generator
    def random_initialization(self):
        """
        Returns an array of m patient mrns that were randomly selected from the
        all patient mrn list

        Returns
        -------
            list: randomly picked patient mrns of size m
        """
        print("Setting up initial dataset randomly")
        np.random.seed(self.config["random_seed"])
        random.seed(self.config["random_seed"])
        all_patients = self.get_all_files()
        np.random.shuffle(all_patients)

        return all_patients[:self.config["initial_dataset_generator"]["m"]]

    def pyradiomics_feature_description(self, patient):
        """
        Saves feature descriptors using pyradiomics for patient in object's patient_feature_values dict

        Parameters
        ----------
        patient : str
            patient mrn to describe

        Returns
        -------
            adds feature vector to patient_feature_values dictionary, does not return anything
        """

        # get image and mask for patient
        images = self.get_images(patient)
        masks = [self.generate_mask(image) for image in images]

        # settings used in pyradiomics example
        settings = {
            'binWidth': 25,
            'interpolator': sitk.sitkBSpline,
            'resampledPixelSpacing': None,
        }

        # empty list is used to enable all features
        features_to_enable_dict = {
            'firstorder': [],
            # 'glcm': [],
            # 'glszm': [],
            # 'glrlm': [],
            # 'ngtdm': [],
            # 'gldm': [],
            # 'shape': [],
        }

        # find features for each image
        feature_vectors_list = []
        for image, mask in zip(images, masks):
            # using firstorder as default for generating feature vector
            # there are also several additional features that pyradiomics can calculate
            feature_descriptors_dict = {
                'firstorder': firstorder.RadiomicsFirstOrder(image, mask, **settings),
                # 'glcm': glcm.RadiomicsGLCM(image, mask, **settings),
                # 'glszm': glszm.RadiomicsGLSZM(image, mask, **settings),
                # 'glrlm': glrlm.RadiomicsGLRLM(image, mask, **settings),
                # 'ngtdm': ngtdm.RadiomicsNGTDM(image, mask, **settings),
                # 'gldm': gldm.RadiomicsGLDM(image, mask, **settings),
                # 'shape': shape.RadiomicsShape(image, mask, **settings),
            }

            # initialize dict to store results from feature description
            feature_results_dict = {}

            # determine maximum number of features for each category, will pad with 0s in array to make uniform size
            max_num_features = 0

            # calculate features
            for feature_descriptor_name, feature_descriptor in feature_descriptors_dict.items():

                # enable features as specified in features_to_enable_dict
                features_to_enable = features_to_enable_dict.get(feature_descriptor_name, [])
                if not features_to_enable:
                    feature_descriptor.enableAllFeatures()
                else:
                    feature_descriptor.disableAllFeatures()
                    for feature_to_enable in features_to_enable:
                        feature_descriptor.enableFeatureByName(feature_to_enable, enable=True)

                # determine num_features to find max_num_features
                num_features = len(feature_descriptor.enabledFeatures)
                max_num_features = max(num_features, max_num_features)

                # get results of feature description and store
                results = feature_descriptor.execute()
                feature_results_dict[feature_descriptor_name] = results

            # include feature results in array
            feature_descriptor_vector = np.zeros((len(feature_descriptors_dict), max_num_features))
            for i, (feature_descriptor_name, results) in enumerate(feature_results_dict.items()):
                # sort keys so ensure order is preserved
                results_sorted_keys = sorted(results.keys())
                # populate array
                for j in range(len(results)):
                    results_key = results_sorted_keys[j]
                    feature_descriptor_vector[i, j] = results[results_key]

            # save patient's feature results
            feature_vectors_list.append(feature_descriptor_vector)
        self.patient_feature_values[patient] = np.array(feature_vectors_list)

    def pyradiomics_feature_descriptions(self, patients):
        """
        Saves feature descriptors using pyradiomics for patients in object's patient_feature_values dict

        Parameters
        ----------
        patients : list
            patient mrns to describe

        Returns
        -------
            adds feature vectors to patient_feature_values dictionary, does not return anything
        """
        for patient in patients:
            self.pyradiomics_feature_description(patient)

    def normalize_feature_values(self):
        """
        Populates scaled_patient_feature_values for each patient, normalizing values between 0 and 1.

        Returns
        -------
            adds scaled feature vectors to scaled_patient_feature_values dictionary by normalizing values in the range
            [0, 1], does not return anything
        """
        # initialize min and max patient feature values
        patient_feature_values_array = np.array(list(self.patient_feature_values.values()))
        min_patient_feature_values = patient_feature_values_array.min(axis=0, initial=None)
        max_patient_feature_values = patient_feature_values_array.max(axis=0, initial=None)

        # normalize on scale from 0 to 1
        for patient, patient_feature_values in self.patient_feature_values.items():
            scaled_patient_feature_values = (patient_feature_values - min_patient_feature_values) / \
                                           (max_patient_feature_values - min_patient_feature_values)
            # fill nans with 0.5
            scaled_patient_feature_values[np.isnan(scaled_patient_feature_values)] = 0.5
            # save results
            self.scaled_patient_feature_values[patient] = scaled_patient_feature_values

    def z_score_feature_values(self):
        """
        Populates scaled_patient_feature_values for each patient with z-scores.

        Returns
        -------
            adds scaled feature vectors to scaled_patient_feature_values dictionary by z-scoring values,
            does not return anything
        """
        # initialize mean and std patient feature values
        patient_feature_values_array = np.array(list(self.patient_feature_values.values()))
        mean_patient_feature_values = patient_feature_values_array.mean(axis=0)
        std_patient_feature_values = patient_feature_values_array.std(axis=0)

        # calculate z scores
        for patient, patient_feature_values in self.patient_feature_values.items():
            scaled_patient_feature_values = (patient_feature_values - mean_patient_feature_values) / \
                                            std_patient_feature_values
            # fill nans with 0
            scaled_patient_feature_values[np.isnan(scaled_patient_feature_values)] = 0
            # save results
            self.scaled_patient_feature_values[patient] = scaled_patient_feature_values

    @staticmethod
    def calculate_array_distances(a, b):
        """
        Returns the Euclidean distance between two arrays

        Parameters
        ----------
        a : array of arrays to find distance from b
        b : reference array that distances are found from a

        Returns
        -------
            array: Euclidean distances of all arrays in a from b

        """
        return np.sqrt(((np.subtract(a, b)) ** 2).sum(axis=-1))

    def get_feature_distances(self, patient):
        """
        Returns a df with distances of patients to ref_patient (given) across different images and feature vectors
        provided in scaled_patient_feature_values

        Parameters
        ----------
        patient : str
            patient mrn to describe

        Returns
        -------
            df: dataframe of columns patient, ref_patient, image_num, feature_vector_num, and distance where patient is
            the mrn that the distance is calculated from the ref_patient (given) and image_num and feature_vector_num
            refer to indices of the array of images and feature values for which the distance was calculated
        """

        # calculate distance to reference patient across all feature vectors
        ref_patient_values = self.scaled_patient_feature_values[patient]
        sorted_patients = sorted(self.scaled_patient_feature_values.keys())
        all_patient_values = np.array([self.scaled_patient_feature_values[sorted_patient]
                                       for sorted_patient in sorted_patients])
        array_distances = self.calculate_array_distances(all_patient_values, ref_patient_values)

        # put in dataframe
        df = pd.DataFrame({
            'patient': sorted_patients,
            'ref_patient': patient,
            'array_distances': list(array_distances),
        })

        # split array_distances into multiple columns for each image and feature vector
        num_images = array_distances.shape[1]
        num_feature_vectors = array_distances.shape[2]
        for i in range(num_images):
            for j in range(num_feature_vectors):
                df[f'image_{i}__feature_vector_{j}'] = df['array_distances'].map(lambda x: x[i, j])

        # reshape
        df = pd.melt(df, id_vars=['patient', 'ref_patient'],
                     value_vars=[f'image_{i}__feature_vector_{j}'
                                 for i in range(num_images) for j in range(num_feature_vectors)],
                     var_name='image_feature_vector_nums', value_name='distance')
        df[['image_num', 'feature_vector_num']] = df['image_feature_vector_nums'].str.split('__', expand=True)

        return df[['patient', 'ref_patient', 'image_num', 'feature_vector_num', 'distance']]

    def feature_description_initialization(self, num_rand_patient_mrns=1):
        """
        Returns an array of m patient mrns that were identified to be most
        different from each other using feature description. The principle
        behind this is to create a well representative initial dataset using
        computer vision.

        Parameters
        ----------
        num_rand_patient_mrns : int
            the number of patient mrns to randomly select, defaults to 1

        Returns
        -------
            list: patient mrns that were identified as being most different from each other
        """

        # pull number of patients to initialize with from config
        num_initial_patients_to_find = self.config['initial_dataset_generator']['m']

        # get all patients
        all_patients = self.get_all_files()

        # return if number of all patients is less than or equal to the number to find
        if len(all_patients) <= num_initial_patients_to_find:
            print(f"Number of initial patients to find ({num_initial_patients_to_find}) is greater than or equal to "
                  f"number of patients for which there are records ({len(all_patients)}). Returning all patients for "
                  f"which there are records.")
            return all_patients

        # make sure farthest_metric is appropriate value
        assert self.config["initial_dataset_generator"]["farthest_metric"] in ["mean", "minimum"], \
            "Backend for initial dataset generator farthest_metric not recognized"

        # TODO: limit to some number of patients in all_patients with highest entropy as in MedAL paper?
        # initialize with num_rand_patient_mrns and find patients farthest on average from them
        print("Setting up initial dataset with patient mrns identified to be most different from each other.")
        np.random.seed(self.config["random_seed"])
        random.seed(self.config["random_seed"])

        # generate feature values for patients
        self.pyradiomics_feature_descriptions(all_patients)

        # scale feature values
        self.z_score_feature_values()

        # initialize patients with random selection
        initial_patients = list(np.random.choice(all_patients, size=num_rand_patient_mrns, replace=False))

        # get distances for all patients from initial patients
        distance_dfs = [self.get_feature_distances(patient=initial_patient)
                        for initial_patient in initial_patients]
        distance_df = pd.concat(distance_dfs, axis='rows', ignore_index=True)

        # find farthest patient and add to initial patients, repeat until found enough patients
        while len(initial_patients) < num_initial_patients_to_find:
            # only consider distances for patients not already in initial_patients
            remaining_patient_distance_df = distance_df[~distance_df['patient'].isin(initial_patients)]\
                .reset_index(drop=True)

            # aggregate distances across ref_patients, then feature_vector_num, then image_num to find farthest patient
            if self.config["initial_dataset_generator"]["farthest_metric"] == "mean":
                # find mean distance to all ref patients across all image_num and feature_vector_num
                remaining_patient_distance_agg_df = remaining_patient_distance_df \
                    .groupby(['patient', 'image_num', 'feature_vector_num'])[['distance']].mean().reset_index()
                # find mean distance across all feature_vectors for all images
                remaining_patient_distance_agg_df = remaining_patient_distance_agg_df \
                    .groupby(['patient', 'image_num'])[['distance']].mean().reset_index()
                # find mean distance across all images for each patient
                remaining_patient_distance_agg_df = remaining_patient_distance_agg_df \
                    .groupby(['patient'])[['distance']].mean().reset_index()
            elif self.config["initial_dataset_generator"]["farthest_metric"] == "minimum":
                # find minimum distance to all ref patients across all image_num and feature_vector_num
                remaining_patient_distance_agg_df = remaining_patient_distance_df \
                    .groupby(['patient', 'image_num', 'feature_vector_num'])[['distance']].min().reset_index()
                # find minimum distance across all feature_vectors for all images
                remaining_patient_distance_agg_df = remaining_patient_distance_agg_df \
                    .groupby(['patient', 'image_num'])[['distance']].min().reset_index()
                # find minimum distance across all images for each patient
                remaining_patient_distance_agg_df = remaining_patient_distance_agg_df \
                    .groupby(['patient'])[['distance']].min().reset_index()
            else:
                raise ValueError("Backend for initial dataset generator farthest_metric not recognized")

            # add patient farthest away to initial patients (rank is min)
            farthest_patient = remaining_patient_distance_agg_df.loc[
                remaining_patient_distance_agg_df['distance'] == remaining_patient_distance_agg_df['distance'].max(),
                'patient'
            ].values[0]
            initial_patients.append(farthest_patient)

            # add farthest_patient as reference patient in distance_df, if need to find more initial patients
            if len(initial_patients) < num_initial_patients_to_find:
                farthest_patient_distances_df = self.get_feature_distances(farthest_patient)
                distance_df = distance_df.append(farthest_patient_distances_df, ignore_index=True)

        return initial_patients

    def initial_training_dataset(self):
        """
        Returns an array of m patient mrns for initial training based on
        the backend specified in the config

        Returns
        -------
            list: patient mrns for initial dataset training
        """
        if self.config["initial_dataset_generator"]["backend"] == "random":
            return self.random_initialization()
        elif self.config["initial_dataset_generator"]["backend"] == "feature_description_initialization":
            return self.feature_description_initialization()
        else:
            raise ValueError("Backend for initial dataset generator not recognized")
        
    # uncertainty
    def uncertainty_none(self):
        """
        Returns an array of K randomly selected patient mrns without annotations.
        This is identical to if active learning did not select for uncertain
        samples

        Returns
        -------
            list: patient mrns randomly selected
        """
        print("Getting uncertain samples by random selection")
        return self.get_x_random_unannotated_files(self.config["uncertainty"]["K"], seed = self.config["random_seed"])
        
    def uncertainty_bootstrapped(self):
        """
        Returns an array of K patient mrns with the most variance across the
        predictions from the bootstrapped models

        Returns
        -------
            list: most uncertain patient mrns derived from bootstrapped predictions
        """
        print("Getting uncertain samples by bootstrapping variance selection")
        # get the list of unannotated patients
        unannotated_patients = self.get_unannotated_files()
        variance = []

        # print the labels that we are looking at
        print("Model Predictions at...")
        print("  ", self.iteration_path)
        predicted_names = glob(os.path.join(self.iteration_path, unannotated_patients[0], self.config["file_names"]["probability_map_name"].split(".nii")[0] + "*"))
        print("Using the following probability map names")
        for x in predicted_names:
            print("  ", os.path.basename(x))
        
        # calculate the variances of all of the patients
        if self.config["uncertainty"]["parallel"]:
            variance = pqdm(unannotated_patients, self._get_variance, self.config["n_jobs"])
        else:
            for patient in tqdm(unannotated_patients):
                variance.append(self._get_variance(patient))

        # create a df of all of the unannotated patients and their variences
        sorted_patients_variances = np.array([[pt, v] for v, pt in sorted(zip(variance, unannotated_patients))])
        variance_df = pd.DataFrame(data = sorted_patients_variances, columns = ["Patient_mrn", "Variance"])
        variance_df["Selected"] = ["N" for i in range(len(unannotated_patients) - self.config["uncertainty"]["K"])] + ["Y" for i in range(self.config["uncertainty"]["K"])]
        variance_logger = Logger(self.config)
        variance_logger.write_uncertainty_log(variance_df, "bootstrapped")
        
        # return the k patients with the highest variance
        return sorted_patients_variances[:,0][-1 * self.config["uncertainty"]["K"]:]
    
    def _get_variance(self, patient):
        proportion = self.config["uncertainty"]["variance_pixel_proportion"]

        # if variance array is already provided
        variance_array_path = os.path.join(self.iteration_path, patient, self.config["file_names"]["variance_array"])
        if os.path.exists(variance_array_path):
            variance_array = np.array(nib.load(variance_array_path).dataobj)
        else:
            all_labels_for_patient = glob(os.path.join(self.iteration_path, patient, self.config["file_names"]["probability_map_name"].split(".nii")[0] + "*"))
            all_label_data_for_patient = [np.array(nib.load(x).dataobj) for x in all_labels_for_patient]
            np_all_label_data_for_patient = np.array(all_label_data_for_patient)
            variance_array = np.var(np_all_label_data_for_patient, axis = 0)
        flattened = variance_array.flatten()
        flattened.sort()
        ncapture = int(len(flattened) * proportion)
        mean_variance = np.mean(flattened[-1 * ncapture:])
        return mean_variance

    def uncertainty_prob_roi(self):
        """"
        Returns an array of K patient mrns with the lowest mean probability at the
        ROI.

        Returns
        -------
            list: most uncertain patient mrns derived from mean probability of ROI
        """
        print("Getting uncertain samples by mean probability")
        unannotated_patients = self.get_unannotated_files()
        uncertainties = []

        # calculate the mean probability at the ROI of all of the patients
        if self.config["uncertainty"]["parallel"]:
            uncertainties = pqdm(unannotated_patients, self._get_mean_prob_at_roi, self.config["n_jobs"])
        else:
            for patient in tqdm(unannotated_patients):
                uncertainties.append(self._get_mean_prob_at_roi(patient))
        
        # create a df of all of the unannotated patients and their prob roi
        sorted_patients_uncertainties = np.array([[pt, v] for v, pt in sorted(zip(uncertainties, unannotated_patients))])
        uncertainty_df = pd.DataFrame(data = sorted_patients_uncertainties, columns = ["Patient_mrn", "Prob_at_ROI"])
        uncertainty_df["Selected"] = ["Y" for i in range(self.config["uncertainty"]["K"])] + ["N" for i in range(len(unannotated_patients) - self.config["uncertainty"]["K"])]
        uncertainty_logger = Logger(self.config)
        uncertainty_logger.write_uncertainty_log(uncertainty_df, "prob_at_roi")
        
        # return k patients with the lowest uncertainties
        return sorted_patients_uncertainties[:, 0][:self.config["uncertainty"]["K"]]
    
    def _get_mean_prob_at_roi(self, patient):
        all_prob_for_patient = glob(os.path.join(self.iteration_path, patient, self.config["file_names"]["probability_map_name"].split(".nii")[0] + "*"))
        all_groundtruth_for_patient = glob(os.path.join(self.config["all_files_path"], patient, self.config["file_names"]["roi_name"]))
        if len(all_prob_for_patient) > 1:
            raise AssertionError("There are more identified predictions than allowed")
        
        prob_data_for_patient = np.array(nib.load(all_prob_for_patient[0]).dataobj)
        gt_data_for_patient = np.array(nib.load(all_groundtruth_for_patient[0]).dataobj)

        if np.sum(gt_data_for_patient) > 0:
            return np.mean(prob_data_for_patient[gt_data_for_patient == 1])
        return 0


    def uncertainty_margin(self):
        """
        Returns an array of K patient mrns with the largest negative margin in probability
        between the foreground and background.

        Returns
        -------
            list: most uncertain patient mrns derived from margin sampling
        """
        print("Getting uncertain samples by least margins")
        unannotated_patients = self.get_unannotated_files()
        margins = []

        # print the name of the probability map we are looking at
        print("Model Predictions at...")
        print("  ", self.iteration_path)
        print("Using the following probability map names")
        for x in glob(os.path.join(self.iteration_path, unannotated_patients[0], self.config["file_names"]["probability_map_name"].split(".nii")[0] + "*")):
            print("  ", os.path.basename(x))

        # calculate the margins of all of the patients
        if self.config["uncertainty"]["parallel"]:
            margins = pqdm(unannotated_patients, self._get_margin, self.config["n_jobs"])
        else:
            for patient in tqdm(unannotated_patients):
                margins.append(self._get_margin(patient))
        
        # create a df of all of the unannotated patients and their margins
        sorted_patients_margins = np.array([[pt, v] for v, pt in sorted(zip(margins, unannotated_patients))])
        uncertainty_df = pd.DataFrame(data = sorted_patients_margins, columns = ["Patient_mrn", "Margins"])
        uncertainty_df["Selected"] = ["N" for i in range(len(unannotated_patients) - self.config["uncertainty"]["K"])] + ["Y" for i in range(self.config["uncertainty"]["K"])]
        uncertainty_logger = Logger(self.config)
        uncertainty_logger.write_uncertainty_log(uncertainty_df, "margins")
            
        # return k patients with the highest margins
        return sorted_patients_margins[:,0][-1 * self.config["uncertainty"]["K"]:]

    def _get_margin(self, patient):
        all_prob_for_patient = glob(os.path.join(self.iteration_path, patient, self.config["file_names"]["probability_map_name"].split(".nii")[0] + "*"))
        if len(all_prob_for_patient) > 1:
            raise AssertionError("There are more identified predictions than allowed")
        
        prob_data_of_foreground_for_patient = np.array(nib.load(all_prob_for_patient[0]).dataobj)
        prob_data_of_background_for_patient = np.ones(prob_data_of_foreground_for_patient.shape) - prob_data_of_foreground_for_patient
        margin_data = np.abs(prob_data_of_foreground_for_patient - prob_data_of_background_for_patient)
        return -1 * np.mean(margin_data)

    def uncertainty_dropout(self):
        """
        Returns an array of K patient mrns with the largest variance/max entropy

        Returns
        -------
            list: most uncertain patient mrns derived from dropout predictions
        """
        print("Getting uncertain samples by bootstrapping variance selection")
        # get the list of unannotated patients
        unannotated_patients = self.get_unannotated_files()
        metric = []

        # print the labels that we are looking at
        print("Model Predictions at...")
        print("  ", self.iteration_path)
        predicted_names = glob(os.path.join(self.iteration_path, unannotated_patients[0], self.config["file_names"]["probability_map_name"].split(".nii")[0] + "*"))
        print("Using the following probability map names")
        for x in predicted_names:
            print("  ", os.path.basename(x))
        
        # calculate the variances of all of the patients
        header_names = []
        acquisition_function = ""

        if self.config["uncertainty"]["if_dropout"] == "variance":
            header_names = ["Patient_mrn", "Variance"]
            acquisition_function = "Variance"
            if self.config["uncertainty"]["parallel"]:
                metric = pqdm(unannotated_patients, self._get_variance, self.config["n_jobs"])
            else:
                for patient in tqdm(unannotated_patients):
                    metric.append(self._get_variance(patient))
        elif self.config["uncertainty"]["if_dropout"] == "max_entropy":
            header_names = ["Patient_mrn", "Max_Entropy"]
            acquisition_function = "Max_Entropy"
            if self.config["uncertainty"]["parallel"]:
                metric = pqdm(unannotated_patients, self._get_max_entropy, self.config["n_jobs"])
            else:
                for patient in tqdm(unannotated_patients):
                    metric.append(self._get_max_entropy(patient))

        # create a df of all of the unannotated patients and their metrics (variance/max_entropy)
        sorted_patients_metrics = np.array([[pt, v] for v, pt in sorted(zip(metric, unannotated_patients))])
        metric_df = pd.DataFrame(data = sorted_patients_metrics, columns = header_names)
        metric_df["Selected"] = ["N" for i in range(len(unannotated_patients) - self.config["uncertainty"]["K"])] + ["Y" for i in range(self.config["uncertainty"]["K"])]
        metric_logger = Logger(self.config)
        metric_logger.write_uncertainty_log(metric_df, "dropout_" + acquisition_function)
        
        # return the k patients with the highest variance
        return sorted_patients_metrics[:,0][-1 * self.config["uncertainty"]["K"]:]

    def _get_max_entropy(self, patient):
        all_labels_for_patient = glob(os.path.join(self.iteration_path, patient, self.config["file_names"]["probability_map_name"].split(".nii")[0] + "*"))

        # load all of the data into an array
        all_label_data_for_patient = [np.array(nib.load(x).dataobj) for x in all_labels_for_patient]
        np_all_label_data_for_patient = np.array(all_label_data_for_patient)

        # find the mean probability map across all predictions
        mean_prob_map = np.mean(np_all_label_data_for_patient, axis = 0)

        # get the max entropy: -(xlog(x) + (1-x)(log(1-x)))
        max_entropy_map = -1 * ((mean_prob_map * np.log(mean_prob_map+0.00000000001)) + ((1-mean_prob_map) * np.log(1-mean_prob_map + 0.00000000001)))
        mean_max_entropy = np.mean(max_entropy_map)
        return mean_max_entropy

    def get_uncertain_samples(self):
        """
        Returns an array of K patient mrns using the backend specified in the
        config

        Returns
        -------
            list: most uncertain patient mrns 
        """
        backend_ix = -1
        current_iteration = self.config["active_learning_iteration"]
        if self.config["uncertainty"]["switch"][0] == "None":
            backend_ix = 0
        else:
            if len(self.config["uncertainty"]["switch"]) != len(self.config["uncertainty"]["backend"]):
                raise ValueError("The length of the backend and switch array within the uncertainty parameter must be equal")
            
            if not self._check_ascending(self.config["uncertainty"]["switch"]):
                raise ValueError("The switch array for uncertainty parameter must be in ascending order")
            
            for i, switch_i in enumerate(self.config["uncertainty"]["switch"]):
                if current_iteration >= switch_i:
                    backend_ix = i
        
        selected_backend = self.config["uncertainty"]["backend"][backend_ix]
        if selected_backend == "None":
            return selected_backend, self.uncertainty_none()
        elif selected_backend == "bootstrapped":
            return selected_backend, self.uncertainty_bootstrapped()
        elif selected_backend == "prob_roi":
            return selected_backend, self.uncertainty_prob_roi()
        elif selected_backend == "margin":
            return selected_backend, self.uncertainty_margin()
        elif selected_backend == "dropout":
            return selected_backend, self.uncertainty_dropout()
        else:
            raise ValueError(selected_backend + " not recognized")            

    # representativeness
    def representativeness_none(self, subset):
        """
        Returns an array of k random patient mrns from subset. The size of the subset
        must be larger or equal to k.

        Parameters
        ----------
        subset : list
            List of patient mrns to choose from

        Returns
        -------
        list : k random samples within the subset
        """
        return self.get_x_random_files_from_subset(self.config["representativeness"]["k"], subset, seed = self.config["random_seed"])

    @staticmethod
    def flatten_encoded_feature_map(encoded_feature_map):
        """
        Returns a flattened encoded feature map by averaging along the last dimension.

        Parameters
        ----------
        encoded_feature_map : numpy array
            array for encoded feature map

        Returns
        -------
        numpy array : result of averaging the arrays of the last dimensions of encoded feature map (array of shape
        (x, y, z) will return an array of length z representing the average of all x*y arrays that have z elements in
        the last dimension)
        """

        last_dimension_size = encoded_feature_map.shape[-1]
        flattened_encoded_feature_map = np.zeros(last_dimension_size)
        for i in range(last_dimension_size):
            flattened_encoded_feature_map[i] = encoded_feature_map[..., i].mean()
        return flattened_encoded_feature_map

    @staticmethod
    def cosine_similarity_map(patient_encoded_feature_maps):
        """
        Returns a dictionary of cosine similarities for each pairwise combination of patients' encoded feature maps

        Parameters
        ----------
        patient_encoded_feature_maps : dict
            dictionary where the key is the patient and the value is the encoded feature map (a numpy array)

        Returns
        -------
        dict : dictionary of cosine similarities for each pairwise combination of patients' encoded feature maps, where
        the key is the patient and the value is another dictionary, which has key of all other patients and value of
        cosine similarity
        """

        patients = list(patient_encoded_feature_maps.keys())
        cosine_similarity_map = {patient: dict() for patient in patients}

        for i in range(len(patients)):  # include last patient for self comparison
            for j in range(i, len(patients)):  # start with i instead of i + 1 to include self comparison
                patient_1 = patients[i]
                patient_2 = patients[j]

                encoded_feature_map_1 = patient_encoded_feature_maps[patient_1]
                encoded_feature_map_2 = patient_encoded_feature_maps[patient_2]

                cosine_similarity_val = cosine_similarity(
                    encoded_feature_map_1.reshape(1, -1),
                    encoded_feature_map_2.reshape(1, -1)
                )[0, 0]
                cosine_similarity_map[patient_1][patient_2] = cosine_similarity_val
                cosine_similarity_map[patient_2][patient_1] = cosine_similarity_val

        return cosine_similarity_map

    @staticmethod
    def find_max_cosine_similarity(patients_tba, patient, encoded_feature_cosine_similarity_map):
        """
        Returns the max cosine similarity of all cosine similarities when patient's encoded_feature_map is paired with
        any encoded feature map of a patient in patients_tba, which is a measure of how well the patient's
        encoded_feature_map is represented by those of patients_tba

        Parameters
        ----------
        patients_tba : list
            List of patients to be annotated
        patient : str
            patient
        encoded_feature_cosine_similarity_map: dict
        dictionary of cosine similarities for each pairwise combination of patients' encoded feature maps, where
        the key is the patient and the value is another dictionary, which has key of all other patients and value of
        cosine similarity

        Returns
        -------
        float : maximum cosine similarity for encoded_feature_map with an element in encoded_feature_maps_tba
        """

        patients_tba_cosine_similarities = [encoded_feature_cosine_similarity_map[patient][patient_tba]
                                            for patient_tba in patients_tba]
        assert len(patients_tba_cosine_similarities) == len(patients_tba), 'Some cosine similarities not found'

        return max(patients_tba_cosine_similarities)

    def calculate_representativeness(self, patients_tba, other_patients, encoded_feature_cosine_similarity_map):
        """
        Returns the max cosine similarity of all cosine similarities when patient's encoded_feature_map is paired with
        any encoded feature map of a patient in patients_tba, which is a measure of how well the patient's
        encoded_feature_map is represented by those of patients_tba

        Parameters
        ----------
        patients_tba : list
            List of patients to be annotated
        other_patients : list
            list of other patients considered for representativeness
        encoded_feature_cosine_similarity_map: dict
            dictionary of cosine similarities for each pairwise combination of patients' encoded feature maps, where
            the key is the patient and the value is another dictionary, which has key of all other patients and value of
            cosine similarity

        Returns
        -------
        float : representativeness from sum of all maximum cosine similarities from encoded feature maps for other
        patients, where the max cosine similarity is found for each patient in all patients with each patient in
        the list of patients to be annotated
        """
        representativeness = 0
        for patient in other_patients:
            max_cosine_similarity = self.find_max_cosine_similarity(patients_tba, patient,
                                                                    encoded_feature_cosine_similarity_map)

            representativeness += max_cosine_similarity

        return representativeness

    def representativeness_cosine_similarity(self, subset, highest_uncertainty_patient=None):
        """
        Returns an array of k patient mrns from subset that are most representative
        using the cosine similarity function

        Parameters
        ----------
        subset : list
            List of patient mrns to choose from
        highest_uncertainty_patient : str or None, default is None
            patient mrn with highest uncertainty to initialize with

        Returns
        -------
        list : k most representative samples within subset using the cosine
            similarity backend
        """

        # number of patients from subset to select
        k = self.config["representativeness"]["k"]

        if len(subset) < k:
            print(f'subset is too small to select {k} representative samples, returning None')
            return
        if len(subset) == k:
            return subset

        patient_encoded_feature_maps = {patient: self.flatten_encoded_feature_map(self.get_encoded_feature_map(patient))
                                        for patient in subset}
        encoded_feature_cosine_similarity_map = self.cosine_similarity_map(patient_encoded_feature_maps)
        representative_patients = []

        if highest_uncertainty_patient is not None:
            representative_patients.append(highest_uncertainty_patient)
        else:
            random.seed(self.config["random_seed"])
            representative_patients.append(random.choice(subset))
        remaining_patients = list(set(subset) - set(representative_patients))

        while len(representative_patients) < k:
            most_representative_patient = None
            max_representativeness = None
            for patient in remaining_patients:
                temp_representative_patients = representative_patients + [patient]
                temp_remaining_patients = list(set(subset) - set(temp_representative_patients))
                representativeness = self.calculate_representativeness(temp_representative_patients,
                                                                       temp_remaining_patients,
                                                                       encoded_feature_cosine_similarity_map)
                if max_representativeness is None or representativeness > max_representativeness:
                    most_representative_patient = patient
                    max_representativeness = representativeness
            assert most_representative_patient is not None, 'next most representative patient not found'
            representative_patients.append(most_representative_patient)
            remaining_patients = list(set(remaining_patients) - {most_representative_patient})

        return representative_patients

    def most_dissimilar_cosine_similarity(self, subset, highest_uncertainty_patient=None):
        """
        Returns an array of k patient mrns from subset that are most dissimilar
        using the cosine similarity function

        Parameters
        ----------
        subset : list
            List of patient mrns to choose from
        highest_uncertainty_patient : str or None, default is None
            patient mrn with highest uncertainty to initialize with

        Returns
        -------
        list : k most dissimilar samples within subset using cosine similarity
        """

        # number of patients from subset to select
        k = self.config["representativeness"]["k"]

        if len(subset) < k:
            print(f'subset is too small to select {k} representative samples, returning None')
            return
        if len(subset) == k:
            return subset

        patient_encoded_feature_maps = {patient: self.flatten_encoded_feature_map(self.get_encoded_feature_map(patient))
                                        for patient in subset}
        encoded_feature_cosine_similarity_map = self.cosine_similarity_map(patient_encoded_feature_maps)
        most_dissimilar_patients = []

        if highest_uncertainty_patient is not None:
            most_dissimilar_patients.append(highest_uncertainty_patient)
        else:
            random.seed(self.config["random_seed"])
            most_dissimilar_patients.append(random.choice(subset))
        remaining_patients = list(set(subset) - set(most_dissimilar_patients))

        while len(most_dissimilar_patients) < k:
            most_dissimilar_patient = None
            min_cosine_similarity_score = None

            for patient in remaining_patients:
                cosine_similarities = [encoded_feature_cosine_similarity_map[patient][dissimilar_patient]
                                       for dissimilar_patient in most_dissimilar_patients]
                cosine_similarity_score = sum(cosine_similarities)
                if min_cosine_similarity_score is None or cosine_similarity_score < min_cosine_similarity_score:
                    most_dissimilar_patient = patient
                    min_cosine_similarity_score = cosine_similarity_score
            assert most_dissimilar_patient is not None, 'next most dissimilar patient not found'
            most_dissimilar_patients.append(most_dissimilar_patient)
            remaining_patients = list(set(remaining_patients) - {most_dissimilar_patient})

        return most_dissimilar_patients

    def get_representative_samples(self, subset):
        """
        Returns an array of k patient mrns from subset that are most representative
        using the backend specified in the config

        Parameters
        ----------
        subset : list
            List of patient mrns to choose from

        Returns
        -------
        list : k most representative samples within subset
        """
        backend_ix = -1
        current_iteration = self.config["active_learning_iteration"]
        if self.config["representativeness"]["switch"][0] == "None":
            backend_ix = 0
        else:
            if len(self.config["representativeness"]["switch"]) != len(self.config["representativeness"]["backend"]):
                raise ValueError("The length of the backend and switch array within the representativeness parameter must be equal")
            
            if not self._check_ascending(self.config["representativeness"]["switch"]):
                raise ValueError("The switch array for representativeness parameter must be in ascending order")
            
            for i, switch_i in enumerate(self.config["representativeness"]["switch"]):
                if current_iteration >= switch_i:
                    backend_ix = i
        
        selected_backend = self.config["representativeness"]["backend"][backend_ix]
        if selected_backend == "None":
            return selected_backend, self.representativeness_none(subset)
        elif selected_backend == "cosine_similarity":
            return selected_backend, self.representativeness_cosine_similarity(subset)
        else:
            raise ValueError(selected_backend + " not recognized")       

    # pseudo labeled
    def get_samples_to_pseudo_label(self):
        """
        Returns an array of k patient mrns to use predictions as pseudo labels.
        """
        if self.config["pseudo_labels"]["incorporate"]:
            pass
        else:
            return []

# file logger
class Logger():
    def __init__(self, config):
        self.config = config
        self.iteration_number = config["active_learning_iteration"]

    def write_text_log(self, K, operation, time):
        roundtime = round(time, 3)
        logfolder = os.path.join(self.config["export_path"], self.config["unique_id"])
        logpath = os.path.join(logfolder, "log.txt")

        os.makedirs(logfolder, exist_ok = True)
        f = open(logpath, "a+")
        f.write(operation + ": " + str(K) + " (" + str(roundtime) + "s)" + "\n")
        f.close()

    def write_csv_log(self, df):
        csvfolder = os.path.join(self.config["export_path"], self.config["unique_id"], "iteration_" + \
            str(self.iteration_number))
        csvpath = os.path.join(csvfolder, "AL_groupings.csv")

        os.makedirs(csvfolder, exist_ok = True)
        df.to_csv(csvpath, index = False)
    
    def write_uncertainty_log(self, df, backend):
        csvfolder = os.path.join(self.config["export_path"], self.config["unique_id"], "iteration_" + \
            str(self.iteration_number - 1))
        csvpath = os.path.join(csvfolder, "uncertainty_" + backend + ".csv")

        os.makedirs(csvfolder, exist_ok = True)
        df.to_csv(csvpath, index = False)
    
    def write_iteration_in_txt_log(self, i):
        logfolder = os.path.join(self.config["export_path"], self.config["unique_id"])
        logpath = os.path.join(logfolder, "log.txt")

        os.makedirs(logfolder, exist_ok = True)
        f = open(logpath, "a+")
        f.write("\nIteration " + str(i) + "\n------------\n")
        f.close()


# build dataset
class Dataset_Builder():
    def __init__(self, config):
        self.config = config
        self.iteration_path = os.path.join(self.config["model_predictions_path"], "iteration_" + str(self.config["active_learning_iteration"] - 1) )

    def build_from_log(self, iteration):
        if self.config["delete_other_iterations_when_creating_new"]:
            print("Deleting data from other iterations...")
            self._delete_other_iteration_data()

        # get the names of the patients to move
        log_path = os.path.join(self.config["export_path"], self.config["unique_id"], "iteration_" + str(iteration), \
            "AL_groupings.csv")
        al_grps = pd.read_csv(log_path, dtype = str)
        annotated = list(al_grps[~al_grps["annotated"].isnull()]["annotated"])
        toannotate = list(al_grps[~al_grps["to_annotate"].isnull()]["to_annotate"])
        pseudo = list(al_grps[~al_grps["pseudo_label"].isnull()]["pseudo_label"])

        #move the files images and true labels
        datapath = os.path.join(self.config["export_path"], self.config["unique_id"], "iteration_" + str(iteration), \
            "AL_data")

        for x in tqdm(annotated + toannotate, desc="Moving True Labels and Respective Images"):
            src_patient_path = os.path.join(self.config["all_files_path"], x)
            dest_patient_path = os.path.join(datapath, x)
            
            os.makedirs(dest_patient_path, exist_ok=True)

            for img_name in self.config["file_names"]["image_names"]:
                src = os.path.join(src_patient_path, img_name)
                dest = os.path.join(dest_patient_path, img_name)
                shutil.copy(src, dest)
            
            src = os.path.join(src_patient_path, self.config["file_names"]["roi_name"])
            dest = os.path.join(dest_patient_path, self.config["file_names"]["roi_name"])
            shutil.copy(src, dest)

            src = os.path.join(src_patient_path, self.config["file_names"]["roi_name_in_organ_extraction"])
            dest = os.path.join(dest_patient_path, self.config["file_names"]["roi_name_in_organ_extraction"])
            shutil.copy(src, dest)
        
        #move the file images and pseudo labels
        for x in tqdm(pseudo, desc="Moving Pseudo Labels and Respective Images"):
            src_patient_path = os.path.join(self.config["all_files_path"], x)
            dest_patient_path = os.path.join(datapath, x)
            os.makedirs(dest_patient_path, exist_ok=True)

            for img_name in self.config["file_names"]["image_names"]:
                src = os.path.join(src_patient_path, img_name)
                dest = os.path.join(dest_patient_path, img_name)
                shutil.copy(src, dest)
            
            src = os.path.join(src_patient_path, self.config["file_names"]["roi_name"])
            dest = os.path.join(dest_patient_path, self.config["file_names"]["roi_name"])
            shutil.copy(src, dest)

            src = os.path.join(self.config['model_predictions_path'], 'iteration_' + str(iteration), x, self.config["file_names"]["prediction_name"])
            dest = os.path.join(dest_patient_path, self.config["file_names"]["roi_name_in_organ_extraction"])
            shutil.copy(src, dest)

        # check if bootstrapping
        backend_ix = -1
        current_iteration = self.config["active_learning_iteration"]
        if self.config["uncertainty"]["switch"][0] == "None":
            backend_ix = 0
        else:
            if len(self.config["uncertainty"]["switch"]) != len(self.config["uncertainty"]["backend"]):
                raise ValueError("The length of the backend and switch array within the uncertainty parameter must be equal")
            
            if not self._check_ascending(self.config["uncertainty"]["switch"]):
                raise ValueError("The switch array for uncertainty parameter must be in ascending order")
            
            for i, switch_i in enumerate(self.config["uncertainty"]["switch"]):
                if current_iteration >= switch_i:
                    backend_ix = i
        
        # if bootstrapping, need to create bootstrapped data paths
        # else just split into train/val sets
        if self.config["uncertainty"]["backend"][backend_ix] != "bootstrapped":
            #split data into train/val sets
            print("Splitting data into train/val splits")
            self._train_val_split(datapath)
        else:
            bootstrapped_paths = [os.path.join(self.config["export_path"], self.config["unique_id"], "iteration_" + str(iteration), \
                "AL_data", "bootstrapped_" + str(i)) for i in range(N_BOOTSTRAPPED_MODELS)]

            train_patients = []
            val_patients = []

            #create train val splits for each bootstrapped model
            for i in range(N_BOOTSTRAPPED_MODELS):
                t_pt, v_pt = self._get_train_val_patients(datapath, self.config["random_seed"] + (i * 500))
                train_patients.append(t_pt)
                val_patients.append(v_pt)

            # bootstrap each split for each bootstrapped model
            for i, (x, t_pt, v_pt) in enumerate(zip(bootstrapped_paths, train_patients, val_patients)):
                bootstrapped_train_dir = os.path.join(x, "Train")
                bootstrapped_val_dir = os.path.join(x, "Val")

                os.makedirs(bootstrapped_train_dir, exist_ok=False)
                os.makedirs(bootstrapped_val_dir, exist_ok = False)

                bootstrapped_train_pts = resample(t_pt, replace=True, n_samples = len(t_pt), random_state = self.config["random_seed"] + (i * 500))
                bootstrapped_val_pts = resample(v_pt, replace=True, n_samples = len(v_pt), random_state = self.config["random_seed"] + (i * 500))

                already_included = dict()

                for bootstrapped_train_pt in bootstrapped_train_pts:
                    if bootstrapped_train_pt not in already_included:
                        shutil.copytree(os.path.join(datapath, bootstrapped_train_pt), os.path.join(bootstrapped_train_dir, bootstrapped_train_pt))
                        already_included[bootstrapped_train_pt] = 1
                    else:
                        shutil.copytree(os.path.join(datapath, bootstrapped_train_pt), os.path.join(bootstrapped_train_dir, bootstrapped_train_pt + "_" + str(already_included[bootstrapped_train_pt])))
                        already_included[bootstrapped_train_pt] += 1
                
                for bootstrapped_val_pt in bootstrapped_val_pts:
                    if bootstrapped_val_pt not in already_included:
                        shutil.copytree(os.path.join(datapath, bootstrapped_val_pt), os.path.join(bootstrapped_val_dir, bootstrapped_val_pt))
                        already_included[bootstrapped_val_pt] = 1
                    else:
                        shutil.copytree(os.path.join(datapath, bootstrapped_val_pt), os.path.join(bootstrapped_val_dir, bootstrapped_val_pt + "_" + str(already_included[bootstrapped_val_pt])))
                        already_included[bootstrapped_val_pt] += 1
            
            # remove the full training set and only keep the bootstrapped versions
            for x in os.listdir(datapath):
                if x not in ["bootstrapped_" + str(i) for i in range(N_BOOTSTRAPPED_MODELS)]:
                    shutil.rmtree(os.path.join(datapath, x))

        print("Data build from log at")
        print("\t" + datapath)

    def build_next_iteration(self):
        logger = Logger(self.config)
        learner = ActiveLearner(self.config)
        logger.write_iteration_in_txt_log(self.config["active_learning_iteration"])

        # first iteration will just create an initial dataset
        total_initial_time = time.time()

        if self.config["active_learning_iteration"] == 0:
            initial_time = time.time()
            initial_training_dataset = learner.initial_training_dataset()
            end_time = time.time()
            logger.write_text_log(len(initial_training_dataset), "Created initial training set with " + self.config["initial_dataset_generator"]["backend"] + " backend" , end_time - initial_time)

            df = pd.DataFrame()
            df["initial"] = initial_training_dataset
            df["to_annotate"] = initial_training_dataset

            empty_headers = ["annotated", "uncertain", "representative", "pseudo_label"]
            for x in empty_headers:
                df[x] = [np.nan for i in range(len(initial_training_dataset))]
            
            logger.write_csv_log(df)
        # subsequent iterations will first select K uncertain samples, k representative samples
        # from the uncertain subset. Pseudo labels may be incorporated
        else:
            # get the uncertain samples
            initial_time = time.time()
            backend, uncertain_samples = learner.get_uncertain_samples()
            end_time = time.time()
            logger.write_text_log(len(uncertain_samples), "Found uncertain samples with " + backend + " backend", end_time - initial_time)

            # get the representative samples
            initial_time = time.time()
            backend, representative_samples = learner.get_representative_samples(uncertain_samples)
            end_time = time.time()
            logger.write_text_log(len(representative_samples), "Found representative samples with " + backend + " backend", end_time - initial_time)

            # get the pseudo label samples
            initial_time = time.time()
            pseudo_labels = learner.get_samples_to_pseudo_label()
            end_time = time.time()
            logger.write_text_log(len(pseudo_labels), "Found samples to pseudo label", end_time - initial_time)

            # log the results
            df = pd.DataFrame()
            annotated = learner.get_annotated_files()
            toannotate = representative_samples.copy()
            initial = learner.get_initial_dataset_from_log()
            all_arrays = [uncertain_samples, representative_samples, pseudo_labels, annotated, toannotate, initial]
            largest_length = np.max([len(x) for x in all_arrays])
            
            all_arrays = [np.concatenate((x, np.empty(largest_length - len(x)) * np.nan)) for x in all_arrays] #make sure all the arrays are same size
            
            headers = ["uncertain", "representative", "pseudo_label", "annotated", "to_annotate", "initial"]
            for data, header in zip(all_arrays, headers):
                df[header] = data
            
            logger.write_csv_log(df)
        
        total_end_time = time.time()
        logger.write_text_log("", "Total time to build log", total_end_time - total_initial_time)

        #update the iteration and save the config file in the data file
        new_config_path = os.path.join(self.config["export_path"], self.config["unique_id"], "config.yaml")
        f = open(new_config_path, "w+")
        new_config = self.config.copy()
        new_config["active_learning_iteration"] += 1
        yaml.dump(new_config, f)
        f.close()

        output_dir = os.path.join(self.config["export_path"], self.config["unique_id"])
        tomove = ["helper_setup.py", "initial_setup.py"]
        for x in tomove:
            shutil.copy(x, os.path.join(output_dir, "temp_" + x))
            if os.path.exists(os.path.join(output_dir, x)):
                os.remove(os.path.join(output_dir, x))
            shutil.move(os.path.join(output_dir, "temp_" + x), os.path.join(output_dir, x))

    def _delete_other_iteration_data(self):
        todelete = []
        for root, dir, files in os.walk(self.config["export_path"]):
            if "AL_data" in dir:
                todelete.append(os.path.join(root, "AL_data"))
        
        for x in todelete:
            shutil.rmtree(x)
    
    def _train_val_split(self, directory):
        train_patients, val_patients = self._get_train_val_patients(directory, self.config["random_seed"])

        os.makedirs(os.path.join(directory, "Train"))
        os.makedirs(os.path.join(directory, "Val"))
        for x in train_patients:
            src = os.path.join(directory, x)
            dest = os.path.join(directory, "Train", x)
            shutil.move(src, dest)
        for x in val_patients:
            src = os.path.join(directory, x)
            dest = os.path.join(directory, "Val", x)
            shutil.move(src, dest)
    
    def _get_train_val_patients(self, directory, seed):
        random.seed(seed)
        np.random.seed(seed)

        patients = os.listdir(directory)
        np.random.shuffle(patients)
        train_ix_cutoff = int(len(patients) * self.config["train_dataset_percentage"])
        train_patients = patients[:train_ix_cutoff]
        val_patients = patients[train_ix_cutoff:]

        return train_patients, val_patients

