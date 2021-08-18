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

# IO and file setup
class ActiveLearner():
    def __init__(self, config):
        self.config = config
    
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
            return list(df["annotated"]) + list(df["to_annotate"])
    
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
            return [x for x in all_patients if x not in annotated]

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
            return list(df["initial"])
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
        np.random.seed(self.config["random_seed"])
        random.seed(self.config["random_seed"])
        all_patients = self.get_all_files()
        np.random.shuffle(all_patients)

        return all_patients[:self.config["initial_dataset_generator"]["m"]]

    def SIFT3D_initialization(self):
        """
        Returns an array of m patient mrns that were identified to be most
        different from each other using the SIFT3D algorithm. The principle
        behind this is to create a well representative initial dataset using
        computer vision.

        Returns
        -------
            list: patient mrns that were identified with SIFT3D
        """
        pass

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
        elif self.config["initial_dataset_generator"]["backend"] == "SIFT3D":
            return self.SIFT3D_initialization()
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
        return self.get_x_random_unannotated_files(self.config["uncertainty"]["K"], seed = self.config["random_seed"])
        
    def uncertainty_bootstrapped(self):
        """
        Returns an array of K patient mrns with the most variance across the
        predictions from the bootstrapped models

        Returns
        -------
            list: most uncertain patient mrns derived from bootstrapped predictions
        """
        pass 

    def uncertainty_prob_roi(self):
        """"
        Returns an array of K patient mrns with the lowest mean probability at the
        ROI.

        Returns
        -------
            list: most uncertain patient mrns derived from mean probability of ROI
        """
        pass 

    def uncertainity_margin(self):
        """
        Returns an array of K patient mrns with the smallest margin in probability
        between the foreground and background.

        Returns
        -------
            list: most uncertain patient mrns derived from margin sampling
        """
        pass 

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
            return selected_backend, self.uncertainity_margin()
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

    def representativeness_cosine_similarity(self, subset):
        """
        Returns an array of k patient mrns from subset that are most representative
        using the cosine similarity function

        Parameters
        ----------
        subset : list
            List of patient mrns to choose from 

        Returns
        -------
        list : k most representative sampels within subset using the cosine
            similarity backend
        """
        pass

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

    def build_from_log(self, iteration):
        print("Deleting data from other iterations...")
        if self.config["delete_other_iterations_when_creating_new"]:
            self._delete_other_iteration_data()
        log_path = os.path.join(self.config["export_path"], self.config["unique_id"], "iteration_" + str(iteration), \
            "AL_groupings.csv")
        al_grps = pd.read_csv(log_path, dtype = str)
        annotated = al_grps["annotated"]
        toannotate = al_grps["to_annotate"]
        pseudo = al_grps["pseudo_label"]

        annotated = [str(x) for x in annotated if x == x]
        toannotate = [str(x) for x in toannotate if x == x]
        pseudo = [str(x) for x in pseudo if x == x]

        datapath = os.path.join(self.config["export_path"], self.config["unique_id"], "iteration_" + str(iteration), \
            "AL_data")

        #move the files images and true labels
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

            src = os.path.join(self.config["model_predictions_path"], x, self.config["file_names"]["prediction_name"])
            dest = os.path.join(dest_patient_path, self.config["file_names"]["roi_name_in_organ_extraction"])
            shutil.copy(src, dest)
        
        #split data into train/val sets
        print("Splitting data into train/val splits")
        self._train_val_split(datapath)

        print("Data build from log at")
        print("\t" + datapath)

    def build_next_iteration(self):
        logger = Logger(self.config)
        learner = ActiveLearner(self.config)
        logger.write_iteration_in_txt_log(self.config["active_learning_iteration"])

        # first iteration will just create an initial dataset
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
            
            for i in range(len(all_arrays)):
                all_arrays[i] = [x for x in all_arrays[i] if x == x]

            for x in all_arrays:
                while len(x) != largest_length:
                    x.append(np.nan)
            
            headers = ["uncertain", "representative", "pseudo_label", "annotated", "to_annotate", "initial"]
            for data, header in zip(all_arrays, headers):
                df[header] = data
            
            logger.write_csv_log(df)

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
        random.seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])

        patients = os.listdir(directory)
        np.random.shuffle(patients)
        train_ix_cutoff = int(len(patients) * self.config["train_dataset_percentage"])
        train_patients = patients[:train_ix_cutoff]
        val_patients = patients[train_ix_cutoff:]

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

