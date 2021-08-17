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
            pass 

    def get_x_random_unannotated_files(self, x):
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
        pass


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
        This is the identical to if active learning did not select for uncertain
        samples

        Returns
        -------
            list: patient mrns randomly selected
        """
        pass 

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
        pass 

    # representativeness
    def representativeness_none(subset):
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
        pass

    def representativeness_cosine_similarity(subset):
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

    def get_representative_samples(subset):
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
        pass 

    # pseudo labeled
    def get_samples_to_pseudo_label(config):
        """
        Returns an array of k patient mrns to use predictions as pseudo labels.
        """
        pass 

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
        else:
            pass

        

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

