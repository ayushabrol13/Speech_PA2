# Speech Understanding - Programming Assignment 2

The readme file provides a comprehensive description of the code and includes instructions on how to execute each file to successfully reproduce the given findings. (includes Speaker Verification and Source Separation)

### Problem Statements

#### Question 1

**Goal**:

    In speaker verification, the training dataset consists of audio clips paired with speaker IDs, denoted as (D = (xi, yi)). Given an audio clip (x) and a reference clip (x0), the objective is to ascertain whether (x0) and (x) belong to the same speaker.

**Tasks**:

    1. Choose three pre-trained models from the list: ’ecapa tdnn’, ’hubert large’, ’wav2vec2 xlsr’, ’unispeech sat’, ’wavlm base plus’, ’wavlm large’ trained on the VoxCeleb1 dataset. You can find the pre-trained models on this link.

    2. —- Calculate the EER(%) on the VoxCeleb1-H dataset using the above selected models. You can get the
    dataset from here.

    3. Compare your result with Table II of the WavLM paper.

    4. Evaluate the selected models on the test set of any one Indian language of the Kathbath Dataset. Report the EER(%).

    5. Fine-tune, the best model on the validation set of the selected language of Kathbath Dataset. Report the EER(%).

    6. Provide an analysis of the results along with plausible reasons for the observed outcomes.

#### Question 2

**Goal**:

    The goal of speech separation is to estimate individual speaker signals from their mixture, where the source signals may be overlapped with each other entirely or partially.

**Tasks**:

    1. Generate the LibriMix dataset by combining two speakers from the LibriSpeech dataset, focusing solely on the LibriSpeech test clean partition. Take help from this GitHub repo.

    2. Partition the resulting LibriMix dataset into a 70-30 split for training and testing purposes. Evaluate the performance of the pre-trained SepFormer on the testing set, employing scale-invariant signal-to-noise ratio improvement (SISNRi) and signal-to-distortion ratio improvement (SDRi) as metrics. For metric computation, consult the provided paper and utilize the code from torchmetrics.

    3. Fine-tune the SepFormer model using the training set and report its performance on the test split of the LibriMix dataset.

    4. Provide observations on the changes in performance throughout the experiment.

### Env

Prior to running the supplied code, it is essential to have an environment that has all the necessary packages installed. As a result, I have included a file called _'packages.txt'_ that lists the names and versions of all the packages used in this programming assignment. These packages are essential for reproducing the findings.

Furthermore, it is important to acknowledge that the programming assignment requires a significant amount of processing power and storage capacity. This indicates that it need powerful GPU resources and substantial storage capacity for data, models, and other related components. Therefore, it is imperative that you execute the given code files exclusively on a system of this nature.

### Pre-requisites

#### Data-wise pre-requisites

Now, for each question the required data must be downloaded and located in the specified path.

- For question 01, two datasets are required. First is the VoxCeleb1-H dataset, which should be downloaded and stored under the path 'root/voxceleb1' and its metadata text file as 'root/VoxCelebVerification.txt'. Second is the Kathbath dataset for which 'hindi' language data should be downloaded and stored under tha path 'root/KathBath/hindi' and its metadata as 'root/KathBath_MetaData/hindi'.
- For question 02, the required generated data is stored under the path 'Q2/LibriMix/storage_dir/Libri2Mix/wav16k/max/test'.
