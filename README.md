# ML - DeFungi Dataset - Hiram Israel Mendoza López / A01613963

## Project Description

This project aims to prepare an image dataset for a future classification task using machine learning techniques.

The work focuses on the **DeFungi** dataset, which contains microscopic fungi images classified into different categories. At this stage, the initial data preparation process is performed.

Through this process, the project focuses on:

- selecting an image dataset
- splitting data into training and testing sets
- image preprocessing
- pixel value scaling
- data augmentation
- organizing the dataset for future use with TensorFlow/Keras

## Dataset Description

This project uses the **DeFungi** dataset, available in the UCI Machine Learning Repository:

[DeFungi Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/773/defungi)

The dataset contains:

- approximately 9,114 images
- 5 classes
- image-format data
- labels organized by folders

The dataset classes are:

| Class | Code | Full Name | Description |
|---|---|---|---|
| H1 | TSH | Tortuous Septate Hyaline Hyphae | Transparent, twisted fungal filaments with cross-walls |
| H2 | BASH | Beaded Arthroconidial Septate Hyaline Hyphae | Transparent filaments with bead-like spore chains |
| H3 | GMA | Groups or Mosaics of Arthroconidia | Clustered groups of fungal spores |
| H5 | SHC | Septate Hyaline Hyphae with Chlamydoconidia | Transparent filaments with thick-walled survival spores |
| H6 | BBH | Broad Brown Hyphae | Wide, brown-pigmented fungal filaments |

The original dataset structure was:

```text
defungi/
  H1/
  H2/
  H3/
  H5/
  H6/
```

Each folder represents a class and works as the label for the images it contains.

## Process

#### Dataset Selection

The **DeFungi** dataset was selected because it allows working with real images and applying preprocessing and data augmentation techniques.

#### Dataset Storage

Due to the number of image files, the dataset was stored in Google Drive instead of being uploaded directly to the GitHub repository.

The original dataset was stored in:

```text
MyDrive/TC3002/defungi
```

After the train/test split, the generated dataset was stored in:

```text
MyDrive/TC3002/defungi_separation
```

This allowed the dataset to be accessed from Google Colab while keeping the GitHub repository lightweight and focused on the notebook and documentation.

#### Training and Testing Set Split

The dataset was divided into two subsets:

- `training` → used to prepare the training data
- `test` → used as the testing set

The split was performed using the following proportion:

```text
80% training x 20% testing
```

To perform this split, each class folder was iterated through, valid image files were selected, the files were randomly shuffled, and then they were copied into a new folder structure.

A fixed random seed was also used with `random.seed(10)` to make the split reproducible.

The new generated structure was:

```text
defungi_separation/
  training/
    H1/
    H2/
    H3/
    H5/
    H6/
  test/
    H1/
    H2/
    H3/
    H5/
    H6/
```

#### Data Preprocessing

After generating the split, the images were loaded using TensorFlow/Keras `ImageDataGenerator`.

The preprocessing included:

- loading images from folders
- automatically assigning labels based on the folder name
- resizing images to 128x128 pixels
- grouping images into batches of 32

Resizing is necessary because machine learning models require all images to have the same input size.

#### Data Scaling

For image scaling, the following value was used:

```python
rescale=1./255
```

This technique converts pixel values from the original `0-255` range to the `0-1` range.

Scaling was applied to both the training and testing sets.

#### Data Augmentation

To increase the variety of the training set, data augmentation was applied.

The techniques used were:

| Technique | Description |
|---|---|
| `rotation_range=10` | Slightly rotates the images |
| `zoom_range=0.1` | Applies a small zoom variation |
| `width_shift_range=0.1` | Shifts the image horizontally |
| `height_shift_range=0.1` | Shifts the image vertically |
| `horizontal_flip=True` | Flips the image horizontally |

These transformations generate variations of the original images without changing their class.

Data augmentation was applied only to the training set.

#### Test Set Preprocessing

The test set did not receive data augmentation.

Only scaling with `rescale=1./255` was applied, since the test set should represent unseen data and should not be artificially modified.

#### Loading Data with Keras

Keras automatically detected the classes based on the folder names.

The detected classes were:

```python
{'H1': 0, 'H2': 1, 'H3': 2, 'H5': 3, 'H6': 4}
```

This means each class was converted into a numeric label so it can be used in future machine learning processes.

#### Visualization of Augmented Images

Finally, a group of images from the training generator was visualized to confirm that the images were correctly loaded and that the data augmentation transformations were being applied.

This visualization made it possible to verify that:

- the images are loaded correctly
- the labels are assigned according to their folder
- the training set receives transformations
- the test set remains without augmentation


## Tools Used

- Python
- Google Colab
- Google Drive
- TensorFlow/Keras
- Matplotlib
- pathlib
- random
- shutil

---

## Phase 2: Model Implementation and Initial Model Evaluation

#### Model Selection

A **Convolutional Neural Network (CNN)** was selected for this classification task. The selection is based on the state-of-the-art article by Korkmaz et al. (2026) [1], which demonstrates that CNN architectures achieve the best results for fungi image classification, reaching up to 93% accuracy with a single model. Following the same principle used in the paper, convolutional layers for feature extraction followed by dense layers for classification, a custom CNN was implemented from scratch using TensorFlow/Keras.

#### Model Architecture

The model is composed of 3 convolutional blocks followed by dense layers:

<img width="495" height="297" alt="Captura de pantalla 2026-05-31 a la(s) 7 24 45 p m" src="https://github.com/user-attachments/assets/bd768b10-228c-4965-8493-6aa1b5e5b847" />

- **Conv2D(16)** → detects simple patterns such as edges and colors in the images
- **Conv2D(32)** → detects more complex patterns by combining the previous ones
- **Conv2D(64)** → detects high-level features such as shapes and textures of the fungi
- **MaxPooling2D** → reduces the spatial dimensions after each convolutional block
- **MaxPooling2D (extra)** → additional reduction before flattening to keep the parameter count manageable
- **Flatten** → converts the 2D output into a 1D vector
- **Dense(64)** → combines all learned features for classification
- **Dropout(0.5)** → randomly disables 50% of neurons during training to reduce overfitting
- **Dense(5, softmax)** → outputs the probability for each of the 5 classes

The total number of trainable parameters is **286,117 (1.09 MB)**, which is intentionally kept small for this initial model to allow faster training times.

**Parameters:**

| Parameter | Value | Description |
|---|---|---|
| Optimizer | Adam | Adjusts weights by adapting the learning rate automatically |
| Loss function | Sparse Categorical Crossentropy | Calculates the error between the prediction and the real class |
| Metric | Accuracy | Used to monitor training and detect overfitting by comparing train vs validation |

#### Training

The model was trained for 10 epochs using the training generator, with the test generator used as validation data. The dataset was copied from Google Drive to Colab's local storage before training to avoid slow file I/O from the cloud, which significantly improved training speed.

<img width="773" height="268" alt="Captura de pantalla 2026-05-31 a la(s) 7 25 11 p m" src="https://github.com/user-attachments/assets/5ffa960d-129e-4a1b-b655-02bd8dce7bdc" />
<br>&nbsp;<br>
<img width="825" height="288" alt="Captura de pantalla 2026-05-31 a la(s) 7 25 18 p m" src="https://github.com/user-attachments/assets/1fe38314-01ec-42c3-a5f2-b7a83c8efc01" />

The training curves show that:

- The training accuracy steadily increased from ~52% to ~62% over the 10 epochs
- The validation accuracy started higher (~62%) and remained relatively stable throughout, ending at ~63%
- Both the training loss and validation loss decreased consistently, with no significant gap between them

#### Evaluation Metrics

The metrics used are based on those reported in Korkmaz et al. (2026) [1] for evaluating fungi classification models. Accuracy is used only to verify that there is no overfitting by comparing train and validation values. Precision, recall, and F1-score are the main metrics used to evaluate the actual performance of the model.

| Metric | Description |
|---|---|
| Precision | Indicates how reliable the positive predictions are for a given class |
| Recall | Indicates how many actual cases of a class the model correctly identified |
| F1-Score | Balance between precision and recall; especially important with class imbalance |

A confusion matrix was also included to see the breakdown of correct and incorrect predictions per class (TP, TN, FP, FN).

#### Results

<img width="514" height="59" alt="Captura de pantalla 2026-05-31 a la(s) 7 30 07 p m" src="https://github.com/user-attachments/assets/408fce60-3ba4-463c-84dc-03a9f0759b23" />
<br>&nbsp;<br>
<img width="514" height="165" alt="Captura de pantalla 2026-05-31 a la(s) 7 30 42 p m" src="https://github.com/user-attachments/assets/8d4d3cda-d291-4fae-bd6d-f9aaa9016279" />
<br>&nbsp;<br>
<img width="550" height="438" alt="Captura de pantalla 2026-05-31 a la(s) 7 31 28 p m" src="https://github.com/user-attachments/assets/7643f02e-046a-4db9-833f-e558eb73c5a5" />

The `model.evaluate` method reports a test accuracy of **63.21%**. However, the actual accuracy calculated from the classification report and confusion matrix is **39%** (718 correct out of 1,824 total predictions). After investigating this difference, it was found that `model.evaluate` with data generators does not always reflect the real accuracy over the full dataset, so the classification report and confusion matrix were used as the primary source for evaluating the model's actual performance.

#### Observations

- The model does not show signs of overfitting since the train and validation accuracy values remained close throughout all 10 epochs (~62% vs ~63%)
- The real test accuracy is **39%**, as confirmed by the confusion matrix (718/1,824 correct predictions)
- The model developed a strong bias toward class **H1**, predicting it for the majority of samples regardless of their true class. This is a direct consequence of the class imbalance in the dataset, where H1 has 4,404 training images compared to only 739 for H6
- The classes with the least training data (H3, H5, H6) have F1-scores below 0.10, confirming that the model was unable to learn these minority classes effectively
- H1 achieved the best performance with a recall of 0.74 and an F1-score of 0.58, since it has by far the most training examples

---

## Phase 3: Model Refinement

#### Baseline (Phase 2)

Before making any changes, the performance of the initial model was measured to establish a clear baseline for comparison:

| Metric | Value |
|---|---|
| Test Accuracy | 39% |
| Weighted F1-Score | 0.32 |
| H1 F1-Score | 0.58 |
| H2 F1-Score | 0.09 |
| H3 F1-Score | 0.07 |
| H5 F1-Score | 0.08 |
| H6 F1-Score | 0.07 |

The feedback from Phase 2 confirmed that the model was clearly in **underfit**, and the confusion matrix showed a strong bias toward class H1, predicting it for the majority of samples regardless of their true class.

#### Problems Identified and Adjustments Made

Three main problems were identified from the Phase 2 results, and a specific adjustment was made for each one:

**Problem 1: Class imbalance → class weights**

The model learned to predict H1 for almost everything because H1 has 4,404 training images while H6 only has 591. Class weights were computed using sklearn's `compute_class_weight` with `class_weight='balanced'` and added to the training process. This penalizes the model more when it misclassifies minority classes, forcing it to learn patterns from all 5 classes equally.

The resulting weights were:

<img width="103" height="85" alt="Captura de pantalla 2026-06-03 a la(s) 11 27 20 a m" src="https://github.com/user-attachments/assets/501a12ca-a330-47e6-8284-03f35203cad6" />


**Problem 2: Underfitting → larger architecture**

The previous model (16→32→64 filters, Dense(64)) was too small to learn the visual differences between 5 fungi classes. The number of filters was increased to 32→64→128 and the dense layer to Dense(128) to give the model more learning capacity.

**Problem 3: Too much regularization → lower Dropout**

Dropout(0.5) was too aggressive for a model that was already underfitting, as it discarded half of the learned information at each step. It was reduced to Dropout(0.3).

**Additional: Learning rate adjustment**

When first trained with class weights using the default learning rate (0.001), the model failed to converge — the training accuracy fluctuated between 10-24% over 12 epochs, worse than random guessing. After investigation, the learning rate was reduced to 0.0001 to allow the model to take smaller, more stable steps when navigating the more complex loss landscape created by the class weights. This was the key change that allowed the model to learn effectively.

The full comparison between the two model configurations is:

| | Model v1 (Phase 2) | Model v2 (Phase 3) |
|---|---|---|
| Filters | 16 → 32 → 64 | 32 → 64 → 128 |
| Dense layer | Dense(64) | Dense(128) |
| Dropout | 0.5 | 0.3 |
| Learning rate | 0.001 (default) | 0.0001 |
| Epochs | 10 | 20 (EarlyStopping) |
| Class weights | No | Yes |

#### Results

<img width="491" height="339" alt="Captura de pantalla 2026-06-03 a la(s) 11 27 47 a m" src="https://github.com/user-attachments/assets/a2d5bd9b-29cb-4168-ab06-ee633c0ceba5" />

The refined model has **1,142,597 trainable parameters (4.36 MB)**, a significant increase from the 286,117 of the initial model, giving it more capacity to learn the visual differences between the 5 fungi classes.


<img width="755" height="453" alt="Captura de pantalla 2026-06-03 a la(s) 11 28 02 a m" src="https://github.com/user-attachments/assets/6f9abce1-7830-4bb7-99fb-d28fc07019b4" />

The model trained for 17 epochs before EarlyStopping triggered, restoring the best weights automatically. The training accuracy increased steadily from ~22% to ~57%, confirming that the model was learning progressively with the class weights applied.

<img width="1052" height="378" alt="Captura de pantalla 2026-06-03 a la(s) 11 28 16 a m" src="https://github.com/user-attachments/assets/211676d9-14cb-4b90-b790-fb68b0362277" />

The curves show both train and validation improving consistently over epochs with no significant divergence between them, indicating a balanced model with no overfitting.


<img width="546" height="195" alt="Captura de pantalla 2026-06-03 a la(s) 11 28 32 a m" src="https://github.com/user-attachments/assets/b495a766-c9f5-4e03-aa65-4454e388cc25" />

The `model.evaluate` result of **63.16%** is now reliable since `shuffle=False` was correctly set in the test generator, meaning the predictions and true labels are in the same order.


<img width="351" height="167" alt="Captura de pantalla 2026-06-03 a la(s) 11 28 48 a m" src="https://github.com/user-attachments/assets/4daf4f00-57ba-4d63-9e8d-9c238cd58ee3" />

The classification report shows a significant improvement across all classes compared to Phase 2, with the minority classes (H5, H6) achieving F1-scores of 0.60 and 0.70 respectively, up from below 0.10.


<img width="550" height="438" alt="Captura de pantalla 2026-06-03 a la(s) 11 29 01 a m" src="https://github.com/user-attachments/assets/aabb1ac1-f9f9-46bb-8346-210b67341cb5" />

The confusion matrix confirms that the model is no longer collapsing to predict H1 for everything. Each class now has a visible diagonal value, showing the model learned distinguishing features for all 5 fungi types.


#### Comparison with Phase 2

| Metric | Model v1 (Phase 2) | Model v2 (Phase 3) | Change |
|---|---|---|---|
| Accuracy | 39% | **63%** | +24% |
| Weighted F1-Score | 0.32 | **0.60** | +0.28 |
| H1 F1-Score | 0.58 | 0.78 | +0.20 |
| H2 F1-Score | 0.09 | 0.32 | +0.23 |
| H3 F1-Score | 0.07 | 0.30 | +0.23 |
| H5 F1-Score | 0.08 | 0.60 | +0.52 |
| H6 F1-Score | 0.07 | 0.70 | +0.63 |


The comparison table summarizes the improvement across all metrics between the initial model (Phase 2) and the refined model (Phase 3).


#### Observations

- Adding class weights was the most impactful change — the minority classes (H3, H5, H6) went from F1-scores below 0.10 to 0.30, 0.60, and 0.70 respectively. H6 showed the most dramatic improvement, going from 0.07 to 0.70, a 10x increase
- The key to making class weights work was reducing the learning rate from 0.001 to 0.0001. With the default learning rate the model failed to converge over 12 epochs (train accuracy 10-24%). The lower learning rate allowed the model to take more stable steps with the heavier class penalties
- The training curves show a model with no signs of overfitting — the validation accuracy remained close to training accuracy throughout all epochs. The oscillations in validation accuracy are expected behavior when class weights create a more complex loss landscape
- EarlyStopping triggered automatically, restoring the best weights from the epoch with the lowest validation loss
- H2 remains one of the most difficult class with an F1-score of 0.32, likely due to its visual similarity with other classes combined with its intermediate size in the dataset
- The confusion matrix is significantly more balanced compared to Phase 2 — the model no longer collapses to predicting H1 for everything, and correctly identifies the majority of H1 (792/881) and H6 (118/148) samples
- The gap between the refined model (63%) and the state of the art reported in the reference paper (93%) suggests that a future improvement using transfer learning with pretrained architectures such as MobileNetV2 or EfficientNet would be the next step

## References

- [1] Korkmaz, A. F., Ekinci, F., Kumru, E., Aydogan, A., Kaymak, H. S., Sevindik, M., Guzel, M. S., & Akata, I. (2026). Explainable deep learning ensemble framework for accurate classification of wild poisonous mushroom species. *BMC Biotechnology*, 26, 11. https://doi.org/10.1186/s12896-025-01092-z


