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

| Class | Description |
|---|---|
| H1 | Class 1 of microscopic fungi images |
| H2 | Class 2 of microscopic fungi images |
| H3 | Class 3 of microscopic fungi images |
| H5 | Class 4 of microscopic fungi images |
| H6 | Class 5 of microscopic fungi images |

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

The **DeFungi** dataset was selected because it allows working with real images and applying preprocessing and data augmentation 
techniques.

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

