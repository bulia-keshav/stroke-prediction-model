# Comprehensive Analysis: Stroke Prediction and mRS Classification System

## Executive Summary

This document provides a detailed analysis of a comprehensive stroke prediction and modified Rankin Scale (mRS) classification system developed for clinical decision-making in stroke patients. The system integrates deep learning-based stroke volume estimation with machine learning models for functional outcome prediction.

---

## System Architecture Overview

The system consists of four main components organized in a modular structure:

```
Scripts/
├── Preprocessing/          # Data preparation and normalization
├── UNET/                  # Deep learning segmentation model
├── Prediction/            # Volume estimation and inference
└── MRS Classification/    # Outcome prediction models
```

---

## 1. Preprocessing Pipeline (`Scripts/Preprocessing/`)

### **Core Components:**
- **`preprocessing.ipynb`**: Main preprocessing pipeline
- **`move.ipynb`**: Data organization and file management

### **Technical Implementation:**

#### **Image Normalization:**
```python
# Hounsfield Unit standardization for CT images
HOUNSFIELD_MAX = 1300
HOUNSFIELD_MIN = 0
HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

def normalizeImageIntensityRange(img):
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE
```

#### **Multi-planar Slice Extraction:**
- **Slice Orientation Control**: X, Y, Z plane selection
- **Slice Decimation**: Every 3rd slice extraction to reduce computational overhead
- **Format Support**: NIfTI (.nii) and NRRD (.nrrd) file handling

#### **Segmentation Mask Processing:**
- **SlicerIO Integration**: Advanced segmentation file handling
- **Multi-segment Support**: Individual segment extraction and labeling
- **Binary Mask Generation**: Conversion to binary format for training

---

## 2. U-Net Deep Learning Architecture (`Scripts/UNET/`)

### **Core Components:**
- **`training.ipynb`**: Complete training pipeline
- **`unet.py`**: Model architecture definition
- **Trained Models**: `UNET_nd_dl.h5`, `UNETv1.h5`

### **Architecture Specifications:**

#### **U-Net Model Design:**
```python
def unet(n_levels=4, initial_features=32, n_blocks=2, 
         kernel_size=3, pooling_size=2, in_channels=1, out_channels=1):
    # Encoder (Contracting Path)
    - 4 levels of downsampling
    - Feature maps: 32 → 64 → 128 → 256
    - MaxPooling2D for dimensionality reduction
    
    # Decoder (Expanding Path)  
    - Skip connections for feature preservation
    - Conv2DTranspose for upsampling
    - Concatenation layers for multi-scale features
    
    # Output Layer
    - Sigmoid activation for binary segmentation
```

#### **Training Configuration:**
- **Input Size**: 128×128 grayscale images
- **Batch Size**: 32 (training), 32 (validation)
- **Epochs**: 50
- **Dataset**: 2,502 training images, 575 test images
- **Augmentation**: Rescaling (1./255)

#### **Loss Functions and Metrics:**
```python
# Combined Loss Function
def combined_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

# Evaluation Metrics
- IoU Score (Intersection over Union)
- F1 Score
- Dice Coefficient
```

---

## 3. Prediction and Volume Estimation (`Scripts/Prediction/`)

### **Core Components:**
- **`Prediction.ipynb`**: Complete inference pipeline
- **`Voxel2Volume.py`**: Volume calculation utilities
- **Model Files**: Multiple trained U-Net variants

### **Technical Implementation:**

#### **Multi-planar Prediction:**
```python
def predictVolume(inImg, toBin=True):
    # Process all three anatomical planes
    if SLICE_Z:  # Axial slices
        for i in range(zMax):
            img = scaleImg(inImg[:,:,i], IMAGE_HEIGHT, IMAGE_WIDTH)
            prediction = model.predict(img)
            outImgZ[:,:,i] = scaleImg(prediction, xMax, yMax)
    
    # Ensemble averaging across planes
    outImg = (outImgX + outImgY + outImgZ) / count_planes
    
    # Binary thresholding
    if toBin:
        outImg[outImg > 0.45] = 1.0
        outImg[outImg <= 0.45] = 0.0
```

#### **Volume Calculation:**
```python
def voxel2volume(nifti_file, mask_array):
    # Extract voxel dimensions from NIfTI header
    dim = nifti_file.header["pixdim"][1:4]
    per_voxel_volume_cm3 = (dim[0] * dim[1] * dim[2]) / 1000
    total_volume = mask_array.sum() * per_voxel_volume_cm3
    return round(total_volume, 4)
```

#### **Image Processing Pipeline:**
1. **Hounsfield Unit Normalization**: [150, 1300] HU range
2. **Slice-by-slice Prediction**: 128×128 resolution
3. **Multi-planar Ensemble**: Combines X, Y, Z predictions
4. **Volume Quantification**: Converts voxel count to milliliters

---

## 4. mRS Classification System (`Scripts/MRS Classification/`)

### **System Overview:**
Comprehensive machine learning pipeline for predicting functional outcomes using multiple approaches and datasets.

### **Folder Structure Analysis:**

#### **Primary Development Branches:**
- **`Khrityshree/`**: Feature selection optimization
- **`Classification 204/`**: 204-patient cohort analysis
- **`Classification 208 + 166/`**: Combined dataset (374 patients)
- **`Clot Burden/`**: Clot burden score integration
- **`Streamlit App/`**: Production deployment interface

#### **Temporal Development:**
- **`April/`**, **`May 6/`**, **`May 25/`**, **`May 27/`**, **`May 29 Classification/`**: Iterative model improvements

### **Technical Implementation:**

#### **Data Preprocessing:**
```python
# Binary Classification Setup
data['MRS_Class'] = data['MRS'].apply(lambda x: 1 if x <= 2 else 0)
# 1 = Good outcome (mRS 0-2)
# 0 = Poor outcome (mRS 3-6)

# Missing Data Handling
missing_threshold = 20  # Drop columns with >20% missing values
imputer = SimpleImputer(strategy="median")
```

#### **Machine Learning Pipeline:**
```python
# Model Ensemble
models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(), 
    "LightGBM": LGBMClassifier(),
    "Logistic Regression": LogisticRegression()
}

# Hyperparameter Optimization
grid_search = GridSearchCV(
    pipeline, param_grid, 
    cv=10, scoring='f1', 
    n_jobs=-1
)
```

#### **Evaluation Framework:**
```python
# Cross-Validation Setup
kf = KFold(n_splits=10, random_state=42, shuffle=True)
y_pred = cross_val_predict(best_model, X, y, cv=kf)

# Performance Metrics
- F1 Score (primary metric)
- Precision and Recall
- Confusion Matrix
- Classification Report
```

### **Key Findings from Analysis:**

#### **Best Performing Models:**
1. **Random Forest**: Consistently high performance across datasets
2. **XGBoost**: Strong performance with feature interactions
3. **LightGBM**: Efficient processing with competitive accuracy

#### **Performance Metrics:**
- **Cross-validation F1 Score**: 0.73-0.81 range
- **Accuracy**: 81% on validation set
- **Precision/Recall**: Balanced performance (0.80-0.86)

---

## 5. Production Deployment (`Streamlit App/`)

### **Application Architecture:**
```python
# Dual Model Integration
unet_model = load_model('UNET_nd_cl_64v2.h5')          # Segmentation
classification_model = joblib.load('GSCV_NOFS.pkl')    # mRS Prediction

# Workflow
1. Upload NIfTI file → 
2. U-Net segmentation → 
3. Volume calculation → 
4. Feature extraction → 
5. mRS prediction
```

### **User Interface Features:**
- **File Upload**: NIfTI format support
- **Real-time Processing**: Integrated pipeline execution  
- **Volume Estimation**: Automated stroke volume calculation
- **Outcome Prediction**: mRS classification with confidence
- **Visualization**: Processing results and predictions

---

## 6. Data Architecture and Flow

### **Data Pipeline:**
```
Raw Medical Data (.nii, .nrrd)
    ↓
Preprocessing (Normalization, Slicing)
    ↓
Training Data (128×128 slices)
    ↓
U-Net Training (Segmentation)
    ↓
Trained Model (.h5)
    ↓
Inference Pipeline (Volume Estimation)
    ↓
Feature Engineering (Clinical + Imaging)
    ↓
mRS Classification Model (.pkl)
    ↓
Production Deployment (Streamlit)
```

### **File Format Standards:**
- **Medical Images**: NIfTI (.nii, .nii.gz)
- **Segmentations**: NRRD (.seg.nrrd)
- **Models**: HDF5 (.h5), Pickle (.pkl)
- **Results**: PNG images, CSV reports

---

## 7. Clinical Integration and Impact

### **Clinical Workflow Integration:**
1. **Input**: Patient CT scans (DWI, FLAIR sequences)
2. **Processing**: Automated stroke volume quantification
3. **Analysis**: Multi-feature mRS outcome prediction  
4. **Output**: Quantitative prognostic information

### **Key Innovations:**
- **Automated Volume Quantification**: Eliminates manual segmentation
- **Multi-modal Integration**: Combines imaging and clinical features
- **Real-time Prediction**: Streamlit deployment for clinical use
- **Robust Validation**: 10-fold cross-validation framework

### **Clinical Significance:**
- **Prognostic Accuracy**: 81% accuracy in functional outcome prediction
- **Time Efficiency**: Reduces analysis time from hours to minutes
- **Standardization**: Consistent, reproducible measurements
- **Decision Support**: Evidence-based treatment planning

---

## 8. Technical Specifications Summary

### **Deep Learning Component:**
- **Framework**: TensorFlow/Keras
- **Architecture**: U-Net with skip connections
- **Input Resolution**: 128×128 pixels
- **Training Dataset**: 3,077 annotated slices
- **Validation Method**: Hold-out test set (575 images)

### **Machine Learning Component:**  
- **Framework**: scikit-learn, XGBoost, LightGBM
- **Optimization**: GridSearchCV with 10-fold CV
- **Feature Selection**: Automated missing value handling
- **Patient Cohorts**: 204-374 patients across studies
- **Primary Metric**: F1-score for balanced evaluation

### **Production System:**
- **Deployment**: Streamlit web application
- **Model Formats**: HDF5 (deep learning), Pickle (ML)
- **Integration**: Seamless pipeline from imaging to prediction
- **Performance**: Real-time inference capabilities

---

## Conclusion

This comprehensive stroke prediction system represents a significant advancement in clinical decision-making tools. The integration of deep learning-based volume quantification with machine learning outcome prediction provides clinicians with robust, automated prognostic capabilities. The system's modular architecture ensures maintainability and scalability, while the rigorous validation framework demonstrates clinical reliability.

**Key Achievements:**
- End-to-end automation of stroke volume quantification
- High-accuracy functional outcome prediction (81%)
- Production-ready clinical deployment interface
- Comprehensive validation across multiple patient cohorts
- Standardized, reproducible clinical measurements