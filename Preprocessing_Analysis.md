# Comprehensive Preprocessing Analysis for Stroke Prediction Pipeline

## Overview
This document analyzes the preprocessing approaches used in the existing Scripts folder and compares them with standard medical imaging preprocessing techniques. The analysis is divided into three main sections: preprocessing pipeline, UNet training approach, and recommendations.

---

## Section 1: Preprocessing Folder Analysis

### What They Did in the Preprocessing Folder

#### 1. Core Preprocessing Pipeline (`preprocessing.ipynb`)

**Normalization Approach:**
- **Method:** Hounsfield Unit (HU) based normalization
- **Range:** 0 to 1300 HU units
- **Formula:** `(pixel_value - HOUNSFIELD_MIN) / (HOUNSFIELD_MAX - HOUNSFIELD_MIN)`
- **Constants:**
  ```python
  HOUNSFIELD_MAX = 1300
  HOUNSFIELD_MIN = 0
  ```

**Key Functions:**

1. **`normalizeImageIntesityRange(image)`**
   - Clips image values to [0, 1300] HU range
   - Normalizes to [0, 1] scale
   - Specifically designed for CT scan preprocessing
   - **Strength:** Good for CT images with known HU ranges
   - **Limitation:** Not suitable for MRI images (different intensity characteristics)

2. **`sliceAndSaveVolumeImage(path, output_folder, prefix)`**
   - Extracts 2D slices from 3D volumes
   - **Direction:** Z-axis slicing (axial view)
   - **Format:** Saves as PNG images
   - **Output structure:** Organized slice-by-slice export

**File Format Handling:**
- **Volume files:** `.nii` (NIfTI format)
- **Segmentation files:** `.seg.nrrd` (3D Slicer format)
- **Libraries used:** `slicerio`, `pynrrd`, `SimpleITK`

#### 2. File Organization (`move.ipynb`)
- **Purpose:** Automated file management
- **Functions:** Directory creation, file moving, organization
- **Benefit:** Maintains consistent data structure

### Preprocessing Workflow Summary:
1. **Input:** 3D medical volumes (.nii) and segmentation masks (.seg.nrrd)
2. **Normalization:** Hounsfield unit-based (0-1300 HU → 0-1 scale)
3. **Slicing:** Z-axis extraction to 2D slices
4. **Output:** Organized PNG slice datasets

---

## Section 2: UNet Training Approach

### What They Did During UNet Training

#### 1. Data Loading and Preprocessing

**Image Generator Configuration:**
```python
data_gen_args = dict(rescale=1./255)
```

**Key Observations:**
- **Additional normalization:** Divides by 255 (0-255 → 0-1 scale)
- **Problem:** Double normalization! 
  - First: HU normalization (0-1300 HU → 0-1)
  - Second: /255 rescaling (0-1 → 0-0.004)
- **Result:** Extremely small pixel values (0-0.004 range)

#### 2. Model Architecture

**UNet Configuration:**
- **Input size:** 128x128 grayscale
- **Levels:** 4 (encoder-decoder depth)
- **Initial features:** 32
- **Architecture:** Standard UNet with skip connections

**Loss Function:**
- **Primary:** Dice Loss
- **Metrics:** Accuracy, IoU Score, F1 Score
- **Good choice:** Dice loss excellent for segmentation tasks

#### 3. Training Setup

**Data Split:**
- **Training:** 2,502 images
- **Testing:** 575 images
- **Batch size:** 32
- **Epochs:** 50

**Issues Identified:**
1. **Double normalization problem**
2. **No data augmentation** (only rescaling)
3. **No batch normalization** in UNetv1 (active model)
4. **No dropout** for regularization

---

## Section 3: Analysis and Recommendations

### Critical Issues Found

#### 1. **Double Normalization Problem (CRITICAL)**
- **Current flow:** HU normalization → /255 rescaling
- **Result:** Pixel values in range [0, 0.004] instead of [0, 1]
- **Impact:** Poor model training, loss of image contrast
- **Solution:** Remove `/255` rescaling from ImageDataGenerator

#### 2. **MRI vs CT Preprocessing Mismatch**
- **Current:** Hounsfield unit normalization (CT-specific)
- **Problem:** MRI doesn't use Hounsfield units
- **Your MRI_Plan.ipynb approach is better:** Brain masking + z-score normalization

#### 3. **Limited Augmentation**
- **Current:** Only rescaling
- **Missing:** Rotation, flip, zoom, shift augmentations
- **Impact:** Reduced model generalization

### What's Important vs Not Important

#### ✅ **Good Practices (Keep These):**
1. **Dice Loss:** Excellent choice for segmentation
2. **IoU and F1 metrics:** Appropriate for medical segmentation
3. **File organization:** Clean data structure
4. **Slice extraction:** Good for 2D training from 3D volumes

#### ❌ **Problems (Fix These):**
1. **Double normalization:** Fix immediately
2. **Wrong normalization for MRI:** Use z-score or histogram matching
3. **No augmentation:** Add rotation, flip, elastic deformation
4. **No regularization:** Add batch norm and dropout

#### 🔄 **Improvements (Consider These):**
1. **UNetv2/v3 implementations:** Include batch norm and dropout
2. **Combined loss:** Mix of dice and cross-entropy
3. **Advanced augmentation:** Elastic deformation, intensity shifts
4. **Multi-scale training:** Different input sizes

### Recommended Preprocessing Pipeline

#### For CT Images:
1. **Windowing:** Apply appropriate CT windows (brain: [-20, 80] HU)
2. **Normalization:** HU-based normalization (but fix range)
3. **Augmentation:** Rotation, flip, zoom
4. **No additional /255 rescaling**

#### For MRI Images (Your Current Approach is Better):
1. **Brain masking:** Remove non-brain tissue
2. **Intensity normalization:** Z-score or robust normalization
3. **Skull stripping:** If not already done
4. **Bias field correction:** For MRI-specific artifacts

### Implementation Priority

#### **Immediate Fixes (High Priority):**
1. Fix double normalization in UNet training
2. Separate CT and MRI preprocessing pipelines
3. Add basic data augmentation

#### **Medium Priority:**
1. Implement UNetv2 with batch normalization
2. Add dropout for regularization
3. Experiment with combined loss functions

#### **Future Improvements:**
1. Advanced augmentation techniques
2. Multi-modal fusion strategies
3. Attention mechanisms in UNet

### Conclusion

The existing preprocessing pipeline shows good understanding of medical imaging fundamentals but has critical issues:

1. **Double normalization** is causing poor training performance
2. **MRI preprocessing** needs different approach than CT
3. **Your MRI_Plan.ipynb** already implements better MRI preprocessing

**Recommendation:** Use the existing structure but fix the normalization issues and implement modality-specific preprocessing pipelines.
