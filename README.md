# Invoice Authenticity Evaluation: Conventional Image Analysis and Deep Learning

This repository contains the implementation and experimental evaluation for distinguishing **Real**, **AI-generated Fake**, and **Tampered** invoice images. The study investigates whether conventional image-quality and similarity measures are sufficient to differentiate these invoice categories and further evaluates pretrained deep-learning models for automated invoice authentication.

The experiments are organized into two main components:

1. **Conventional Image Quality and Similarity Evaluation**
2. **Deep Learning-Based Invoice Classification**

---

## 1. Introduction

The increasing use of generative AI and image-editing tools has made the creation and manipulation of digital documents increasingly accessible. Invoice authentication is particularly challenging because AI-generated and tampered invoices can maintain visual characteristics that closely resemble genuine documents.

This repository evaluates invoice images from three categories:

* **Real Invoices** — genuine invoice images.
* **AI-Generated Fake Invoices** — synthetic invoices generated using large language model/generative AI-based approaches.
* **Tampered Invoices** — manipulated versions of genuine invoices containing intentional modifications.

The primary objective of the conventional analysis is to determine whether visual and statistical image characteristics provide sufficient evidence to distinguish these categories. Deep-learning experiments are subsequently performed to evaluate automated classification performance.

---

## 2. Dataset

The experimental dataset contains **821 invoice images** distributed across three classes:

| Class             | Number of Images |
| ----------------- | ---------------: |
| Real              |              265 |
| AI-Generated Fake |              265 |
| Tampered          |              291 |
| **Total**         |          **821** |

For deep-learning experiments, the dataset is divided class-wise into:

* **Training:** 70%
* **Validation:** 15%
* **Testing:** 15%

This stratified division maintains approximately the same class distribution across the experimental subsets.

---

## 3. Experimental Setup

The experiments are implemented in **Python** using commonly used image-processing, computer-vision, machine-learning, and deep-learning libraries.

The overall experimental pipeline consists of:

```text
             Invoice Dataset
                    |
       +-----------------------------+
       |                             |
       v                             v
Conventional Image             Deep Learning
    Analysis                   Classification
       |                             |
       v                             v
Image Characteristics          Pretrained CNNs
       |                             |
       v                             v
Inter-Class Similarity         Model Evaluation
       |                             |
       +-------------+---------------+
                     |
                     v
             Comparative Analysis
```

The conventional evaluation is intentionally independent of the deep-learning classification stage. It examines whether measurable image characteristics alone expose clear differences among genuine, synthetic, and manipulated invoices.

---

# 4. Methodology

## 4.1 Conventional Image Quality Evaluation

Two complementary forms of conventional analysis are performed.

### 4.1.1 Independent Image Characteristics

Each invoice image is independently analyzed using statistical, structural, texture, and edge-based characteristics.

The following measures are extracted:

| Category               | Measure                      |
| ---------------------- | ---------------------------- |
| Intensity              | Mean Intensity               |
| Intensity              | Intensity Standard Deviation |
| Information Content    | Entropy                      |
| Image Detail           | Sharpness                    |
| Structural Information | Edge Density                 |
| Texture                | GLCM Contrast                |
| Texture                | GLCM Homogeneity             |
| Texture                | GLCM Energy                  |
| Texture                | GLCM Correlation             |

These measurements characterize individual images without requiring a reference image from another class.

### Mean Intensity

Mean intensity measures the average grayscale intensity of an image:

$$
\mu = \frac{1}{N}\sum_{i=1}^{N}I_i
$$

where \(I_i\) represents the intensity of pixel \(i\) and \(N\) represents the total number of pixels.

### Intensity Standard Deviation

The standard deviation measures the variation of pixel intensities around the mean:

$$
\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(I_i-\mu)^2}
$$

### Entropy

Entropy quantifies the amount of information or uncertainty contained in an image:

$$
H=-\sum_i p_i\log_2(p_i)
$$

where \(p_i\) represents the probability associated with intensity level \(i\).

### Sharpness

Image sharpness is estimated from local intensity variations to characterize the amount of fine detail and edge information present in an invoice.

### Edge Density

Edge density measures the proportion of pixels identified as edges relative to the complete image area.

### GLCM Texture Features

A Gray-Level Co-occurrence Matrix (GLCM) is used to characterize spatial relationships between neighboring pixel intensities.

The extracted GLCM features include:

* Contrast
* Homogeneity
* Energy
* Correlation

Together, these measurements provide a conventional representation of the visual characteristics of each invoice category.

---

## 4.2 Inter-Class Similarity Analysis

Independent image characteristics indicate the statistical properties of individual images but do not directly quantify how visually similar different invoice categories are.

Therefore, pairwise inter-class similarity analysis is also performed using:

* **Structural Similarity Index Measure (SSIM)**
* **Histogram Similarity**
* **Local Binary Pattern (LBP) Similarity**
* **Histogram of Oriented Gradients (HOG) Similarity**
* **Frequency-Domain Similarity**

The primary class comparisons include:

```text
Real ↔ Fake
Real ↔ Tampered
Fake ↔ Tampered
```

### Structural Similarity

SSIM evaluates similarity based on luminance, contrast, and structural information:

$$
SSIM(x,y)=
\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}
{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}
$$

Higher SSIM values indicate greater structural similarity.

### Histogram Similarity

Intensity histograms are compared to determine whether different invoice categories exhibit similar global intensity distributions.

### Local Binary Pattern

LBP represents local texture structures and is used to evaluate similarity in fine-grained texture characteristics.

### Histogram of Oriented Gradients

HOG captures edge orientation and local shape information, providing a representation of the structural layout of invoice images.

### Frequency-Domain Similarity

Frequency-domain analysis evaluates similarities in the spatial-frequency characteristics of the invoice images.

---

# 5. Conventional Evaluation Results

The conventional analysis demonstrates substantial visual and statistical similarity among the invoice categories.

Representative inter-class results include:

| Comparison           | Measure              |    Mean ± STD |
| -------------------- | -------------------- | ------------: |
| Real vs. Fake        | SSIM                 | 0.674 ± 0.043 |
| Real vs. Tampered    | SSIM                 | 0.710 ± 0.062 |
| Real vs. Tampered    | Histogram Similarity | 0.997 ± 0.003 |
| Real vs. Tampered    | LBP Similarity       | 0.999 ± 0.001 |
| Inter-Class Analysis | Frequency Similarity |       ≈ 0.994 |

Particularly high histogram, LBP, and frequency-domain similarities indicate that manipulated invoices can preserve many low-level visual characteristics of genuine documents.

The results suggest that conventional visual-quality measures alone provide limited separation between the considered invoice categories. This motivates the use of learned representations capable of identifying more discriminative patterns.

---

# 6. Deep Learning-Based Classification

To further investigate automated invoice authentication, pretrained convolutional neural networks are evaluated using transfer learning.

The evaluated architectures range from lightweight CNN models to deeper architectures, including:

* MobileNet-based architecture
* VGG16
* ResNet50

Pretrained weights are used to initialize the feature-extraction networks, while the original output layers are replaced with classification layers corresponding to the three invoice categories:

```text
Real
Fake
Tampered
```

The models are trained and evaluated using the same class-wise train, validation, and test partitions.

The final baseline experiments are performed **without data augmentation** to preserve the original visual properties of the invoice images.

---

## 6.1 Model Evaluation

Classification performance is assessed using standard metrics, including:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Accuracy is calculated as:

$$
Accuracy =
\frac{TP+TN}
{TP+TN+FP+FN}
$$

Precision, recall, and F1-score provide class-level information that complements overall classification accuracy.

---

# 7. Deep Learning Results

Among the evaluated pretrained CNN baselines, **VGG16** and **ResNet50** achieved the highest test accuracy.

| Model    | Test Accuracy |
| -------- | ------------: |
| VGG16    |    **92.31%** |
| ResNet50 |    **92.31%** |

These results demonstrate that pretrained convolutional representations can capture discriminative information that is not sufficiently exposed through conventional image-quality and similarity measurements.

The confusion matrix and classification report can be used for further examination of class-specific behavior, particularly to identify confusion among Real, AI-generated Fake, and Tampered invoices.

---

# 8. Key Findings

The experimental analysis provides three important observations:

1. **Real, AI-generated Fake, and Tampered invoices exhibit substantial visual similarity.**

2. **Conventional image characteristics and similarity measures alone provide limited discrimination**, particularly when manipulated documents preserve texture, intensity distribution, and frequency characteristics.

3. **Pretrained deep-learning models provide considerably stronger discrimination**, with VGG16 and ResNet50 reaching **92.31% test accuracy** in the baseline experiments.

These observations motivate the development of more specialized deep-learning frameworks for robust invoice authentication.

---


# 9. Running the Experiments

Clone the repository:

```bash
git clone <repository-url>
cd Invoice-Authentication
```

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the conventional image-characteristic analysis:

```bash
python conventional_analysis/image_characteristics.py
```

Run the inter-class similarity evaluation:

```bash
python conventional_analysis/similarity_analysis.py
```

The exact commands may be adjusted according to the final organization of the experimental scripts.

---

# 10. Requirements

The implementation primarily relies on Python-based image-processing and deep-learning libraries, such as:

```text
Python
NumPy
Pandas
OpenCV
scikit-image
scikit-learn
Matplotlib
TensorFlow / Keras
```

Specific package versions should be provided in `requirements.txt` to improve reproducibility.

---

# 11. Reproducibility

For reproducible experimentation, it is recommended to maintain:

* Fixed random seeds
* Identical train/validation/test partitions
* Consistent image preprocessing
* Fixed model initialization settings
* Recorded library versions
* Consistent evaluation metrics

Experimental outputs, including classification reports, confusion matrices, training curves, and conventional image-analysis results, should be stored separately from the source dataset.
<!--
---

## Citation

If you use this implementation or experimental framework in academic work, please cite the associated research article once its bibliographic information becomes available.

```bibtex
@article{invoice_authentication,
  title   = {Invoice Authentication Using Conventional Image Analysis and Deep Learning},
  author  = {Authors},
  journal = {Journal},
  year    = {2026}
}
```

---

## License

Please refer to the repository license for permitted use, modification, and redistribution of the source code.
-->
