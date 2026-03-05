

**Explainable Deep Learning for Cervical Cancer Cell Classification: Comparative Analysis of CNN Architectures with DenseNet-121**

**📘 Overview**

Proposes an explainable deep learning framework for automated cervical cancer cell classification using Pap-smear cytology images. Conducts a systematic comparison of nine CNN architectures, including DenseNet, ResNet, VGG, Inception, EfficientNet, MaxViT, and a custom CNN-18 model.

Uses a curated dataset of 6,374 cervical cell images across five epithelial classes.

Ensures a fair evaluation by training all models under identical preprocessing, augmentation, and hyperparameter settings.

DenseNet-121 achieved the best performance with:

✅ 98.85% Accuracy

✅ 0.9885 F1-Score

✅ Statistically significant improvement (ANOVA, p < 0.05)

Incorporates Grad-CAM for model interpretability, highlighting clinically relevant nuclear and cytoplasmic regions.

Achieves a clinically favorable False Negative Rate (FNR) below 2% for abnormal lesions.

Demonstrates strong potential for AI-assisted cervical cancer screening, especially in low-resource clinical settings.

**📁 Dataset**

The dataset used in this study was custom-built by the authors for cervical cancer cell classification. The dataset was curated and organized using the Roboflow platform, where cervical cytology images were collected, preprocessed, and annotated for multi-class classification experiments.

The dataset contains microscopic cervical cell images categorized into several epithelial cell classes, including:

Dyskeratotic cells

Koilocytotic cells

Metaplastic cells

Parabasal cells

Superficial–Intermediate cells

All images were carefully labeled and processed to ensure consistency and quality for training deep learning models. The dataset used in this study is publicly accessible for research purposes through the following link:

👉 Dataset Link: https://app.roboflow.com/first-orenm/cervical-cancer-cell-classificat-i93u6/3

**🚀 Key Features**

🔍 Dataset Preparation

A curated cervical cytology dataset containing 6,374 cell images was constructed.

Images were collected from multiple publicly available Roboflow datasets and verified manually for label consistency.

The dataset includes five epithelial cell classes:

Dyskeratotic

Koilocytotic

Metaplastic

Parabasal

Superficial–Intermediate

Data were split into 70% training, 15% validation, and 15% testing for reliable evaluation.

🧪 Data Preprocessing & Augmentation

Image preprocessing was applied to improve data quality and model generalization.

Key preprocessing steps include:

Image resizing and normalization

Brightness adjustment (±25%) to handle illumination variations

Rotation augmentation (±15°) to improve orientation invariance

These techniques help reduce overfitting and improve model robustness.

🧠 Deep Learning Model Evaluation

Nine CNN architectures were evaluated under the same training pipeline:

DenseNet-121

ResNet-50, ResNet-101, ResNet-152

VGG-16, VGG-19

Inception-V3

EfficientNet-B4

Custom CNN-18

Performance was measured using:

Accuracy

Precision

Recall

F1-Score

AUC

🌳 High-Performance Model

Model: DenseNet-121 (Best Performing)

Accuracy: 98.85%

Precision: 0.9886

Recall: 0.9884

F1-Score: 0.9885

Demonstrates strong ability to capture fine cytological features such as nuclear irregularities and chromatin patterns.

Achieves high performance with relatively fewer parameters, making it suitable for clinical deployment.

🧬 Explainable AI (XAI)

To ensure model interpretability, the study applied:

Grad-CAM Visualization

This technique:

Highlights important regions of cervical cell images influencing model decisions

Helps clinicians understand why the model predicts a specific cell type

Improves transparency and clinical trust in AI-assisted screening systems.

🩺 Clinical Risk-Oriented Evaluation

In cervical cancer screening, false negatives are highly critical because they may delay diagnosis.

Therefore, the study also evaluated:

False Negative Rate (FNR) for abnormal epithelial cells.

Overall abnormal lesion FNR:

1.93%, indicating very few abnormal cells were misclassified.

This demonstrates the model’s high reliability for clinical screening applications.

**📊 Results & Highlights**

DenseNet-121 achieved the best overall performance among all evaluated CNN architectures for cervical cell classification.

The model demonstrated very high classification accuracy (98.85%) across five epithelial cell categories.

Misclassification rate was extremely low, indicating the model’s strong ability to distinguish between morphologically similar cervical cell types.

Grad-CAM based Explainable AI analysis highlighted the most relevant cytological regions such as:

Nuclear morphology

Chromatin texture

Cell boundary patterns These visualizations confirm that the model focuses on clinically meaningful cellular structures.

The proposed approach outperforms several previously reported cervical cytology classification studies in terms of accuracy and reliability.

The model also shows strong generalization capability, maintaining high performance across training, validation, and testing sets.

In addition to standard metrics, clinical risk-based evaluation was performed using False Negative Rate (FNR).

The overall abnormal lesion FNR was only 1.93%, indicating that very few abnormal cervical cells were missed by the model.

These results demonstrate the practical potential of the proposed system for AI-assisted cervical cancer screening and early diagnosis.
