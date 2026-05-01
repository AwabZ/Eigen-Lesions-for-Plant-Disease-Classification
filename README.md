# A Hybrid Eigen-Lesion and Fisher Vector Approach to Plant Disease Diagnosis: Classical Computer Vision
### A Hybrid Eigen-Lesion and Fisher Vector Approach

**Author:** Awwab Zuhair Hamdatu (202201754)  
**Course:** Computer Vision  
**Supervisor:** Dr. Anwar Majid Mirza

---

## Important: Instructions for Evaluation 


### 1. Which Notebook to Run?
* **Do NOT run the Training Notebook (`PlantDetectionProjectWithFisherVectors.ipynb`)** unless absolutely necessary. It performs heavy computation (PCA/GMM on 50,000 images) and takes **over 75 minutes** to complete on standard Colab GPUs. 
* **DO run the Testing Notebook (`TestingScript.ipynb`)**. This script is lightweight, pre-configured, and generates the visual pipeline results instantly using the saved model.

### 2. How to Run the Testing Script
The testing script requires the pre-trained model and sample images to be in the runtime environment.
1.  **Drag and Drop** the provided `final_plant_disease_model.pkl` (800MB) and the images from the `Test Images/` folder into the Colab **Files** pane (left sidebar).
2.  **Wait for Upload:** Google Colab is notoriously slow with large local uploads. 
    > Please wait until the **orange loading circle** (bottom-left of Colab) completely finishes. If you run the cell before the 800MB file is fully uploaded, the script will crash with an `UnpicklingError`. 
    > *Recommendation: Since this upload may take time, I suggest initiating the upload and checking other students' work while it completes.*
3.  **Select an Image:** In the first code cell, uncomment the `sample_img_path` variable corresponding to the disease you wish to visualize.
4.  **Run:** Execute the cell. It will produce the full diagnostic pipeline visualization (Segmentation -> Eigen-Lesions -> Fisher Vector -> Prediction).

### 3. Fallback
In the event that the file upload is too slow or fails due to connection issues, I have saved the **Testing Notebook with the outputs pre-rendered**. You can view the visual results directly in the notebook without executing the code. To be honest, I have attempted to put myself in your shoes and replicate this process, and I have noticed that in some cases, the model `final_plant_disease_model.pkl` refuses to load into the Colab Notebook.

---

##  Project Overview

This project challenges the dominance of Deep Learning in agriculture by establishing a **"Classical Ceiling"** for plant disease diagnosis. Instead of using "Black Box" Neural Networks, I developed a fully interpretable, mathematically rigorous pipeline using **Subspace Learning**.

My approach treats biological lesions as "Visual Words." I use **Color-Space Segmentation** to isolate pathology, **Principal Component Analysis (PCA)** to learn the fundamental structure of disease spots ("Eigen-Lesions"), and **Fisher Vectors** to encode high-order gradient statistics.

### Key Features
* **Interpretability:** Visualizes exactly *what* features the model uses (e.g., necrotic centers vs. yellow halos).
* **Efficiency:** Relies on linear algebra (PCA/SVM) rather than heavy convolutions, suitable for edge devices.
* **Novelty:** Adapts the "Eigenfaces" (1991) facial recognition technique to biological pathology.

## Results

Tested on the **PlantVillage Dataset** (38 Classes, 54k Images):

| Method | Feature Type | Accuracy |
| :--- | :--- | :--- |
| Random Guess | N/A | 2.6% |
| Bag of Visual Words (BoW) | Counts (Histogram) | 66.8% |
| Standard GLCM | Texture Statistics | ~74.0% |
| **Fisher Eigen-Lesions** | **Gradient Statistics** | **79.87%** |

While Deep Learning achieves higher accuracy (~99%), our method demonstrates that nearly **80% of the diagnostic signal** can be captured using interpretable, classical methods, offering a robust baseline for resource-constrained environments. Additionally, I do believe that, given enough time, I could push the accuracy of this methodology to **85%+**.

---

## References and LLM Assistance:
In the "research paper" attached with this project submission, you'll find that I have included all references of scientific research that my project utilizes. For the sake of academic honesty, I have included them here once more and underscored the sections of this project that have been assisted by Large Language Models (Gemini):

### *References*

**[1]** D. P. Hughes and M. Salath´e, “An open access repository of images on
plant health to enable the development of mobile disease diagnostics,”
arXiv preprint arXiv:1511.08060, 2015. \
**[2]** S. P. Mohanty, D. P. Hughes, and M. Salath´e, “Using deep learning for
image-based plant disease detection,” Frontiers in Plant Science, vol. 7,
p. 1419, 2016. \
**[3]** M. Turk and A. Pentland, “Eigenfaces for recognition,” Journal of
Cognitive Neuroscience, vol. 3, no. 1, pp. 71–86, 1991. \
**[4]** G. Csurka, C. R. Dance, L. Fan, J. Willamowski, and C. Bray, “Visual
categorization with bags of keypoints,” in Workshop on Statistical
Learning in Computer Vision, ECCV, vol. 1, pp. 1–22, 2004. \
**[5]** F. Perronnin and C. Dance, “Fisher kernels on visual vocabularies,” in
IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
pp. 1–8, 2007. \
**[6]** R. M. Haralick, K. Shanmugam, and I. Dinstein, “Textural features
for image classification,” IEEE Transactions on Systems, Man, and
Cybernetics, vol. 3, no. 6, pp. 610–621, 1973. 


### *LLM Assistance*:
- The crucial function`def fisher_vector(xx, gmm)`which computes Fisher Vector gradients for a set of descriptors (xx) against a GMM was almost entirely provided by Gemini. Although I have conceptually understood the concept behind Fisher Vectors, I did not have enough time to develop enough proficiency to be able to programmatically write down the rigorous mathematics behind Fisher Vectors. There probably exists a python module that computes these vectors and performs this encoding with a single imported function call, but I did not want to abstract away the mathematics behind it from myself. 

- While I have originally written down the fully working dataset download function myself, I have asked Gemini further "fortify" the function to combat any errors that may occur on your side because Google Colab acts unpredictably when it comes to dataset downloads sometimes, resulting in the current dense`def force_download_plantvillage(root_dir)`function.

- Some portions of the code, such as the "Visualization" snippets, have been refined by Gemini for optimization/prettification after my original implementation was already functional. 
