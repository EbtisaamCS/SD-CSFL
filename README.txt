# SD-CSFL: A Synthetic Data-Driven Conformity Scoring Framework for Robust Federated Learning  
### Official Implementation — WACV 2026 (Algorithms Track)

<p align="center">
  <img src="FMW.pdf" width="780"/>
</p>

This repository contains the official implementation of the WACV 2026 paper:

**“SD-CSFL: A Synthetic Data-Driven Conformity Scoring Framework for Robust Federated Learning”**

---

##  Overview

SD-CSFL is a unified defense framework that protects Federated Learning (FL) from:

- Gradient manipulation attacks (IPM, ALIE)
- Backdoor attacks (A3FL, F3BA, CerP)
- Heterogeneous (Non-IID) client distributions
- Privacy leakage from validation data

**Core Idea**  
A clean **synthetic calibration dataset** is used to compute **entropy-based nonconformity scores** for each client model.  
These scores are evaluated with **adaptive percentile thresholds** to classify clients as benign or malicious.  
Only benign updates are aggregated into the global model, enabling a unified and privacy-preserving anomaly detection mechanism.

---

## Abstract 

Federated Learning remains highly vulnerable to gradient manipulation and backdoor attacks, especially under heterogeneous client data. Existing defenses are narrow, data-dependent, or unstable under Non-IID conditions.  
SD-CSFL introduces a **synthetic data–driven entropy-based conformity scoring mechanism**, combined with **adaptive percentile thresholding** and **stratified calibration**, enabling robust and privacy-preserving detection of malicious model updates.  
Experiments on **CIFAR-10** and **Birds-525** demonstrate up to **35% higher detection** and **80% lower backdoor success rates** compared to recent defenses.

---

##  Method Summary

### **1. Synthetic Calibration Data**  
- Generated using Stable Diffusion v2  
- Artistic, non-photorealistic, and abstract variants  
- Designed to amplify entropy sensitivity for anomaly detection  

### **2. Entropy-Based Nonconformity Score**  
- Each client model is evaluated on synthetic data  
- Mean entropy across batches forms the score  
- Higher scores indicate uncertainty or malicious behavior  

### **3. Adaptive Percentile Thresholding**  
- Thresholds are set using symmetric quantiles (e.g., 30–70%)  
- False-positive budget guarantees bounded error under Non-IID settings  
- Recalculated each round to adapt to evolving score distributions  

### **4. Aggregation**  
- Only benign client updates are aggregated  
- Optional: send perturbed global model to suspected malicious clients  

---

##  Datasets

### Birds-525 Real Dataset  
Training data for real clients  
🔗 https://drive.google.com/file/d/1NvVfcrvXNOzX8mz1A-yhudegYJZXprSJ/view

### Birds-Synth (Synthetic Calibration Dataset)  
Used exclusively for entropy-based scoring  
🔗 https://drive.google.com/file/d/10akkldmavU_CxsMlWacL0Lq0Nr9eQuz6/view?usp=drive_link

### CIFAR-10 Real Dataset  
Official link  
🔗 https://www.cs.toronto.edu/~kriz/cifar.html

### CIFAR-10-Synth (Synthetic Calibration Dataset)  
Synthetic artistic CIFAR-10 images  
🔗 https://drive.google.com/file/d/10akkldmavU_CxsMlWacL0Lq0Nr9eQuz6/view?usp=drive_link

###  Synthetic Dataset Generator Code  
Stable Diffusion prompts & synthetic data pipeline  
🔗 https://github.com/A-Kerim/Synthetic-Data-Usability



---

##  How to Run

### ** Install dependencies**
```bash
pip install -r requirements.txt

---
### License & Usage
The SD-CSFL framework and the associated synthetic datasets are made freely available for academic and non-commercial research purposes.  
If you use the framework or the datasets, please cite our WACV 2026 paper.

---

### Acknowledgements
This work was supported by the Saudi Arabian Ministry of Education, the Saudi Arabian Cultural Bureau in London, and Umm Al-Qura University. We also thank the High-End Computing facility at Lancaster University for providing essential computational resources.

