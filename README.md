# Lunar Topographic Classification Under Variable Solar Illumination (SunFiLMNet)

A deep learning framework engineered to classify lunar micro-topography into positive relief (rises/mounds) and negative relief (depths/craters) from single-channel orbital imagery, resolving illumination-induced topographic inversion without geometric distortion.

---

## 1. Problem Formulation & Illumination Physics

Monocular lunar orbital imagery presents a fundamental computer vision ambiguity known as **topographic inversion**:
- Optical orbital sensors capture reflected sunlight without intrinsic depth information.
- Illumination direction dictates shadow placement:
  - An eastward depression (crater) exhibits cast shadows on its eastern rim and high reflectance on its western slope.
  - An eastward elevation (mound) exhibits specular highlights on its eastern face and trailing shadows on its western slope.
- Without accounting for the solar illumination vector, convolutional neural networks are prone to confusing concave depressions with convex mounds.

Rather than rotating images geometrically—which introduces border padding, black triangular artifacts, and bilinear interpolation blur—this architecture resolves solar ambiguity directly at the feature representation level via **Feature-wise Linear Modulation (FiLM)**.

---

## 2. Model Architecture: SunFiLMNet

`SunFiLMNet` conditions intermediate convolutional feature representations dynamically on the solar illumination vector:
### 2.1 Inputs
- **Image Domain:** 256×256 single-channel grayscale lunar surface crops.
- **Solar Vector:** Continuous azimuth angle $\theta \in [0^\circ, 360^\circ)$ transformed into a 2D directional unit vector:
  $$\vec{s} = (\sin(\theta),\, \cos(\theta))$$

### 2.2 Feature-wise Linear Modulation (FiLM)
- **Backbone:** Modified ResNet-18 receiving 1 input channel.
- **Conditioning Mechanism:** Four specialized FiLM blocks interleave between residual stages (layer1 to layer4). A dedicated Multi-Layer Perceptron projects the 2D solar vector into affine transformation parameters $(\gamma_c, \beta_c)$ for each channel $c$:
  $$\text{FiLM}(\mathbf{F}_c) = \gamma_c \cdot \mathbf{F}_c + \beta_c$$
- This enables intermediate feature detectors to dynamically translate and scale their responses based on the sun's physical lighting orientation.

---

## 3. Topographic Classes & Target Metric

- **Class 0 (Depth):** Craters, impact pits, and depressions.
- **Class 1 (Rise):** Mounds, hills, rocks, and boulders.

### Evaluation Metric
Performance is evaluated using **Balanced Accuracy** to penalize majority-class bias and ensure equal sensitivity across both geological landforms:

$$\text{Balanced Accuracy} = \frac{\text{Recall}_{\text{Rise}} + \text{Recall}_{\text{Depth}}}{2}$$

---

## 4. Prior-Matched Threshold Calibration

Because raw sigmoid outputs tend to saturate toward majority classes under binary cross-entropy, decision thresholding was calibrated against the empirical terrain class prior (~63.6% Rise / ~36.4% Depth).

By applying an empirical percentile threshold ($t \approx 0.8657$), the model eliminates class collapse, delivering balanced sensitivity and specificity across evaluation samples.

---

## 5. Performance Summary

- **Validation Balanced Accuracy:** ~76.8% – 77.2%
- **Evaluation Outputs:** 2,000 calibrated test predictions.

---

## 6. Repository Structure

- `submission.csv`: Calibrated predictions (2,000 evaluation samples)
- `lunar_classification.ipynb`: End-to-end data pipeline, FiLM implementation & training
- `README.md`: Technical project documentation

---

## 7. Model Weights

The trained model checkpoint (`final_lunar_model.pth`) is hosted externally:
- [Download final_lunar_model.pth](https://drive.google.com/file/d/18heVyWNB0XFpPO12lAARpHRfU4wauCLk/view?usp=sharing)
