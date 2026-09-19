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
- `notebook.ipynb`: End-to-end data pipeline, FiLM implementation & training
- `README.md`: Technical project documentation

---

## 7. Model Weights

The trained model checkpoint (`final_lunar_model.pth`) is hosted externally:
- [Download final_lunar_model.pth](https://drive.google.com/file/d/18heVyWNB0XFpPO12lAARpHRfU4wauCLk/view?usp=sharing)
