# PDF_using_GAN

# Assignment–2  
## Learning Probability Density Functions Using Data-Driven GANs

---

## 1. Objective

The aim of this assignment is to model and learn an **unknown probability density function (PDF)** of a transformed random variable using a **Generative Adversarial Network (GAN)**. The approach is entirely data-driven and does not rely on any predefined analytical or parametric distribution.

---

## 2. Dataset

- **Dataset Title**: India Air Quality Data  
- **Selected Feature**: NO₂ Concentration (x)  
- **Data Source**: Kaggle  
- **Dataset Link**: https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data  

---

## 3. Methodology

### Step 1: Nonlinear Data Transformation

Each NO₂ concentration value is transformed using a nonlinear function that depends on the university roll number:

z = x + a_r sin(b_r x)


**University Roll Number:**  
r = 102303738  

#### Computed Transformation Parameters

| Parameter | Formula | Value |
|---------|--------|-------|
| a_r | 0.5 × (r mod 7) | 2.5 |
| b_r | 0.3 × ((r mod 5) + 1) | 1.2 |

**Final Transformation Equation:**
z = x + 2.0 sin(0.9x)


---

### Step 2: PDF Learning Using GAN

The transformed variable **z** is assumed to originate from an **unknown probability distribution**.  
To approximate this distribution, a **Generative Adversarial Network (GAN)** is constructed and trained using only the observed samples of z.

#### Generator Architecture

- **Input**: Random noise sampled from a standard normal distribution N(0,1)
- **Layers**:
  - Linear (1 → 64) with ReLU  
  - Linear (64 → 128) with ReLU  
  - Linear (128 → 1)

#### Discriminator Architecture

- **Input**: Real and generated samples of z
- **Layers**:
  - Linear (1 → 128) with LeakyReLU  
  - Linear (128 → 64) with LeakyReLU  
  - Linear (64 → 1) with Sigmoid  

**Training Mechanism:**

- The discriminator learns to differentiate between real and generated samples.
- The generator learns to produce samples that resemble the real transformed data.

---

### Step 3: PDF Approximation

After training the GAN:

- A large number of synthetic samples are generated using the trained generator.
- The probability density function is approximated using:
  - Histogram-based density estimation  
  - Kernel Density Estimation (KDE)

---


## 4. Observations

### Mode Representation

- The generator captures the dominant modes of the transformed distribution.
- Minor mode collapse may occur due to adversarial training dynamics.

### Training Behavior

- Training remained stable after normalization.
- Generator and discriminator losses showed gradual convergence.

### Distribution Quality

- The generated probability density closely matches the real transformed data.
- This confirms effective learning of the unknown PDF.

---

## 5. Tools and Libraries Used

- Python  
- PyTorch  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 6. Conclusion

- A nonlinear, roll-number-based transformation was applied to NO₂ concentration data.
- A GAN was trained to learn the unknown probability distribution without assuming any parametric form.
- Generated samples closely matched the original transformed data, verified using histogram and KDE plots.
- This experiment demonstrates the effectiveness of GANs in learning complex probability distributions directly from data.

---



