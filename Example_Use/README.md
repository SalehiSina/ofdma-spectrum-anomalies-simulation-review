# Anomaly Detection

Here we implement and evaluate two approches for anomaly detection in ofdma systems using our dataset.

<p align="center">
  <img src="Figures/anomaly_sample.jpg" width="600">
</p>

---

# Dataset

Download the dataset from the following link:

https://link.com

After downloading the dataset, place it inside the `Example_Use/Dataset` directory.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/akdd11/ofdma-spectrum-anomalies-simulation.git
cd ofdma-spectrum-anomalies-simulation/Example_Use
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

To train the unsupervised (VAE) model, run:

```bash
python src/train_unsupervised.py --data_path ".\Dataset" --Max_LR 1e-4
```

To train the supervised (ResNet) model, run:

```bash
python src/train_unsupervised.py --data_path ".\Dataset" --Max_LR 1e-2
```

Or, to fine-tune the ResNet18 model using the pretrained weights located at *initial\_path*, run:

```bash
python src/train_unsupervised.py --data_path ".\Dataset" --Max_LR 1e-2 --initial_weights "initial_path"
```


