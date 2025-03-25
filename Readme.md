<p align="center">
  <img src="https://raw.githubusercontent.com/Muratcanisik4/QML4Enzyme/main/docs/assets/qml4enzyme_logo.png" alt="QML4Enzyme Logo">
</p>

# QML4Enzyme - Quantum Machine Learning Framework for Enzyme Function Prediction

QML4Enzyme is a multi-modal quantum machine learning pipeline designed to predict enzyme function via Enzyme Commission (EC) classification. By integrating quantum mechanical descriptors derived from QM/MM calculations with generative models and transformer architectures, QML4Enzyme enhances predictive accuracy across the full EC classification hierarchy.

## Installation

Clone the repository:

```bash
git clone https://github.com/Muratcanisik4/QML4Enzyme.git
cd QML4Enzyme
```

## Required Packages

To use QML4Enzyme effectively, ensure you have the following packages installed:

- **PyJoules**
- **Torch**
- **Pandas**
- **Scikit-Learn**
- **Transformers**
- **Thop**
- **Qiskit**
- **rdkit**

## Usage

You can run the training script using:

```bash
python train_qvt.py --config configs/qml4enzyme_config.yaml

```
## Data

The enzyme subset of our dataset comprises representatives from all seven top-level EC classes—(1) Oxidoreductases, (2) Transferases, (3) Hydrolases, (4) Lyases, (5) Isomerases, (6) Ligases, and (7) Translocases—enabling multi-class prediction across the complete EC classification hierarchy. Using QM/MM-derived descriptors, such as Nuclear Repulsion Energy, SCF Total Energy, and Maximum and RMS Gradients, the model captures crucial electronic and structural characteristics influencing enzyme reactivity.


## Supported Architectures

We tested QML4Enzyme on IBM's 16-qubit ibmq_guadalupe quantum hardware and GPU-accelerated systems for hybrid training.


## Future Development

Planned features include:
DNMT3 detailed investigations
Incorporate 1) graph-based structural representations, 2) tabular physical properties, and 3) secondary structure details.




## Citation
If you use QML4Enzyme in your work, please cite the following:

```bash
@article{Isik2024QML4Enzyme,
  title={Quantum Machine Learning for Enzyme Property Prediction},
  author={Murat Isik and Mandeep Kaur Saggi and Humaira Gowher and Sabre Kais},
  year={2024},
  url={https://github.com/Muratcanisik4/QML4Enzyme}
}

```
