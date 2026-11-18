\# Satellite Land-Cover Classification



\### Exploring Computer Vision with CNNs, TensorFlow and Apache Spark



!\[Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python\&logoColor=white)

!\[TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow\&logoColor=white)

!\[Apache Spark](https://img.shields.io/badge/Apache\_Spark-4.2-E25A1C?logo=apachespark\&logoColor=white)

!\[Status](https://img.shields.io/badge/Status-Completed-2E8B57)



\*\*A practical investigation into how convolutional neural networks recognise patterns in satellite imagery.\*\*



\---



\## At a Glance



| Dataset | Classes | Images | Test Accuracy |

|:---:|:---:|:---:|:---:|

| EuroSAT RGB | 10 | 27,000 | \*\*80.77%\*\* |



I built and evaluated a convolutional neural network (CNN) to classify satellite images into ten land-cover categories.



Rather than relying on a large pretrained model, I developed a compact CNN from scratch and investigated whether data augmentation could improve its performance.



The project combines \*\*PySpark-based exploratory data analysis\*\*, image preprocessing, deep learning and detailed model evaluation.



\## The Approach



The investigation followed five stages:



\*\*01 · Explore\*\* — Examine image quality, class distribution, colour characteristics and potential data leakage using Python and PySpark.



\*\*02 · Prepare\*\* — Normalise image pixels and create reproducible, stratified training, validation and test sets.



\*\*03 · Build\*\* — Train a compact CNN with three convolutional layers and 102,154 trainable parameters.



\*\*04 · Improve\*\* — Introduce random flips and rotations to investigate the effect of data augmentation.



\*\*05 · Evaluate\*\* — Examine accuracy, precision, recall, F1-scores, confusion patterns and prediction confidence.



\## Results



| Metric | Result |

|:---|---:|

| Baseline validation accuracy | 76.42% |

| Augmented validation accuracy | \*\*79.26%\*\* |

| Improvement | \*\*+2.84 percentage points\*\* |

| Final test accuracy | \*\*80.77%\*\* |

| Macro F1-score | 79.42% |

| Weighted F1-score | 80.12% |


### Where the Model Gets Confused

![CNN confusion matrix](figures/cnn_confusion_matrix.png)

The augmented model correctly classified \*\*3,271 of 4,050 test images\*\*.



\### What I Learned



The CNN performed particularly well on \*\*Forest, Residential and SeaLake\*\* images.



\*\*Highway\*\* proved considerably more difficult, achieving just \*\*32.53% recall\*\*. Many Highway images were confused with River and vegetation-related categories.



This highlighted an important lesson: a model can achieve reasonable overall accuracy while still struggling with specific classes.



\## Dataset

### A Look at the Dataset

![EuroSAT satellite image samples](figures/sample_images_by_category.png)

The project uses the \[EuroSAT RGB dataset](https://zenodo.org/records/7711810), developed from Sentinel-2 satellite imagery.


\- \*\*27,000 images\*\*, each measuring 64 × 64 pixels.

\- \*\*10 land-cover categories\*\*, including agricultural, urban, forest and water environments.

\- \*\*70% / 15% / 15%\*\* stratified training, validation and test split.



All images passed the initial integrity checks, and no byte-identical duplicates were detected.



The dataset is downloaded separately and is not stored in this repository.



\## Technology Stack



`Python` · `PySpark` · `TensorFlow / Keras` · `Pandas` · `NumPy` · `Matplotlib` · `Scikit-learn` · `Pillow` · `JupyterLab`



\## Getting Started



\*\*1. Clone the repository\*\* and open its main directory.



\*\*2. Create an Anaconda environment:\*\*



```bash

conda create -n eurosat_cnn python=3.12

conda activate eurosat_cnn

```



\*\*3. Install the dependencies:\*\*



```bash

python -m pip install -r requirements.txt

python -m pip install jupyterlab

```



PySpark also requires a compatible Java installation.



\*\*4. Download EuroSAT RGB\*\* from the \[official dataset repository](https://zenodo.org/records/7711810) and place the ZIP archive at:



```text

data/raw/EuroSAT\_RGB.zip

```



\*\*5. Launch JupyterLab:\*\*



```bash

jupyter lab

```



Open `notebooks/PDAN8412w\_Part3\_CNN\_EuroSAT.ipynb` and execute the cells in order.



The notebook documents data preparation, model training and evaluation. Reproduced results may vary slightly between environments.



\## Limitations \& Next Steps



Although augmentation improved validation accuracy, the results are limited to the EuroSAT benchmark.



The available data does not establish geographic independence between partitions, and the experiment was not repeated across multiple random seeds.



Future work could explore \*\*transfer learning, geographically independent testing and Grad-CAM visual explanations\*\*.



\## References



The project builds on the \[EuroSAT research by Helber et al. (2019)](https://doi.org/10.1109/JSTARS.2019.2918242).



Further academic references and methodological explanations are provided in the notebook.



\---



\*\*Tumisi Mosoeu\*\*  

Postgraduate Diploma in Data Analytics  

Programming for Data Analytics 2 · PDAN8412/w · 2026

