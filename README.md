# CNN_MIC_Prediction
Welcome to the _CNN_MIC_Model_ repository. This Convolutional Neural Network model has been designed to predict antibiotics MIC values of Klebsiella pneumoniae using almagated data of Kpn WGS and antibiotics SMILES.
## Background
### Feature Engineering & Data Processing
**K-mer Strategy:** Our approach toward feature engineering was rooted in the k-mer strategy. From a dataset containing 1667 genomic samples, we identified 524,800 unique 10-mers.

**Feature Selection:** Directly using such an expansive feature set poses computational constraints. Taking inspiration from the methodology outlined in Nguyen et al., ten XGBoost Regressor models were constructed using 10-fold cross-validation. Filtering these 10-mers based on feature significance, we determined that only 22,209 had a significant feature importance score. This narrowed feature set made CNN model development manageable. The selected 10-mers are indexed in the _useful_10mer_index.txt_ file.

### Encoding & Data Transformation
**Genomic Data Transformation:** Following the procedure in Nguyen et al., we transformed the count values of each 10-mer using one-hot encoding. A logarithmic transformation (base 1.5529) was applied to rescale the count values, which spanned from 0 to 4,281, ensuring they ranged between 0 and 19. Consequently, each WGS sample was encoded into a matrix of size (22,209, 20), where each row pertains to a specific 10-mer sequence, and each column represents the adjusted frequency of the respective 10-mer.

**Antibiotics Data Encoding:** We procured the isomeric SMILES data for antibiotics from the PubChem21 database. This data was transformed into a (130, 20) matrix using one-hot character encoding and then vertically stacked to synchronize with the dimensions of the WGS matrix.

### Final Data Preparation
**Combining & Labeling:** The encoding for each genome-antibiotic pair from our 32,309 samples was achieved by summing the WGS and SMILES matrices and then undergoing a linear scaling operation (division by 2). Each pair was then labeled using the integer value equivalent to the Log2 of the laboratory-derived MIC value. This transformed our challenge into a multi-class classification task. We explored two labeling techniques: a conventional one-hot encoding of the exact label and a soft labeling strategy that highlighted 1-tier accurate labels close to the precise label to differentiate them from inaccurate labels.

![Alt text](./Figure_1_1200.png?raw=true "Data encoding and labeling schematics in CNN based MIC prediction")
