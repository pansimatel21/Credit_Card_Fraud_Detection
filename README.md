# Credit_Card_Fraud_Detection
Credit Card Fraud Detection using Neural Networks (Keras) This involves pattern classification in an unbalanced dataset to determine the fraudulent transactions.

Dataset
The dataset used for this research is collected from Kaggle at https://www.kaggle.com/mlg-ulb/creditcardfraud. It consists of 284,807 transactions that occurred in 2 days, of which 492 are labelled as Fraud. This means that the dataset is highly unbalanced with only 0.172% accounting for the Fraud transactions. It consists of 31 features of which 28 (V1-V28) are the result of PCA transformation, due to confidentiality issues. The remaining features that are not transformed are ‘Time’ and ‘Amount’, which represent the seconds elapsed between each transaction and the first transaction in the dataset and, the transaction amount respectively. The ‘Class’ feature represents the label of the transaction with ‘1’ for a Fraud transaction and ‘0’ for a ‘Genuine’ transaction.

[Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015]

# Experimental Design
# # Algorithm Selection: 
This research is basically based on whether the performance of a Neural Network would significantly differ if one part of the algorithm’s configuration is changed. Firstly, a multi-layer perceptron with 1-hidden layer was trained followed by a multi-layer perceptron with 2 hidden layers. Most of the parameters were tuned using Gridsearch algorithm followed by Trial and Error and, the rest were used as default provided by the Keras library.
