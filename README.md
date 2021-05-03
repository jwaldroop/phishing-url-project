# phishing-url-project
This project is being performed by MSBA students Jordan Waldroop and Jack Beck.
The goal is to work on improving a current URL phishing detection algorithm, which has the machine learning goal of creating a model that can predict whether or not a given URL will lead to a phishing website, based on the characters within the URL itself.

The original algorithm and associated datasets come from the following journal: https://www.sciencedirect.com/science/article/pii/S2352340920313202

The current version of this algorithm has produced two dataset variations, one with 58,645 and the other with 88,647 labeled URLs.
It is very likely that we will continue to add other various datasets to our project to help improve our model performance.
Thank you to the original authors of the data article for paving the way for our project.
This project is a WIP and will be updated over time.  

---

## Project Structure and Methods

---

## Random Forest Model

Jack Beck was in charge of this portion of the project.

### Running the Model
To run the deployable RF model, ensure that all of the required libraries are installed by using the command 'pip install -r RF_requirements.txt'. Next, extract 'final-RF-model.pkl' from 'final-RF-model.zip'. From here ensure that the script 'deployable_RF_model.py' and the pickled model 'final-model.pkl' are in the same directory, then in a command line, simply navigate to the directory that contains both of those files and run the script with the following command: 'python deployable_RF_model.py'. The data should be loaded from GitHub automatically, so there is no need to have the data in the working directory.

### Data split
For the dataset split, the train_test_split function from the sklearn library was used to do a 75/25 split with a random_state of 426.

### Feature Selection
In the creation of this final model, several different iterations were created to identify feature importance and to select which features to use in the final model. Initially, a comprehensive model was created which used all 111 features. From here, 6 individual models were created to test out the various feature 'table' subsets that were outlined in the original data article. With those models evaluated, it was decided that the best course of action would be to use the initial comprehensive model, and to refine it using the recursive feature elimination function from the sklearn.feature_selection module. This narrowed the final model's feature set down to the 50 most important features. All of the various model iterations are present in the 'RandomForest_Final.ipynb' Jupyter notebook.

### Model Structure
The structure of this random forest model is relatively straight forward. The final model uses the top 50 features identified by the recursive feature elimination. The model has 200 individual trees, each with a maximum depth of 15.

### Model Evaluation
The model was evaluated to create the following performance metrics and charts: Mean Accuracy, F-1 Score, Mean Precision, ROC Curve, Precision-Recall curve. All of these evaluation metrics will be generated upon running the final model script. Model accuracy was then verified using a 5-fold cross validation.

### Analyzing Feature Importance
To get a better understanding of the relative importance of various features in the final model, use of the SHAP library was performed. This allowed us to create visualizations to better understand feature significance using each features respective SHAP value. The SHAP calculations are very computationally intensive, and have been removed from the final .py script in a effort to reduce runtime. The relevant SHAP graphs can be found in this model's Jupyter notebook. Note that SHAP is NOT installed with 'RF_requirements.txt', so you will have to install it separately in you Jupyter environment. To learn more about SHAP, check out the following links:
- https://github.com/slundberg/shap
- https://christophm.github.io/interpretable-ml-book/shap.html

---

## Neural Network Model

Jordan Waldroop was in charge of this portion of the project.


To run the .py file, the .py file and the dataset (or any other compatible dataset) will first need to be downloaded. Then, open a Powershell window and navigate to the folder containing the .py file and the dataset. Run the command ```python deployable_nn.py --inputfile dataset_full.csv ```. You will need to be sure you have TensorFlow and Keras installed (```pip install tensorflow``` and ```pip install keras```).

My personal runtime for this file is about  minutes since this is computationally intense. This will run on the CPU unless you have a dedicated GPU with CUDA and the NVIDIA Toolkit set up.

The .ipynb file with the final model has a slightly longer runtime, as the model is built two different ways with additional performance metrics included.

### Data split
For the dataset split, the train_test_split function from the sklearn library was used to do a 75/25 split with a random_state of 808.

### Feature Selection
Initially, the neural network model was fit on the full dataset to gain the most insight from the data, as well as insight on how the model structure would need to be optimized for this particular dataset. To further investigate which features may be most important, the model was fit against each table of feature types (that is, 111 features broken up into 6 tables of varying attributes). In addition, Jack used the Recursive Feature Elimination package to select the top 10, 25, and 50 most important features. Once those had been obtained, the model was fit against each. In order to avoid the model from performing continuous learning, the TensorFlow backend session was cleared each time the model was fit.

After reviewing the results of the model against a smaller number of features, the best model version was still the use of the entire dataset and all features.

### Model Structure
The neural network model is a TensorFlow Keras Sequential model. The model consists of 26 layers. These 26 layers are the input and output layers, 8 dense layers, 8 batch normalization layers, 7 dropout layers (0.20 or 0.40), and a flatten layer (directly before the output layer is applied).

Each dense layer and the output layer have activation functions applied in the layer structure. The dense layers have a rectified linear unit (ReLU) activation applied, with the output layer having a Sigmoid activation to keep the predictions range to (0,1).


### Model Compilation
The model was compiled using the Adam algorithm optimizer, in large part because of the algorithm's ability to handle noise and the computational efficiency. The loss function selected for measurement is binary cross-entropy, calculating the cross-entropy loss between true labels and predicted data. This is a probabilistic loss measure. The metrics measured are: binary accuracy and AUC.

A callback was applied to monitor the validation data's maximum binary accuracy with a patience of 25 epochs, in addition to restoring the best weights measured. Having an early stopping callback is integral in this model because of the parameters of the model fit.

### Model Fit
The model fit uses the variables train_X and train_y, with a validation split of 0.30 with shuffle=True. The purpose for validation split and shuffle is for the model to take the training data (75% of the dataset) and further split it (70/30) for model fitting. In addition to now having the data split into thirds (52.5% - training, 22.5% - testing, 30% validation), the training and testing set were shuffled, or certain rows alternated between training and testing, after each epoch.

It is important to note that the original validation split for X and y *are not* used at all during the model fit. The data split ends up being 52.5% training, 22.5% validation, and the remaining 25% being held out for model evaluation and predictions. This has been done in the hopes of mitigating overfitting.

The batch size (how many rows are fed into the model at a time) is set to 1500. The model is set to run for 250 epochs - this is where the early stopping callback function plays an important role. I didn't change the Adam optimizer algorithm's learning rate from 0.001, to retain algorithm integrity. With the early stopping callback applied, no matter the number of features, the model never came close to reaching the 200th epoch.

### Model Evaluation
Again using the TensorFlow Keras library and the evaluate() function, after the model had completed fitting, both the testing and validation data were evaluated on what the model had learned during fitting. This is the first time the model encountered the original 25% validation data. The evaluate() function returned the same metrics that the model measured during fitting - binary cross-entropy as 'loss', binary accuracy, and AUC.

### Model Predictions
Predictions were done using the TensorFlow Keras function predict(), returning probabilistic predictions on the validation data, indicating the likelihood that a URL is benign or phishing. In functionality, you could apply the predict function to the entire dataset, but it is inadvisable to ask the model to make predictions on data that it was already trained on.

### Model Built as Function + Cross-Validation Scoring & Predictions
Because we hope to transform our models into a deployable format for API/website development, I additionally programmed the model as a function and built it using Keras' KerasClassifier() sklearn wrapper with the model as the build function.

A 5-fold cross-validation scoring was ran against the model to ensure accuracy and to mitigate overfitting. Further, after the cross-validation scoring was completed, I had the model predict using a 5-fold cross validation prediction.


---
