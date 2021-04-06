# phishing-url-project
This project is being performed by MSBA students Jordan Waldroop and Jack Beck.
The goal is to work on improving a current URL phishing detection algorithm, which has the machine learning goal of creating a model that can predict whether or not a given URL will lead to a phishing website, based on the characters within the URL itself.

The original algorithm and associated datasets come from the following journal: https://www.sciencedirect.com/science/article/pii/S2352340920313202

The current version of this algorithm has produced two dataset variations, one with 58,645 and the other with 88,647 labeled URLs.
It is very likely that we will continue to add other various datasets to our project to help improve our model performance.
Thank you to the original authors of the data article for paving the way for our project.
This project is a WIP and will be updated over time.  

---

### Project Structure and Methods

#### Random Forest Model





#### Neural Network Model

Jordan Waldroop was in charge of this portion of the project.

##### Data split and normalization
After assigning X and y for each model, I used the TensorFlow Keras utility library to normalize X. For the dataset split, the train_test_split function from the sklearn library was used to do a 75/25 split with a random_state of 808.

##### Feature Selection
Initially, the neural network model was fit on the full dataset to gain the most insight from the data, as well as insight on how the model structure would need to be optimized for this particular dataset. To further investigate which features may be most important, the model was fit against each table of feature types (that is, 111 features broken up into 6 tables of varying attributes). In addition, Jack used the Recursive Feature Elimination package to select the top 10, 25, and 50 most important features. Once those had been obtained, the model was fit against each. In order to avoid the model from performing continuous learning, the TensorFlow backend session was cleared each time the model was fit.

##### Model Structure
The neural network model is a TensorFlow Keras Sequential model. The model consists of 18 layers. hese 18 layers are the input and output layers, 8 dense layers, 7 dropout layers (0.20 or 0.40), and a flatten layer (directly before the output layer is applied).

Each dense layer and the output layer have activation functions applied in the layer structure. The dense layers have a rectified linear unit (ReLU) activation applied, with the output layer having a Sigmoid activation.

Below is a visualization of the model structure using the ANN Visualizer library, in addition to a table showing the layer showing the input and output shape for each layer (table created using TensorFlow Keras library).

##### Model Compilation
The model was compiled using the Adam algorithm optimizer, in large part because of the algorithm's ability to handle noise and the computational efficiency. The loss function selected for measurement is binary cross-entropy, calculating the cross-entropy loss between true labels and predicted data. This is a probabilistic loss measure. The metrics measured are: binary accuracy, with a threshold of 0.5; and AUC.

A callback was applied to monitor the validation data's maximum binary accuracy with a patience of 25 epochs, in addition to restoring the best weights measured. Having an early stopping callback is integral in this model because of the parameters of the model fit.

##### Model Fit
The model fit uses the variables train_X and train_y, with a validation split of 0.30 with shuffle=True. The purpose for validation split and shuffle is for the model to take the training data (75% of the dataset) and further split it (70/30) for model fitting. In addition to now having the data split into thirds (52.5% - training, 22.5% - testing, 30% validation), the training and testing set were shuffled, or certain rows alternated between training and testing, after each epoch.

It is important to note that the original validation split for X and y *are not* used at all during the model fit. The data split ends up being 52.5% training, 22.5% validation, and the remaining 25% being held out for model evaluation and predictions. This has been done in the hopes of mitigating overfitting.

The batch size (how many rows are fed into the model at a time) is set to 15. The model is set to run for 500 epochs -this is where the early stopping callback function plays an important role. While the model does not have many rows being fed in at once, the high number of epochs with a generous patience level allows the model to have time to learn.I didn't change the Adam optimizer algorithm's learning rate from 0.001, to retain algorithm integrity. With the early stopping callback applied, no matter the number of features, the model never came close to breaking 200 epochs during model fitting, much less 500.

##### Model Evaluation
Again using the TensorFlow Keras library and the evaluate() function, after the model had completed fitting, both the testing and validation data were evaluated on what the model had learned during fitting. This is the first time the model encountered the original 25% validation data. The evaluate() function returned the same metrics that the model measured during fitting - binary cross-entropy as 'loss', binary accuracy, and AUC.

##### Model Predictions
Predictions were done using the TensorFlow Keras function predict(), returning probabilistic predictions on the validation data, indicating the likelihood that a URL is benign or phishing. In functionality, you could apply the predict function to the entire dataset, but it is inadvisable to ask the model to make predictions on data that it was already trained on.


---

### Keras Sequential Model Structure

<img src="model_structure_2.png?raw=true"/>

<img src="model_structure_1.png?raw=true"/>
