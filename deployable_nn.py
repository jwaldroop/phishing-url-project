import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

def main():
  # Process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputfile', help="The file with the features and labels.")
    FLAGS = parser.parse_args()

    with open(FLAGS.inputfile) as f:
        reader = pd.read_csv(f)
    return reader

main()

print('Data Loaded')

def unit_test():
    print('Start Unit Testing')
    reader = main()
    #unit testing
    reader.dtypes == 'int64'
    def remove_negatives(df):
        df[df == -1] = 0
    remove_negatives(reader)

    is_it_working = []
    def data_cleaning_unit_test(column):
        did_it_work = {'Yes' : 0, 'No' : 0}
        for i in column:
            if i >= 0:
                did_it_work['Yes'] += 1
            elif i < 0:
                did_it_work['No'] += 1
        if did_it_work['No'] > 0:
            print(column.name, '=', 'Not working')
        else:
            print(column.name, '=', 'It worked!')

    for col in reader.columns.tolist():
        data_cleaning_unit_test(reader[col])
    return

unit_test()

def data_time():
    reader = main()
    #set X and y
    y = reader.iloc[:,-1]
    X = reader.iloc[:,0:111]
    return X, y

print('X and y are set.')

def train_test():
    X, y = data_time()
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.25, random_state=808)
    return train_X, val_X, train_y, val_y

print('train_test_split completed')

tf.keras.backend.clear_session()
print('TensorFlow Backend Session Cleared')

print("For a more detailed output, change verbosity to 'verbose = 1'")


def phish_nn():
    print('Starting Model Build')
    model = keras.Sequential()
    train_X, val_X, train_y, val_y = train_test()
    model.add(tf.keras.layers.InputLayer(input_shape=[111]))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=50, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.20))
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=16, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.40))
    model.add(tf.keras.layers.Dense(units=16, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.40))
    model.add(tf.keras.layers.Dense(units=111, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', mode = 'max', patience = 25, restore_best_weights = True)
    print('Starting Model Fit')
    history = model.fit(train_X, train_y, validation_split=0.30, batch_size=1500, epochs=250,
                        callbacks = [earlystopping], workers=8, verbose=0)
    predictions = (model.predict(val_X) > 0.50).astype('int32')
    val_scores = model.evaluate(val_X, val_y) #give the model the ability to evaluate like was done above
    #saved_model = model.save(filepath='./deploy-nn/saved_deployable_nn/', include_optimizer=True, overwrite=True)
    train_scores = model.evaluate(train_X, train_y) #give the model the ability to evaluate like was done above
    print('Training Binary Cross Entropy: \n', round(((train_scores[0])*100),2), '%')
    print('Training Accuracy: \n', round(((train_scores[1])*100),2), '%')
    print('Validation Binary Cross Entropy: \n', round(((val_scores[0])*100),2), '%')
    print('Validation Accuracy: \n', round(((val_scores[1])*100),2), '%')
    print('Predictions: \n', predictions)
    cm = confusion_matrix(val_y, predictions)
    print('Confusion Matrix: \n', cm)
    return model

phish_nn()

print("For a more detailed output, change verbosity to 'verbose = 1'")

def cross_val_nn():
    print('Cross validation initializing')
    X, y = data_time()
    mod = KerasClassifier(build_fn = phish_nn, epochs = 15, batch_size = 1500, verbose=0)
    num_folds = 5
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=808)
    print('Starting 5-fold cross validation of model')
    cv_results = cross_val_score(mod, X, y, cv=kfold)
    print('Starting 5-fold cross-validation predictions')
    cv_preds = cross_val_predict(mod, X, y, cv=kfold, verbose=0, method='predict')
    print('The average cross-validation accuracy is: ', round(cv_results.mean(), 4)*100, '%')
    print('The 5-fold cross validation accuracy results are: \n', cv_results)
    acc = accuracy_score(y, cv_preds)
    cm = confusion_matrix(y, cv_preds)
    print('Confusion Matrix \n', cm)
    print('Accuracy Score: \n', acc)
    f1s = f1_score(y, cv_preds)
    print('The F1 score for the cross validated model is: \n', f1s)
    precis = precision_score(y, cv_preds, average='binary')
    rec = recall_score(y, cv_preds, average='binary')
    print('The precision-recall score is: \n', precis)
    print('The recall score is: \n', rec)
    return cm

cm = cross_val_nn()
holup2 = sns.heatmap(cm, annot=True, fmt='d', linewidths=0.5,
            xticklabels=True, yticklabels=True)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()


print('Script Finished. Thank you :)')
