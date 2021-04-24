import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def main():
  # Process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputfile', help="The file with the features and labels.")
    FLAGS = parser.parse_args()

    with open(FLAGS.inputfile) as f:
        reader = pd.read_csv(f)

    #unit testing
    reader.dtypes == 'int64'
    def remove_negatives(df):
        df[df == -1] = 0
    remove_negatives(reader)

    #set X and y
    y = reader.iloc[:,-1]
    X = reader.iloc[:,0:111]
    X = tf.keras.utils.normalize(X, axis=-1, order=2)
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.25, random_state=808)
    return train_X, val_X, train_y, val_y

main()

tf.keras.backend.clear_session()


def phish_nn():
    train_X, val_X, train_y, val_y = main()
    model = keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[111]))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=50, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.20))
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=16, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.40))
    model.add(tf.keras.layers.Dense(units=16, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.40))
    model.add(tf.keras.layers.Dense(units=111, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', mode = 'max', patience = 15, restore_best_weights = True)
    history = model.fit(train_X, train_y, validation_split=0.30, batch_size= 250, epochs=500, callbacks = [earlystopping], workers=8)
    predictions = model.predict(val_X)
    val_scores = model.evaluate(val_X, val_y) #give the model the ability to evaluate like was done above
    #saved_model = model.save(filepath='./deploy-nn/saved_deployable_nn/', include_optimizer=True, overwrite=True)
    train_scores = model.evaluate(train_X, train_y) #give the model the ability to evaluate like was done above
    print('Training Binary Cross Entropy: \n', round(((scores[0])*100),2), '%')
    print('Prediction probabilities: \n', predictions)
    print('Validation Binary Crossentropy: \n', round(((scores[0])*100),2), '%')
    print('Validation Accuracy: \n', round(((val_scores[1])*100),2), '%')
    print('Training Accuracy: \n', round(((train_scores[1])*100),2), '%')
    return model
phish_nn()
