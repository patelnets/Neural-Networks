import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import csv

from sklearn.utils import class_weight
from tensorflow import keras
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix

class FNN:

    def __init__(self, hp):
        self.hp = hp
        self.model=None
        self.create_ANN()

    def create_ANN(self):
        model = keras.Sequential()

        model.add(keras.layers.Dense(self.hp['hidden_layer_n'][0], input_shape=(3,)))
        model.add(keras.layers.Activation(self.hp['hidden_layer_af'][0]))

        for n, af, dr in zip(self.hp['hidden_layer_n'][1:], self.hp['hidden_layer_af'][1:], self.hp['dropout_rates']):
            model.add(keras.layers.Dense(n))
            model.add(keras.layers.Activation(af))
            model.add(keras.layers.Dropout(dr))

        model.add(keras.layers.Dense(4))
        model.add(keras.layers.Activation('softmax'))
        model.compile(optimizer=self.hp['optimizer'],
                      loss=self.hp['loss_function'],
                      metrics=['accuracy'])
        self.model = model
        return model

    def evaluate_architecture(self, x_train_val, y_train_val):
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)
        accuracy_scores = []
        f1_scores = []
        count = 1
        for train, val in kfold.split(x_train_val, y_train_val):
            history = self.train_model(x_train_val[train], y_train_val[train], x_train_val[val], y_train_val[val])
            scores = self.model.evaluate(x_train_val[val], y_train_val[val], verbose=0)
            accuracy_scores.append(scores[1] * 100)
            f1 = self.get_average_f1_score(x_train_val[val], y_train_val[val])
            f1_scores.append(f1)
            count += 1
            self.print_model_metrics(history, x_train_val[val], y_train_val[val])
            # self.print_classification_report(x_train_val[val], y_train_val[val])
        print("Average accuracy, sd accuracy, f1_score: ", np.mean(accuracy_scores), np.std(accuracy_scores), np.mean(f1_scores))
        return np.mean(accuracy_scores), np.std(accuracy_scores), np.mean(f1_scores)

    def print_model_metrics(self,  history, x_test, y_test):
        print('\n\nTest metrics')
        print(self.model.metrics_names)
        self.model.evaluate(x_test, y_test, batch_size=128)

        plt.plot(history.history['acc'], label='Training')
        plt.plot(history.history['val_acc'], label='Validation')
        plt.legend(loc='right')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig('architecture_accuracy.png')
        plt.show()

        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.legend(loc='right')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('architecture_loss.png')
        plt.show()

    def get_average_f1_score(self, x, y):
        prediction = tf.one_hot(tf.argmax(self.model.predict(x), dimension=1), 4)
        sess = tf.Session()
        with sess.as_default():
            p = prediction.eval()
        test = [np.where(r == 1)[0][0] for r in y]
        pred = [np.where(r == 1)[0][0] for r in p]
        return f1_score(y_true=test, y_pred=pred, average='macro')

    def print_classification_report(self, x_test, y_test):
        prediction = tf.one_hot(tf.argmax(self.model.predict(x_test), dimension=1), 4)
        print('===== Classification Report =====')
        print(self.hp)
        sess = tf.Session()
        with sess.as_default():
            print()
            regions = ['Zone 1', 'Zone 2', 'Ground', 'Rest of Space']
            cr = classification_report(y_test, prediction.eval(), target_names=regions)
            print(cr)
            p = prediction.eval()

        # Compute confusion matrix
        print("Confusion Matrix: ")
        test= [np.where(r == 1)[0][0] for r in y_test]
        pred= [np.where(r == 1)[0][0] for r in p]
        cnf_matrix = confusion_matrix(test, pred)
        print(f1_score(y_true=test, y_pred=pred, average='macro'))
        print(cnf_matrix)

    def train_model(self, x_train, y_train, x_val, y_val):
        self.create_ANN()
        sample_weights = np.sqrt(class_weight.compute_sample_weight(class_weight='balanced', y=y_train))
        return self.model.fit(x_train, y_train,
                       epochs=self.hp["epochs"],
                       batch_size=self.hp["batch_size"],
                       validation_data=((x_val, y_val)),
                       callbacks=self.hp["callbacks"],
                       sample_weight=sample_weights,
                       verbose=0)

    def train_final_model(self, x_train, y_train):
        self.create_ANN()
        sample_weights = np.sqrt(class_weight.compute_sample_weight(class_weight='balanced', y=y_train))
        self.model.fit(x_train, y_train,
                      epochs=self.hp["epochs"],
                      batch_size=self.hp["batch_size"],
                      callbacks=self.hp["callbacks"],
                      sample_weight=sample_weights,
                      verbose=0)

def evaluate_base_architecture(x_train_val, y_train_val):
    hp = {'loss_function': 'categorical_crossentropy',
          'epochs': 80,
          'callbacks': [],
          'batch_size': 32,
          'optimizer': 'sgd',
          'hidden_layer_n': [6],
          'dropout_rates': [0],
          'hidden_layer_af': ['relu']
          }
    fnn = FNN(hp)
    mean_accuracy, std_accuracy, f1_score = fnn.evaluate_architecture(x_train_val, y_train_val)
    print(fnn.hp)
    print("Average accuracy, sd accuracy, f1_score: ", mean_accuracy, std_accuracy, f1_score)

def predict_hidden(dataset):
    np.random.shuffle(dataset)
    x_data = dataset[:, :3]
    with open('ROI_model_architecture.json', 'r') as f:
        model = tf.keras.models.model_from_json(f.read())
    model.load_weights('ROI_model_weights.h5')

    prediction = tf.one_hot(tf.argmax(model.predict(x_data), dimension=1), 4)
    sess = tf.Session()
    with sess.as_default():
        output_prediction = prediction.eval()

    return output_prediction

def get_random_hyperparameters():
    hp = {
        "loss_function": 'categorical_crossentropy',
        "callbacks": [],
        "batch_size": 64
    }
    learning_rates = [0.001, 0.0025, 0.005, 0.0075, 0.01]
    lr = random.choice(learning_rates)
    adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                 amsgrad=False)
    dropout_rates = [0, 0.5]
    dr = []

    epoch = [50, 80, 100]
    n_layers = random.randint(1, 3)

    n_neurons= []
    afs = ['relu', 'tanh', 'sigmoid']
    acitvation_functions = []
    for x in range(n_layers):
        n_neurons.append(random.randint(5, 10))
        acitvation_functions.append((random.choice(afs)))
        dr.append(random.choice(dropout_rates))

    hp["epochs"] = random.choice(epoch)
    hp["optimizer"] = adam
    hp["learning_rate"] = lr
    hp["dropout_rates"] = dr
    hp["hidden_layer_n"] = n_neurons
    hp["hidden_layer_af"] = acitvation_functions

    return hp

def randomised_hp_search(x_train_val, y_train_val):
    for i in range(100):
        hp = get_random_hyperparameters()
        fnn = FNN(hp)
        mean_accuracy, std_accuracy, f1_score = fnn.evaluate_architecture(x_train_val, y_train_val)

        if mean_accuracy > 0.96 and f1_score > 0.84:
            # Save the weights
            weights_filepath = 'model_weights' + str(i) + '.h5'
            fnn.model.save_weights(weights_filepath)

            # Save the model architecture
            architecture_filepath = 'model_architecture' + str(i) +'.h5'
            with open(architecture_filepath, 'w') as f:
                f.write(fnn.model.to_json())

            # Write results to a csv for comparison
        with open('Randomized Search.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(
                [mean_accuracy, f1_score, hp['epochs'], hp['hidden_layer_n'], hp['hidden_layer_af'],
                 hp['learning_rate'],
                 hp['dropout_rates'], hp])


def evaluate_final_model(x_train_val, y_train_val):
    adam = keras.optimizers.Adam(lr=0.0025, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                 amsgrad=False)
    hp = {'loss_function': 'categorical_crossentropy',
          'epochs': 100,
          'callbacks': [],
          'batch_size': 32,
          'optimizer': adam,
          'learning_rate': 0.0025,
          'dropout_rates': [0, 0.5],
          'hidden_layer_n': [10, 10],
          'hidden_layer_af': ['tanh', 'tanh']
          }
    fnn = FNN(hp)
    mean_accuracy, std_accuracy, f1_score = fnn.evaluate_architecture(x_train_val, y_train_val)
    print(fnn.hp)
    print("Average accuracy, sd accuracy, f1_score: ", mean_accuracy, std_accuracy, f1_score)

def train_final_model(dataset):
    adam = keras.optimizers.Adam(lr=0.0025, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                 amsgrad=False)
    hp = {'loss_function': 'categorical_crossentropy',
          'epochs': 100,
          'callbacks': [],
          'batch_size': 64,
          'optimizer': adam,
          'learning_rate': 0.0025,
          'dropout_rates': [0, 0.5],
          'hidden_layer_n': [10, 10],
          'hidden_layer_af': ['tanh', 'tanh']
          }
    fnn = FNN(hp)
    fnn.train_final_model(dataset[:, :3], dataset[:, 3:])
    fnn.model.save_weights('ROI_model_weights.h5')
    with open('ROI_model_architecture.json', 'w') as f:
        f.write(fnn.model.to_json())

def split_dataset(dataset):
    np.random.shuffle(dataset)
    x_data, y_data = dataset[:, :3], dataset[:, 3:]
    x_train_val, x_test, y_train_val, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    return x_train_val, x_test, y_train_val, y_test


def main(dataset_filepath):

    dataset = np.loadtxt(dataset_filepath)
    x_train_val, x_test, y_train_val, y_test = split_dataset(dataset)

    # # Base architecture
    # evaluate_base_architecture(x_train_val, y_train_val)

    # # Randomized search for hyper parameters
    # print("Randomise searching for hidden parameter")
    # randomised_hp_search(x_train_val, y_train_val)

    # Final Model
    # print("Evaluating the final model")
    # evaluate_final_model(x_train_val, y_train_val)

    # # Train final model
    # print("Training final model")
    # train_final_model(dataset)

    # Predict hidden dataset
    print("Predicting hidden dataset")
    prediction = predict_hidden(dataset)

if __name__ == "__main__":
    main(sys.argv[1])
