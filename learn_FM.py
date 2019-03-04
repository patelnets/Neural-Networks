import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
import sklearn
import keras
import pickle
import scipy

from illustrate import illustrate_results_FM_tensorflow


def load_pickle(filePath):
    with open(filePath, 'rb') as fp:
        data = pickle.load(fp)
    return data

def load_data(dataFilePath):
    dataset = np.loadtxt(dataFilePath)
    datasetDataFrame = pd.DataFrame(dataset, columns=['theta1', 'theta2', 'theta3','x','y','z'])
    datasetDataFrame = sklearn.utils.shuffle(datasetDataFrame, random_state=2)
    trainDataset = datasetDataFrame.sample(frac=0.8,random_state=1)
    testDataset = datasetDataFrame.drop(trainDataset.index)
    return trainDataset, testDataset
    
class log_uniform():
    def __init__(self, a=-1, b=0, base=10):
        self.loc = a
        self.scale = b - a
        self.base = base
    def rvs(self, size=None, random_state=None):
        uniform = scipy.stats.uniform(loc=self.loc,scale=self.scale)
        if size is None:    
            return np.power(self.base, uniform.rvs(random_state=random_state))
        else:
            return np.power(self.base, uniform.rvs(size=size, random_state=random_state))

class Normalisation():

    def __init__(self, dataset):
        self.mean = 0
        self.std = 0
        datasetStats = dataset.describe()
        datasetStats = datasetStats.transpose()
        self.min = datasetStats['min']
        self.max = datasetStats['max']


    def norm(self, dataset):
        return (dataset - self.min) / (self.max- self.min)

    def deNorm(self, dataset):
        return (dataset * (self.max-self.min)) + self.min

class Forward_Model():

    def __init__(self, trainDataset=None, testDataset=None):
        self.originalTrainDataset = trainDataset.copy()
        self.originalTestDataset = testDataset.copy()

        self.trainDataset = trainDataset
        self.testDataset = testDataset
        self.trainDataLabels = None
        self.testDataLabels = None
    
        self.normaliseYTest = None 
        self.normaliseXTest = None 
        self.normaliseXTrain = None 
        self.normaliseYTrain = None 

        self.normedTrainData = None 
        self.normedTestData = None 
        self.normedTrainDataLabels = None 
        self.normedTestDataLabels = None 

        self.model = None
        self.history = None
        self.score = None

        self.prep_data()

    def prep_data(self):
        self.trainDataLabels = pd.concat([self.trainDataset.pop(x) for x in ['x','y','z']], 1)
        self.testDataLabels = pd.concat([self.testDataset.pop(x) for x in ['x','y','z']], 1)
        
        self.normaliseXTrain = Normalisation(self.trainDataset) 
        self.normaliseYTrain = Normalisation(self.trainDataLabels) 

        self.normedTrainData = self.normaliseXTrain.norm(self.trainDataset)
        self.normedTestData = self.normaliseXTrain.norm(self.testDataset)
        self.normedTrainDataLabels = self.normaliseYTrain.norm(self.trainDataLabels)
        self.normedTestDataLabels = self.normaliseYTrain.norm(self.testDataLabels)
  
    def get_normalised_whole_set(self):
        frames = [self.originalTestDataset, self.originalTrainDataset]
        dataSet = pd.concat(frames)
        dataLabels = pd.concat([dataSet.pop(x) for x in ['x','y','z']], 1)

        normaliseX = Normalisation(dataSet)
        normaliseY = Normalisation(dataLabels)

        return normaliseX.norm(dataSet), normaliseY.norm(dataLabels)

    def create_model(self, neurons1=7, neurons2=18, neurons3=19, neurons4=14, neurons5=12, numberOfLayers=4, activation=tf.keras.activations.elu, learnRate=0.0013, optimizer='adam'):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=neurons1, activation=activation, input_dim=self.normedTrainData.shape[1]))
        if numberOfLayers > 1:
            model.add(tf.keras.layers.Dense(units=neurons2, activation=activation))
        if numberOfLayers > 2:
            model.add(tf.keras.layers.Dense(units=neurons3, activation=activation))
        if numberOfLayers > 3:
            model.add(tf.keras.layers.Dense(units=neurons4, activation=activation))
        if numberOfLayers > 4:
            model.add(tf.keras.layers.Dense(units=neurons5, activation=activation))
        model.add(tf.keras.layers.Dense(units=3))

        if optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr=learnRate)
        if optimizer == 'rmsProp':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learnRate)
        if optimizer == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(lr=learnRate)

        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
        return model 

    def fit_model(self, epochs, batchSize):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=150)
        self.history = self.model.fit(self.normedTrainData, self.normedTrainDataLabels, batch_size=batchSize, epochs=epochs, validation_split = 0.2, verbose=0, callbacks=[early_stop])

    def illustrate_model(self):
        illustrate_results_FM_tensorflow(model=self.model)

    def plot_epoch_vs_accuracy_and_loss(self):
        plt.figure(3)
        plt.title("Training Loss and validation loss vs Epoch")
        plt.plot(self.history.history['loss'], label='Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        plt.show()

    def save_model(self, saveModelName):
        self.model.save("{saveModelName}.h5".format(saveModelName=saveModelName)) 

    def load_model(self, loadModelName):
        self.model = tf.keras.models.load_model("{loadModelName}.h5".format(loadModelName=loadModelName))

    def evaluate_architecture(self, model, epochs, batchSize):
        if not model: 
            self.model = self.create_model()
        else:
            self.model = model
        self.fit_model(epochs=epochs, batchSize=batchSize)
        self.score = self.model.evaluate(self.normedTestData, self.normedTestDataLabels, verbose=0)
        predicted = self.model.predict(self.normedTestData)
        self.plot_epoch_vs_accuracy_and_loss()
        print('Normalised Test set mean squared error: ', self.score[1]) 

        predicted = pd.DataFrame(predicted, columns=['x','y','z'])
        
        mse = np.sqrt(sklearn.metrics.mean_squared_error(self.normaliseYTrain.deNorm(self.normedTestDataLabels), self.normaliseYTrain.deNorm(predicted)))
        print("Unnormalised root mean squared error: ", mse)     

    def train_and_save_model_final_model(self):
        normedDataSet, normedDataLabels = self.get_normalised_whole_set()
        self.model = self.create_model()
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=150)
        self.history = self.model.fit(normedDataSet, normedDataLabels, batch_size=32, epochs=600, validation_split = 0.2, verbose=0, callbacks=[early_stop])
        self.save_model(saveModelName='final_fw_model')

    def hyperparameter_search_epoch_batch(self, fileToSavePath):

        model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=self.create_model, verbose=0)
        batch_size = [32,64,128]
        epochs = [100,200,300,400,500,600,700]
        neurons1 = scipy.stats.randint(1,21) 
        neurons2 = scipy.stats.randint(1,21)
        neurons3 = scipy.stats.randint(1,21)
        neurons4 = scipy.stats.randint(1,21)
        neurons5 = scipy.stats.randint(1,21)
        activation = [tf.nn.relu , tf.nn.crelu, tf.keras.activations.elu, tf.nn.leaky_relu]
        learnRate = log_uniform(a=-4, b=0)
        optimizer = ['rmsProp', 'adam', 'adadelta']
        numberOfLayers = [1,2,3,4,5]
        param_grid = dict(activation=activation, numberOfLayers=numberOfLayers, batch_size=batch_size, epochs=epochs, neurons5=neurons5, neurons4=neurons4, neurons3=neurons3, neurons2=neurons2, neurons1=neurons1, learnRate=learnRate, optimizer=optimizer)
        grid = sklearn.model_selection.RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs=-1, n_iter=100, cv=5, verbose=10)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=150)
        grid_result = grid.fit(self.normedTrainData, self.normedTrainDataLabels, callbacks=[early_stop])

        print("Best: {score} using {bestParam}".format(score=grid_result.best_score_, bestParam=grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

        with open(fileToSavePath, 'wb') as fp:
            pickle.dump(dict(means=means, stds=stds, params=params), fp, protocol=pickle.HIGHEST_PROTOCOL)

    def print_random_hyper_search_iterations(self, filePath):
        hyperParameterZip = load_pickle(filePath=filePath)

        means = hyperParameterZip['means']
        stds = hyperParameterZip['stds']
        params = hyperParameterZip['params']
        for i in range(len(means)):
            print("Mean loss of {mean} and std dev {stdev} with hyper parameters: {param}".format(mean=means[i], stdev=stds[i], param=params[i]))
  
    def spearmens_rank_correlation_hyper_search(self, filePath):
        hyperParameterZip = load_pickle(filePath=filePath)
        parameters = list(hyperParameterZip['params'][0].keys())

        ranks = []
        for parameter in parameters:
            meanLoss = []
            parameterValue = []
            for iteration in range(len(hyperParameterZip['means'])):
                meanLoss.append(hyperParameterZip['means'][iteration])
                parameterValue.append(hyperParameterZip['params'][iteration][parameter]) 
            rank, _ = scipy.stats.spearmanr(parameterValue, meanLoss)
            ranks.append(abs(rank))

        print("Highest Ranking parameter: '{parameter}' with value: {value}".format(parameter=parameters[np.argmax(ranks)], value=max(ranks)))
        print("All parameters: ", parameters)
        print("All paramter ranks: ", ranks)


def evaluate_architecture(model=None, epochs=600, batchSize=32):
    trainDataset, testDataset = load_data(dataFilePath="FM_dataset.dat")
    forwardModel = Forward_Model(trainDataset=trainDataset, testDataset=testDataset)
    forwardModel.evaluate_architecture(model=model, epochs=epochs, batchSize=batchSize)

def predict_hidden(datasetFilePath):
    trainDataset, testDataset = load_data(dataFilePath="FM_dataset.dat")
    forwardModel = Forward_Model(trainDataset=trainDataset, testDataset=testDataset)
    dataset = np.loadtxt(datasetFilePath)
    if dataset.shape[1] == 6:
        predictDataFrame = pd.DataFrame(dataset, columns=['theta1', 'theta2', 'theta3','x','y','z'])
        predictDataLabels = pd.concat([predictDataFrame.pop(x) for x in ['x','y','z']], 1)
    elif dataset.shape[1] == 3:
        predictDataFrame = pd.DataFrame(dataset, columns=['theta1', 'theta2', 'theta3'])
    else:
        raise RuntimeError("Data in file is not in the right form!")
    normPredictDataFrame = forwardModel.normaliseXTrain.norm(predictDataFrame) 
    model = tf.keras.models.load_model("{modelName}.h5".format(modelName="final_fw_model"))
    predictedNormed = model.predict(normPredictDataFrame) 
    predictedNormed = pd.DataFrame(predictedNormed, columns=['x','y','z'])
    predicted = forwardModel.normaliseYTrain.deNorm(predictedNormed)
    print(predicted.values)
    return predicted.values

def main():
    """ Uncomment this to run evaluate architecture. This has the following parameters 
        model: Optional - By default this will use our initial architecture. To evaluate your own 
                architecture set this to equal your compiled tensorflow model.
        epochs: Optional - By default this is set to 600. Takes an int value
        batchSize: Optional - By default this is set to 32. Takes an int value. 
    """
    # evaluate_architecture()

    """ Uncomment this to run the predict hidden method. Change the file path to your dataset file path 
    """
    # predict_hidden(datasetFilePath="FM_dataset.dat")


    """ This is what we used to run and save the results of our hyper parameter search. 
    """
    # trainDataset, testDataset = load_data(dataFilePath="FM_dataset.dat")
    # forwardModel = Forward_Model(trainDataset=trainDataset, testDataset=testDataset)
    # forwardModel.hyperparameter_search_epoch_batch(fileToSavePath='march.p')
    # forwardModel.spearmens_rank_correlation_hyper_search(filePath='march.p')
    # forwardModel.print_random_hyper_search_iterations(filePath='march.p')

    """ This is what we used to save our final model. 
    """
    trainDataset, testDataset = load_data(dataFilePath="FM_dataset.dat")
    forwardModel = Forward_Model(trainDataset=trainDataset, testDataset=testDataset)
    forwardModel.train_and_save_model_final_model()

if __name__ == "__main__":
    main()

