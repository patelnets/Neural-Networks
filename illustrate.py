import numpy as np
from simulator import RobotArm
import pandas as pd
from sklearn.utils import shuffle

area_map = {
    k: v
    for k, v in zip(range(4), ("Zone 1", "Zone 2", "Ground", "Unlabelled area"))
}


def softmax(x):
    numer = np.exp(x - x.max(axis=1, keepdims=True))
    denom = numer.sum(axis=1, keepdims=True)
    return numer / denom


def illustrate_results_ROI(network, prep, nb_pos=10):

    data = (
        (np.random.rand(nb_pos + 1, 7) * 2 - 1) * np.pi / 2
    )  # generating 10 cols to match length of dataset, but only the first 3 are used.
    data[0, :] = 0
    data[1:, :3] = prep.apply(data[1:, :3])
    results = network(data[1:, 0:3])
    robot = RobotArm()

    data[1:, 3:7] = results

    data[1:, :3] = prep.revert(data[1:, :3])

    prediction = [area_map[x] for x in np.argmax(data[1:, 3:7], axis=1)]

    angles = np.zeros((nb_pos + 1, 6))
    angles[:, 0:3] = data[:, 0:3]
    ax = None
    for i in range(nb_pos):
        ax = robot.animate(angles[i, :], angles[i + 1, :], ax, [0, 0, 0])
        print("Predicted region: {}".format(prediction[i]))


def illustrate_results_FM(network, prep, nb_pos=10):

    data = (
        (np.random.rand(nb_pos + 1, 6) * 2 - 1) * np.pi / 2
    )  # generating 10 cols to match length of dataset, but only the first 3 are used.
    data[0, :] = 0
    data = prep.apply(data)
    results = network(data[1:, 0:3])
    robot = RobotArm()

    data[1:, 3:6] = results
    data = prep.revert(data)

    prediction = data[1:, 3:6]
    angles = np.zeros((nb_pos + 1, 6))
    angles[:, 0:3] = data[:, 0:3]
    ax = None
    for i in range(nb_pos):
        ax = robot.animate(angles[i, :], angles[i + 1, :], ax, prediction[i, :])

def illustrate_results_FM_tensorflow(model, nb_pos=10):

    class Normalisation():

        def __init__(self):
            self.mean = 0
            self.std = 0

        def norm(self, dataset):
            datasetStats = dataset.describe()
            datasetStats = datasetStats.transpose()
            self.mean = datasetStats['mean']
            self.std = datasetStats['std']
            return (dataset - datasetStats['mean']) / datasetStats['std']

        def deNorm(self, dataset):
            return (dataset * self.std) + self.mean

    data = ( (np.random.rand(nb_pos, 6) * 2 - 1) * np.pi / 2 )  # generating 10 rows to match length of dataset, but only the first 3 are used.

    dataset = np.loadtxt("FM_dataset.dat")
    wholeDatasetDataFrame = pd.DataFrame(dataset, columns=['theta1', 'theta2', 'theta3','x','y','z'])
    wholeDatasetDataFrame = shuffle(wholeDatasetDataFrame).head(10)

    datasetDataFrame = pd.DataFrame(data, columns=['theta1', 'theta2', 'theta3','x','y','z'])

    # data[0, :] = 0
    # print(data)
    # data = prep.apply(data)
    # results = network(data[1:, 0:3])
    # robot = RobotArm()

    # data[1:, 3:6] = results
    # data = prep.revert(data)

    # prediction = data[1:, 3:6]
    # angles = np.zeros((nb_pos + 1, 6))
    # angles[:, 0:3] = data[:, 0:3]
    # ax = None
    # for i in range(nb_pos):
    #     ax = robot.animate(angles[i, :], angles[i + 1, :], ax, prediction[i, :])

    
    testDataLabels = pd.concat([datasetDataFrame.pop(x) for x in ['x','y','z']], 1)
    normaliseX = Normalisation()
    normaliseY = Normalisation()
    normData = normaliseX.norm(datasetDataFrame)
    _ = normaliseY.norm(pd.concat([wholeDatasetDataFrame.pop(x) for x in ['x','y','z']], 1))

    predictions = model.predict(normData)
    # print(predictions)
    predictions = pd.DataFrame(predictions, columns=['x','y','z'])
    predictions = normaliseY.deNorm(predictions)
    # print('\n Denormalised Predictions:', predictions)
    robot = RobotArm()

    prediction = predictions.values
    angles = np.zeros((nb_pos + 1, 6))
    angles[1:, 0:3] = datasetDataFrame.values
    ax = None
    for i in range(nb_pos):
        ax = robot.animate(angles[i, :], angles[i + 1, :], ax, prediction[i, :])