import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

class ensemble_model_historical_multi:

    def cluster(self,n_clusters, cluster_dataframe, dataframe):
        # Initialize the class object
        means = []
        for i in range(48):
            data_at_time_i = dataframe.loc[dataframe['Timestamp'] == i]
            means.append([i, np.median(data_at_time_i['Temperature']), np.median(data_at_time_i['Load'])])

        #Initialise clusters
        initialisation = []
        for i in range(0, 48, int(48/n_clusters)):
            initialisation.append(means[i])
        end = True
        while len(initialisation) > n_clusters:
            if end:
                del initialisation[-1]
                end = False
            else:
                del initialisation[0]
                end = True
        initialisation = np.array(initialisation)
        kmeans = KMeans(n_clusters=n_clusters, init=initialisation)
        label = kmeans.fit_predict(cluster_dataframe)
        cluster_dataframe["Label"] = label
        dataframe["Label"] = label

        time_labels = kmeans.predict(means)
        data_time_labels = []
        for row in dataframe['Timestamp']:
            data_time_labels.append(time_labels[int(row)])
        dataframe["Timelabel"] = data_time_labels
        return time_labels, label, cluster_dataframe, dataframe


    def train_models(self, dataframe, u_time_labels):
        models = {}
        for i in u_time_labels:
            data_label_i = dataframe[dataframe['Timelabel'] == i]
            loads_label_i = data_label_i["Load"]
            model_i = MLPRegressor(hidden_layer_sizes=[5], solver="sgd", activation="tanh", max_iter=1000, learning_rate='constant', learning_rate_init = 0.0005).fit(
                data_label_i.drop(["Load", "Label", "Timelabel"], axis=1), loads_label_i.values)
            models[str(i)] = model_i
        return models


    def __init__(self, n_clusters, data, historical_data):

        # predict the labels of clusters.
        df = np.array(data)
        dataframe = pd.DataFrame(data, columns=["Timestamp", "Temperature", "Time", "Weekdays", "Isweekend", "Load"])
        for i in range(len(historical_data)):
            column_name = "prev_" + str(i+1) + "_day_data"
            dataframe[column_name] = historical_data[i]
        dataframe_weekend = dataframe[dataframe["Isweekend"] == 1]
        dataframe = dataframe[dataframe["Isweekend"] == 0]
        cluster_dataframe = pd.DataFrame(data, columns=["Timestamp", "Temperature", "Time", "Weekdays", "Isweekend", "Load"])
        weekend_cluster_dataframe = cluster_dataframe[cluster_dataframe["Isweekend"] == 1]
        cluster_dataframe = cluster_dataframe[cluster_dataframe["Isweekend"] == 0]

        del weekend_cluster_dataframe["Time"]
        del weekend_cluster_dataframe["Isweekend"]
        del weekend_cluster_dataframe["Weekdays"]
        del cluster_dataframe["Time"]
        del cluster_dataframe["Isweekend"]
        del cluster_dataframe["Weekdays"]

        time_labels, label, cluster_dataframe, dataframe = self.cluster(n_clusters, cluster_dataframe, dataframe)
        weekend_time_labels, weekend_label, weekend_cluster_dataframe, dataframe_weekend = self.cluster(n_clusters, weekend_cluster_dataframe, dataframe_weekend)

        # Getting unique labels
        u_labels = np.unique(label)
        u_labels.sort()
        u_time_labels = np.unique(time_labels)
        u_weekend_labels = np.unique(weekend_label)
        u_weekend_labels.sort()
        u_weekend_time_labels = np.unique(weekend_time_labels)
        #u_time_labels.sort()
        time_labels.sort()
        weekend_time_labels.sort()
        time = dataframe.index / 2
        dataframe['Time'] = time


        models = self.train_models(dataframe, u_time_labels)
        weekend_models = self.train_models(dataframe_weekend, u_weekend_time_labels)



        self.model = models
        self.n_clusters = n_clusters
        self.cluster_labels = u_labels
        self.time_labels = time_labels
        self.weekend_time_labels = weekend_time_labels
        self.weekend_models = weekend_models


    def predict(self, test_data, historical_test_data):
        prediction = []
        test_data = np.array(test_data)
        test_data = pd.DataFrame(test_data, columns=["Timestamp", "Temperature","Time", "Weekdays", "Isweekend", "Load"])
        for i in range(len(historical_test_data)):
            column_name = "prev_" + str(i+1) + "_day_data"
            test_data[column_name] = historical_test_data[i]
        for i in range(len(test_data)):
            if test_data["Isweekend"][i] != 1:
                label_i = self.time_labels[np.remainder(i,48)]
                input_data = np.array(test_data.drop('Load', axis=1))  #input_data.reshape(1, -1)
                prediction_i = self.model[str(label_i)].predict(input_data[i].reshape(1,-1))
                prediction.extend(prediction_i)
            else:
                label_i = self.weekend_time_labels[np.remainder(i, 48)]
                input_data = np.array(test_data.drop('Load', axis=1)) #input_data.reshape(1, -1)
                prediction_i = self.weekend_models[str(label_i)].predict(input_data[i].reshape(1,-1))
                prediction.extend(prediction_i)

        prediction = np.array(prediction)


        MAPE = mean_absolute_percentage_error(test_data['Load'], prediction)
        #print(MAPE)

        return prediction, MAPE