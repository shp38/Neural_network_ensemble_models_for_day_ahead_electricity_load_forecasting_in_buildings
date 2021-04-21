import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

class ensemble_model_single:

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
            model_i = MLPRegressor(hidden_layer_sizes=[5,5], solver="sgd", activation="logistic", max_iter=1000, learning_rate='constant', learning_rate_init = 0.001).fit(
                data_label_i.filter(items=["Timestamp", "Temperature", "Weekdays","Isweekend"]), loads_label_i.values)
            models[str(i)] = model_i
        return models

    def __init__(self, n_clusters, data):

        # predict the labels of clusters.
        df = np.array(data)
        dataframe = pd.DataFrame(data, columns=["Timestamp", "Temperature", "Time", "Weekdays", "Isweekend", "Load"])
        dataframe_weekend = dataframe[dataframe["Isweekend"] == 1]
        dataframe = dataframe[dataframe["Isweekend"] == 0]
        cluster_dataframe = pd.DataFrame(data, columns=["Timestamp", "Temperature", "Time", "Weekdays", "Isweekend", "Load"])
        cluster_dataframe = cluster_dataframe[cluster_dataframe["Isweekend"] == 0]


        del cluster_dataframe["Time"]
        del cluster_dataframe["Isweekend"]
        del cluster_dataframe["Weekdays"]

        time_labels, label, cluster_dataframe, dataframe = self.cluster(n_clusters, cluster_dataframe, dataframe)
        # Getting unique labels
        u_labels = np.unique(label)
        u_labels.sort()
        u_time_labels = np.unique(time_labels)

        #u_time_labels.sort()
        time_labels.sort()
        time = dataframe.index / 2
        dataframe['Time'] = time
        models = {}



        # if n_clusters == 48:
        #     #time_labels = range(0,48)
        #     for i in range(0,48):
        #         data_label_i = dataframe.loc[dataframe['Timestamp'] == i]
        #         loads_label_i = data_label_i["Load"]
        #         model_i = MLPRegressor(hidden_layer_sizes=[5], solver="sgd", activation="logistic", max_iter=500,
        #                                learning_rate='constant', learning_rate_init=0.001).fit(
        #             data_label_i.filter(items=["Timestamp", "Temperature", "Weekdays", "Isweekend"]),
        #             loads_label_i.values)
        #         models.append(model_i)
        #     weekend_model = MLPRegressor(hidden_layer_sizes=[5], solver="sgd", activation="logistic", max_iter=500, learning_rate='constant', learning_rate_init = 0.001).fit(
        #             dataframe_weekend.filter(items=["Timestamp", "Temperature", "Weekdays", "Isweekend"]), dataframe_weekend.filter(items=["Load"]))
        #     print(cluster_dataframe[label == 0]["Load"])
        #else:



        for i in u_time_labels:
            data_label_i = dataframe[dataframe['Timelabel'] == i]
            loads_label_i = data_label_i["Load"]
            training_data = data_label_i.drop(["Load", "Label", 'Timelabel'], axis =1).values
            model_i = MLPRegressor(hidden_layer_sizes=[5,5], solver="sgd", activation="logistic", max_iter=1000, learning_rate='constant', learning_rate_init = 0.0005).fit(
               training_data , loads_label_i.values)
            models[str(i)] = model_i
        training_data_weekend = dataframe_weekend.drop(["Load"], axis = 1).values
        weekend_model = MLPRegressor(hidden_layer_sizes=[5,5], solver="sgd", activation="logistic", max_iter=1000, learning_rate='constant', learning_rate_init = 0.0005).fit(
            training_data_weekend , dataframe_weekend.filter(items=["Load"]))


        # for i in u_weekend_time_labels:
        #     weekend_data_label_i = dataframe_weekend[dataframe_weekend['Timelabel'] == i]
        #     weekend_loads_label_i = weekend_data_label_i["Load"]
        #     weekend_model_i = MLPRegressor(hidden_layer_sizes=[5,5], solver="sgd", activation="logistic", max_iter=1000, learning_rate='constant', learning_rate_init = 0.001).fit(
        #         weekend_data_label_i.filter(items=["Timestamp", "Temperature", "Weekdays", "Isweekend"]), weekend_loads_label_i.values)
        #     weekend_models[str(i)] = weekend_model_i
        # for i in u_time_labels:
        #     plt.scatter(cluster_dataframe[label == i]["Timestamp"], cluster_dataframe[label == i]["Load"], label=i)
        # #plt.plot(dataframe['Timestamp'], dataframe['Temperature'], label="Temperature")
        # plt.legend()
        # plt.show()

        self.model = models
        self.n_clusters = n_clusters
        self.cluster_labels = u_labels
        self.time_labels = time_labels
        self.weekend_model = weekend_model


    def predict(self, test_data):
        prediction = []
        test_data = np.array(test_data)
        test_data = pd.DataFrame(test_data, columns=["Timestamp", "Temperature","Time", "Weekdays", "Isweekend", "Load"])

        for i in range(len(test_data)):
            if test_data["Isweekend"][i] != 1:
                label_i = self.time_labels[np.remainder(i,48)]
                input_data = test_data.drop(["Load"], axis =1).values
                prediction_i = self.model[str(label_i)].predict(input_data[i].reshape(1, -1))
                prediction.extend(prediction_i)
            else:
                input_data = np.array(test_data.drop('Load', axis=1))  # input_data.reshape(1, -1)
                prediction_i = self.weekend_model.predict(input_data[i].reshape(1, -1))
                prediction.extend(prediction_i)

        prediction = np.array(prediction)
        # plt.plot(test_data['Time'], test_data['Load'], label="Actual")
        # plt.plot(test_data['Time'], prediction, label="Predicted")
        # #plt.plot(test_data[test_data["Isweekend"] == 1]['Time'], prediction_weekend, label="Predicted Weekend")
        # plt.legend()
        # plt.show()

        MAPE = mean_absolute_percentage_error(test_data['Load'], prediction)
        print(MAPE)

        return prediction, MAPE