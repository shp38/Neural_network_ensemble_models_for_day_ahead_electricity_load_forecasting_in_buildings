# This is a sample Python script.

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from ensemble_model_single import ensemble_model_single
from ensemble_model_historical_single import ensemble_model_historical_single
from ensemble_model_multi import ensemble_model_multi
from ensemble_model_historical_multi import ensemble_model_historical_multi
from sklearn.metrics import mean_absolute_percentage_error


from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from pandas.tseries.holiday import (
    AbstractHolidayCalendar, DateOffset, EasterMonday,
    GoodFriday, Holiday, MO,
    next_monday, next_monday_or_tuesday)

class EnglandAndWalesHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('New Years Day', month=1, day=1, observance=next_monday),
        GoodFriday,
        EasterMonday,
        Holiday('Early May bank holiday',
                month=5, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday('Spring bank holiday',
                month=5, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday('Summer bank holiday',
                month=8, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday('Christmas Day', month=12, day=25, observance=next_monday),
        Holiday('Boxing Day',
                month=12, day=26, observance=next_monday_or_tuesday)
    ]


def cluster_parametrisation_single(data, validation_data):
    MAPE_optimal = 8
    MAPEs = []
    for i in range(1,49, 3):
        model_i = ensemble_model_single(i, data)
        prediction, MAPE = model_i.predict(validation_data)
        if MAPE <= MAPE_optimal:
            best_model = model_i
            n_clusters_optimal = i
            MAPE_optimal = MAPE
        MAPEs.append(MAPE)


    return best_model, n_clusters_optimal, MAPEs


def cluster_parametrisation_multi(data, validation_data):
    MAPE_optimal = 8
    MAPEs = []
    for i in range(1,49, 3):
        model_i = ensemble_model_multi(i, data)
        prediction, MAPE = model_i.predict(validation_data)
        if MAPE <= MAPE_optimal:
            best_model = model_i
            n_clusters_optimal = i
            MAPE_optimal = MAPE
        MAPEs.append(MAPE)


    return best_model, n_clusters_optimal, MAPEs


def historical_cluster_parametrisation_single(data, validation_data, historical_data, historical_validation_data):
    MAPE_optimal = 8
    MAPEs = []
    for i in range(1,49, 3):
        model_i = ensemble_model_historical_single(i, data, historical_data)
        prediction, MAPE = model_i.predict(validation_data, historical_validation_data)
        if MAPE <= MAPE_optimal:
            best_model = model_i
            n_clusters_optimal = i
            MAPE_optimal = MAPE
        MAPEs.append(MAPE)


    return best_model, n_clusters_optimal, MAPEs

def historical_cluster_parametrisation_multi(data, validation_data, historical_data, historical_validation_data):
    MAPE_optimal = 8
    MAPEs = []
    for i in range(1,49, 3):
        model_i = ensemble_model_historical_multi(i, data, historical_data)
        prediction, MAPE = model_i.predict(validation_data, historical_validation_data)
        if MAPE <= MAPE_optimal:
            best_model = model_i
            n_clusters_optimal = i
            MAPE_optimal = MAPE
        MAPEs.append(MAPE)


    return best_model, n_clusters_optimal, MAPEs

def preprocess_data(load_data_folder, weather_data_file, meter):


    weather_data = pd.read_csv(weather_data_file)
    weather_data = weather_data[1342:4918]   #Extract the range of dates that interest us
    temp_data = weather_data['tempC'].values
    temp_interpolated = []
    for i in range(len(temp_data)):
        temp_interpolated.append(temp_data[i])
        if i < len(temp_data)-1:
            temp_interpolated.append((temp_data[i] + temp_data[i+1])/2)
        else:
            temp_interpolated.append(temp_data[i])

    load_data = []
    for filename in os.listdir(load_data_folder + meter):
        file_path = load_data_folder + meter + "/" + filename
        load_data.append(pd.read_csv(file_path))

    previous_load_data_folder = "C:/Users/sebpo/OneDrive/Desktop/University Work/FYP/load_data_directory/OSCE EMLite Single/7_days_previous_data"


    previous_load_data = []
    for filename in os.listdir(previous_load_data_folder + "/" + meter):
        file_path = previous_load_data_folder + "/" + meter + "/" + filename
        previous_load_data.append(pd.read_csv(file_path))
    previous_load_data = pd.concat(previous_load_data, ignore_index=True, sort=False)

    previous_load_data = previous_load_data.iloc[:, 1]
    previous_load_data = previous_load_data.diff()
    previous_load_data = previous_load_data.interpolate(method='pad')
    previous_load_data[0] = previous_load_data[1]
    previous_load_data_1_day = previous_load_data[288:].values
    previous_load_data_2_days = previous_load_data[240:-48].values
    previous_load_data_3_days = previous_load_data[192:-96].values
    previous_load_data_4_days = previous_load_data[144:-144].values
    previous_load_data_5_days = previous_load_data[96:-192].values
    previous_load_data_6_days = previous_load_data[48:-240].values
    previous_load_data_7_days = previous_load_data[:-288].values



    # timeseries_list = []
    #
    # for j in range(len(load_data)):
    #     timeseries_i = []
    #     timeseries = load_data[j].iloc[:, 1].tolist()
    #     if j != 0:
    #         load_previous = load_data[j - 1].iloc[-1, 1].tolist()
    #     for i in range(len(timeseries)):
    #         if i == 0:
    #             if j == 0:
    #                 load = timeseries[i + 1] - timeseries[i]
    #             else:
    #                 load = timeseries[i] - load_previous
    #         else:
    #             load = timeseries[i] - timeseries[i - 1]
    #         timeseries_i.append([i, load])
    #     timeseries_list.append(timeseries_i)
    # timeseries_list = np.array(timeseries_list)
    #
    # X_train = timeseries_list
    # # X_train = TimeSeriesScalerMeanVariance().fit_transform(timeseries_list)
    # sz = X_train.shape[1]
    # # Soft-DTW-k-means
    # print("Soft-DTW k-means")
    # sdtw_km = TimeSeriesKMeans(n_clusters=3,
    #                            metric="softdtw",
    #                            metric_params={"gamma": .01}
    #                            )
    # y_pred = sdtw_km.fit_predict(X_train)
    #
    # for yi in range(3):
    #     plt.subplot(3, 3, 7 + yi)
    #     for xx in X_train[y_pred == yi]:
    #         plt.plot(xx.ravel(), "k-", alpha=.2)
    #     plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
    #     plt.xlim(0, sz)
    #     plt.ylim(0, 5000)
    #     plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
    #              transform=plt.gca().transAxes)
    #     if yi == 1:
    #         plt.title("Soft-DTW $k$-means")
    #
    # plt.tight_layout()
    # plt.show()
    load_data = pd.concat(load_data, ignore_index=True, sort=False)
    load_data['created_at'] = pd.to_datetime(load_data['created_at'], format="%Y-%m-%d %H:%M:%S%z")
    weekdays = load_data['created_at'].dt.dayofweek
    dates = load_data['created_at']
    load_data = load_data.iloc[:, 1]
    load_data = load_data.diff()
    load_data = load_data.interpolate(method='pad')
    load_data[0] = load_data[1]
    # print(annex_load_data)
    timestamps = np.remainder(load_data.index, 48)
    time = load_data.index / 2

    # day_of_the_week = np.remainder(annex_load_data.index/48, 7)
    cal = EnglandAndWalesHolidayCalendar()
    dr = pd.date_range(start='2020-10-15', end='2021-03-12')
    winter_holidays = pd.date_range(start='2020-12-25', end='2021-01-01')

    holidays = cal.holidays(start=dr.min(), end=dr.max())
    isweekend = []
    for i in range(len(weekdays)):
        if weekdays[i] == 5 or weekdays[i] == 6 or dates[i] in holidays or dates[i] in winter_holidays:
            isweekend.append(1)
        else:
            isweekend.append(0)

    # print("Weekend?: \n",isweekend)
    # transformer = Normalizer().fit(np.array([timestamps, temp_interpolated]))
    # temp_interpolated = transformer.transform(np.array([timestamps, temp_interpolated]))
    # temp_interpolated = temp_interpolated[1]
    # load_data['Timestamp'] = timestamps
    # print(annex_load_data)

    data = np.column_stack([timestamps, temp_interpolated, time, weekdays, isweekend, load_data.values])
    test_data = data[-1008:-672]
    data = data[:-1008]
    previous_load_test_data_1_day = previous_load_data_1_day[-1008:-672]
    previous_load_test_data_2_days = previous_load_data_2_days[-1008:-672]
    previous_load_test_data_3_days = previous_load_data_3_days[-1008:-672]
    previous_load_test_data_4_days = previous_load_data_4_days[-1008:-672]
    previous_load_test_data_5_days = previous_load_data_5_days[-1008:-672]
    previous_load_test_data_6_days = previous_load_data_6_days[-1008:-672]
    previous_load_test_data_7_days = previous_load_data_7_days[-1008:-672]
    previous_load_data_1_day = previous_load_data_1_day[:-1008]
    previous_load_data_2_days = previous_load_data_2_days[:-1008]
    previous_load_data_3_days = previous_load_data_3_days[:-1008]
    previous_load_data_4_days = previous_load_data_4_days[:-1008]
    previous_load_data_5_days = previous_load_data_5_days[:-1008]
    previous_load_data_6_days = previous_load_data_6_days[:-1008]
    previous_load_data_7_days = previous_load_data_7_days[:-1008]
    plt.plot(time[:-672], load_data[:-672])
    plt.show()
    previous_load_test_data = [previous_load_test_data_1_day,previous_load_test_data_2_days,previous_load_test_data_3_days,previous_load_test_data_4_days,previous_load_test_data_5_days,previous_load_test_data_6_days,previous_load_test_data_7_days]
    previous_load_data = [previous_load_data_1_day, previous_load_data_2_days, previous_load_data_3_days, previous_load_data_4_days,previous_load_data_5_days,previous_load_data_6_days,previous_load_data_7_days]

    return data, test_data, load_data, previous_load_data, previous_load_test_data

if __name__ == '__main__':
    weather_data_file = "C:/Users/sebpo/OneDrive/Desktop/University Work/FYP/Historical Weather Data/weather_data.csv"
    load_data_folder = "C:/Users/sebpo/OneDrive/Desktop/University Work/FYP/load_data_directory/OSCE EMLite Single/"
    meters = [ "OSCE ECC Cafe", "OSCE Annex Pwr","OSCE ECC EHeat"]
    agregate_load_data = np.zeros(7152,)
    agregate_load_data = pd.Series(agregate_load_data)

    prediction_sum = pd.Series(np.zeros(336,))
    prediction_sum_hist = pd.Series(np.zeros(336,))

    for meter in meters:

        # #print(data)
        #

        data, test_data, load_data_meter_i, previous_days_load_data, previous_days_load_test_data = preprocess_data(load_data_folder, weather_data_file, meter)
        agregate_load_data = agregate_load_data + load_data_meter_i
        hist_forecaster, n_clusters_optimal_hist, MAPEs_hist = historical_cluster_parametrisation_multi(data, test_data, previous_days_load_data, previous_days_load_test_data)
        forecaster, n_clusters_optimal, MAPEs = cluster_parametrisation_single(data, test_data)
        n_clusters = range(1,49,3)
        plt.plot(n_clusters, MAPEs)
        plt.show()
        plt.plot(n_clusters, MAPEs_hist)
        plt.show()
        #test_data = pd.DataFrame(test_data, columns=["Timestamp", "Temperature","Time", "Weekdays", "Isweekend", "Load"])
        print(n_clusters_optimal)
        #prediction_meter_i, MAPE = forecaster.predict(test_data, previous_days_load_test_data)
        prediction_meter_i, MAPE = forecaster.predict(test_data)
        prediction_meter_i = pd.Series(prediction_meter_i)
        prediction_sum += prediction_meter_i

        prediction_meter_i_hist, MAPE_hist = forecaster.predict(test_data)
        prediction_meter_i_hist = pd.Series(prediction_meter_i_hist)
        prediction_sum_hist += prediction_meter_i_hist


    agregate_load_test_data = agregate_load_data[-1008:-672]
    agregate_load_data = agregate_load_data[:-1008]
    data[:,-1] = agregate_load_data.values
    test_data[:,-1] = agregate_load_test_data.values
    forecaster, n_clusters_optimal, MAPEs = cluster_parametrisation_single(data, test_data)
    forecaster_hist, n_clusters_optimal_hist, MAPEs_hist = historical_cluster_parametrisation_multi(data, test_data, previous_days_load_data, previous_days_load_test_data)
    n_clusters = range(1, 49, 3)
    plt.plot(n_clusters, MAPEs)
    plt.show()
    plt.plot(n_clusters, MAPEs_hist)
    plt.show()
    test_data = pd.DataFrame(test_data, columns=["Timestamp", "Temperature", "Time", "Weekdays", "Isweekend", "Load"])
    print(n_clusters_optimal)
    prediction, MAPE = forecaster.predict(test_data)
    print("Agregate MAPE: ",MAPE)
    plt.plot(test_data['Time'], test_data['Load'], label="Actual")
    plt.plot(test_data['Time'], prediction, label="Predicted")
    plt.legend()
    plt.title("Agregate Prediction(Multiple clusters at weekends)")
    plt.show()

    print(n_clusters_optimal_hist)
    prediction, MAPE = forecaster_hist.predict(test_data,previous_days_load_test_data)
    print("Agregate MAPE: ", MAPE)
    plt.plot(test_data['Time'], test_data['Load'], label="Actual")
    plt.plot(test_data['Time'], prediction, label="Predicted")
    plt.legend()
    plt.title("Agregate Prediction(Multiple clusters at weekends)")
    plt.show()

    # plt.plot(test_data['Time'], test_data['Load'], label="Actual")
    # plt.plot(test_data['Time'], prediction_sum, label="Predicted")
    # MAPE = mean_absolute_percentage_error(test_data['Load'], prediction_sum)
    # print("Sum MAPE: ", MAPE)
    # plt.legend()
    # plt.title("Sum of individual Meter Predictions")
    # plt.show()


    # prediction, MAPE = hist_forecaster.predict(test_data, previous_days_load_test_data)
    # print("Agregate MAPE(with historical data input): ", MAPE)
    # plt.plot(test_data['Time'], test_data['Load'], label="Actual")
    # plt.plot(test_data['Time'], prediction, label="Predicted")
    # plt.legend()
    # plt.title("Agregate Prediction(with historical data input)")
    # plt.show()
    plt.plot(test_data['Time'], test_data['Load'], label="Actual")
    plt.plot(test_data['Time'], prediction_sum, label="Predicted")
    MAPE = mean_absolute_percentage_error(test_data['Load'], prediction_sum)
    print("Sum MAPE(with single cluster at weekends): ", MAPE)
    plt.legend()
    plt.title("Sum of individual Meter Predictions(with single cluster at weekends)")
    plt.show()

    plt.plot(test_data['Time'], test_data['Load'], label="Actual")
    plt.plot(test_data['Time'], prediction_sum_hist, label="Predicted")
    MAPE = mean_absolute_percentage_error(test_data['Load'], prediction_sum_hist)
    print("Sum MAPE(with multiple clusters at weekends): ", MAPE)
    plt.legend()
    plt.title("Sum of individual Meter Predictions(with multiple clusters at weekends)")
    plt.show()




    # plt.plot(test_data[test_data["Isweekend"] == 1]['Time'], prediction_weekend, label="Predicted Weekend")



    # for i in range(48):
    #     #low = int(8*abs(np.sin(0.75*i)))
    #     high = int(8 + 8*abs(np.sin(0.1*i)))
    #     for j in range(0,140):
    #         #y = np.random.randint(low=low, high=high)
    #         y = np.random.normal() + high
    #         temp = high - 5
    #         datapoint = [i, temp, y]
    #         data.append(datapoint)
    #
    #     y = np.random.normal() + high
    #     temp = high - 5
    #     datapoint = [i, temp, y]
    #     test_data.append(datapoint)

    #forecaster = ensemble_model_temp(5,data)
    #prediction, MAPE = forecaster.predict(test_data)

    # # Initialize the class object
    # kmeans = KMeans(n_clusters=3)
    # data.sort()
    # # predict the labels of clusters.
    # df = np.array(data)
    # test_data = np.array(test_data)
    # test_dataframe = pd.DataFrame(test_data, columns=["Timestamp", "Temperature", "Load"])
    # label = kmeans.fit_predict(df)
    # dataframe = pd.DataFrame(data, columns = ["Timestamp", "Temperature", "Load"])
    # dataframe["Label"] = label
    # #print(dataframe)
    # means = []
    #
    # for i in range(48):
    #     data_at_time_i = []
    #     for j in range(0,29):
    #         data_at_time_i.append([data[i*140+j][-2], data[i*140+j][-1]])
    #     #print(data[i+j][-1])
    #     means.append([i, np.median(data_at_time_i[-2]), np.median(data_at_time_i[-1])])
    # #print(means)
    # timelabels = kmeans.predict(means)
    # #print(timelabels)
    #
    #
    #
    # # Getting unique labels
    # u_labels = np.unique(label)
    # u_labels.sort()
    # models = []
    # data_by_label = []
    # #print(u_labels, type(u_labels))
    # for i in u_labels:
    #     data_label_i = dataframe.loc[dataframe['Label'] == i]
    #     #time_stamps_label_i = [data_label_i.filter(items = "Timestamp").values, data_label_i.filter(items ="Temperature").values]
    #     loads_label_i = data_label_i["Load"]
    #     #print(time_stamps_label_i)
    #     #print(loads_label_i)
    #     model_i = MLPRegressor(hidden_layer_sizes=[10], solver="sgd", activation="logistic", max_iter=500).fit(data_label_i.filter(items = ["Timestamp", "Temperature"]), loads_label_i.values)
    #     models.append(model_i)
    #
    # prediction = []
    # for i in range(48):
    #     label_i = timelabels[i]
    #     print(label_i)
    #     input_data = np.array([test_dataframe['Timestamp'][i], test_dataframe['Temperature'][i]])
    #     prediction_i = models[label_i].predict(input_data.reshape(1, -1))
    #     prediction.extend(prediction_i)
    #
    # prediction = np.array(prediction)
    #print(prediction, np.size(prediction))
    # plotting the results:

    #print(test_dataframe['Timestamp'], np.size(test_dataframe['Timestamp']))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
