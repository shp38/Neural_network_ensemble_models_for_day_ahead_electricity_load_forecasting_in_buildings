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
import warnings


from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from pandas.tseries.holiday import (
    AbstractHolidayCalendar, DateOffset, EasterMonday,
    GoodFriday, Holiday, MO,
    next_monday, next_monday_or_tuesday)
# define bank holidays
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

#function for finding optimal number of clusters using brute force method.
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

#function for finding optimal number of clusters using brute force method.
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

#function for finding optimal number of clusters using brute force method.
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

#function for finding optimal number of clusters using brute force method.
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

#preprocess_data: convenience function for reading data from a meter folder into a dataframe
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

    #time series clustering uncomment if timeseries clustering is of interest

    # ///////////////////////// start of timeseries clustering ///////////////////////////
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
    # sdtw_km = TimeSeriesKMeans(n_clusters=2,
    #                            metric="softdtw",
    #                            metric_params={"gamma": .01}
    #                            )
    # y_pred = sdtw_km.fit_predict(X_train)
    #
    # for yi in range(2):
    #     plt.subplot(1, 2, 1+yi)
    #     for xx in X_train[y_pred == yi]:
    #         plt.plot(xx.ravel(), "k-", alpha=.2)
    #     plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
    #     plt.xlim(0, sz)
    #     plt.ylim(0, 5000)
    #     plt.xlabel("Time of day (half hourly increments)")
    #     plt.ylabel("Load, kWh")
    #     plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
    #              transform=plt.gca().transAxes)
    #
    # plt.suptitle("Soft-DTW $k$-means load timeseries clustering")
    #
    # plt.tight_layout()
    # plt.show()
    # //////////////////////// end of timeseries clustering /////////////////////////////////

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

    data = np.column_stack([timestamps, temp_interpolated, time, weekdays, isweekend, load_data.values/75])
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
    title = meter + " Load Data"
    path = "Saved_figures/meter_load_data/" + title.replace(" ", "_")
    plt.ylabel("Load (kWh")
    plt.xlabel("Time (hours)")
    plt.title(title)
    plt.savefig(path)
    plt.clf()
    previous_load_test_data = [previous_load_test_data_1_day,previous_load_test_data_2_days,previous_load_test_data_3_days,previous_load_test_data_4_days,previous_load_test_data_5_days,previous_load_test_data_6_days,previous_load_test_data_7_days]
    previous_load_data = [previous_load_data_1_day, previous_load_data_2_days, previous_load_data_3_days, previous_load_data_4_days,previous_load_data_5_days,previous_load_data_6_days,previous_load_data_7_days]

    return data, test_data, load_data, previous_load_data, previous_load_test_data


# preprocess_ecc_main_data: separate function for parsing ecc main meter data due to different format used
def preprocess_ecc_main_data():
    file = "C:/Users/sebpo/OneDrive/Desktop/University Work/FYP/load_data_directory/OSCE EMLite Single/OSCE ECC Main/ecc-main-forseb.csv"
    weather_data_file = "C:/Users/sebpo/OneDrive/Desktop/University Work/FYP/Historical Weather Data/weather_data.csv"

    weather_data = pd.read_csv(weather_data_file)
    weather_data = weather_data[1342:4918]  # Extract the range of dates that interest us
    temp_data = weather_data['tempC'].values
    temp_interpolated = []
    for i in range(len(temp_data)):
        temp_interpolated.append(temp_data[i])
        if i < len(temp_data) - 1:
            temp_interpolated.append((temp_data[i] + temp_data[i + 1]) / 2)
        else:
            temp_interpolated.append(temp_data[i])

    load_data = pd.read_csv(file)
    load_data = load_data.dropna()

    previous_load_data_1_day = load_data['IMPORT'][288:-48].values
    previous_load_data_2_days = load_data['IMPORT'][240:-96].values
    previous_load_data_3_days = load_data['IMPORT'][192:-144].values
    previous_load_data_4_days = load_data['IMPORT'][144:-192].values
    previous_load_data_5_days = load_data['IMPORT'][96:-240].values
    previous_load_data_6_days = load_data['IMPORT'][48:-288].values
    previous_load_data_7_days = load_data['IMPORT'][:-336].values
    load_data = load_data[336:]
    load_data = load_data.reset_index()

    load_data['END'] = pd.to_datetime(load_data['END'], format="%d/%m/%Y %H:%M")
    weekdays = load_data['END'].dt.dayofweek
    weekdays = weekdays.values
    dates = load_data['END'].values
    timestamps = np.remainder(load_data.index, 48)
    time = ((load_data.index) / 2)
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

    data = np.column_stack([timestamps, temp_interpolated, time, weekdays, isweekend, load_data['IMPORT'].values/0.075])
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

    plt.plot(time[:-672], load_data['IMPORT'][:-672].values/0.001)
    title = meter + " Load Data"
    path = "Saved_figures/meter_load_data/" + title.replace(" ", "_")
    plt.ylabel("Load (kWh)")
    plt.xlabel("Time (hours)")
    plt.title(title)
    plt.savefig(path)
    plt.clf()
    previous_load_test_data = [previous_load_test_data_1_day, previous_load_test_data_2_days,
                               previous_load_test_data_3_days, previous_load_test_data_4_days,
                               previous_load_test_data_5_days, previous_load_test_data_6_days,
                               previous_load_test_data_7_days]
    previous_load_data = [previous_load_data_1_day, previous_load_data_2_days, previous_load_data_3_days,
                          previous_load_data_4_days, previous_load_data_5_days, previous_load_data_6_days,
                          previous_load_data_7_days]

    return data, test_data, load_data['IMPORT']/0.001, previous_load_data, previous_load_test_data   #multiply by 1000 to match units of other meters


if __name__ == '__main__':
    # ignore warnings in order to clean up terminal output.  Disable this line when debugging
    warnings.filterwarnings("ignore")
    weather_data_file = "C:/Users/sebpo/OneDrive/Desktop/University Work/FYP/Historical Weather Data/weather_data.csv"
    load_data_folder = "C:/Users/sebpo/OneDrive/Desktop/University Work/FYP/load_data_directory/OSCE EMLite Single/"
    meters = [  "OSCE ECC Main", "OSCE ECC Cafe", "OSCE Annex Pwr", "OSCE ECC EHeat"] #
    #initialise array varaiables to the correct shape
    agregate_load_data = np.zeros(7152,)
    agregate_load_data = pd.Series(agregate_load_data)
    agregate_load_historical_data = np.zeros([6144, 7])
    agregate_load_historical_test_data = np.zeros([336,7])

    prediction_sum = pd.Series(np.zeros(336,))
    prediction_sum_multi = pd.Series(np.zeros(336, ))
    prediction_sum_hist = pd.Series(np.zeros(336, ))
    prediction_sum_hist_multi = pd.Series(np.zeros(336, ))

    for meter in meters:

        # #print(data)
        #
        if meter != "OSCE ECC Main":
            data, test_data, load_data_meter_i, previous_days_load_data, previous_days_load_test_data = preprocess_data(load_data_folder, weather_data_file, meter)
        else:
            data, test_data, load_data_meter_i, previous_days_load_data, previous_days_load_test_data = preprocess_ecc_main_data()


        agregate_load_data = agregate_load_data + load_data_meter_i
        agregate_load_historical_data_meter_i = np.array(previous_days_load_data).reshape(6144,7)
        agregate_load_historical_data = np.add(agregate_load_historical_data_meter_i, agregate_load_historical_data)
        agregate_load_historical_test_data_meter_i = np.array(previous_days_load_test_data).reshape(336,7)
        agregate_load_historical_test_data = np.add(agregate_load_historical_test_data_meter_i, agregate_load_historical_test_data)

        hist_forecaster, n_clusters_optimal_hist_single, MAPEs_hist_single = historical_cluster_parametrisation_single(data, test_data, previous_days_load_data, previous_days_load_test_data)
        hist_forecaster_multi, n_clusters_optimal_hist_multi, MAPEs_hist_multi = historical_cluster_parametrisation_multi(data, test_data, previous_days_load_data, previous_days_load_test_data)
        forecaster, n_clusters_optimal, MAPEs = cluster_parametrisation_single(data, test_data)
        forecaster_multi, n_clusters_optimal_multi, MAPEs_multi = cluster_parametrisation_multi(data, test_data)

        n_clusters = range(1,49,3)

        #Plot MAPE-cluster curves for each model, and save the plots in a folder

        print(meter + " Model 1 optimal number of clusters: ",n_clusters_optimal)
        print(meter + " Model 1 best MAPE: ",min(MAPEs))
        plt.plot(n_clusters, MAPEs)
        title = meter + " Model 1 MAPE"
        path = "Saved_figures/meter_MAPEs/" + title.replace(" ", "_")
        plt.ylabel("MAPE")
        plt.xlabel("Number of Clusters")
        plt.title(title)
        plt.savefig(path)
        plt.clf()

        print(meter + " Model 2 optimal number of clusters: ", n_clusters_optimal_multi)
        print(meter + " Model 2 best MAPE: ", min(MAPEs_multi))
        plt.plot(n_clusters, MAPEs_multi)
        title = meter + " Model 2 MAPE"
        path = "Saved_figures/meter_MAPEs/" + title.replace(" ", "_")
        plt.ylabel("MAPE")
        plt.xlabel("Number of Clusters")
        plt.title(title)
        plt.savefig(path)
        plt.clf()

        print(meter + " Model 3 optimal number of clusters: ", n_clusters_optimal_hist_single)
        print(meter + " Model 3 best MAPE:", min(MAPEs_hist_single))
        plt.plot(n_clusters, MAPEs_hist_single)
        title = meter + " Model 3 MAPE"
        path = "Saved_figures/meter_MAPEs/" + title.replace(" ", "_")
        plt.ylabel("MAPE")
        plt.xlabel("Number of Clusters")
        plt.title(title)
        plt.savefig(path)
        plt.clf()

        print(meter + " Model 4 optimal number of clusters: ", n_clusters_optimal_hist_multi)
        print(meter + " Model 4 best MAPE:", min(MAPEs_hist_multi))
        plt.plot(n_clusters, MAPEs_hist_multi)
        title = meter + " Model 4 MAPE"
        path = "Saved_figures/meter_MAPEs/" + title.replace(" ", "_")
        plt.ylabel("MAPE")
        plt.xlabel("Number of Clusters")
        plt.title(title)
        plt.savefig(path)
        plt.clf()
        #test_data = pd.DataFrame(test_data, columns=["Timestamp", "Temperature","Time", "Weekdays", "Isweekend", "Load"])
        #print(n_clusters_optimal)
        #prediction_meter_i, MAPE = forecaster.predict(test_data, previous_days_load_test_data)


        prediction_meter_i, MAPE = forecaster.predict(test_data)
        prediction_meter_i = pd.Series(prediction_meter_i)
        prediction_sum += prediction_meter_i

        prediction_meter_i_multi, MAPE_multi = forecaster_multi.predict(test_data)
        prediction_meter_i_multi = pd.Series(prediction_meter_i_multi)
        prediction_sum_multi += prediction_meter_i_multi

        prediction_meter_i_hist, MAPE_hist = hist_forecaster.predict(test_data, previous_days_load_test_data)
        prediction_meter_i_hist = pd.Series(prediction_meter_i_hist)
        prediction_sum_hist += prediction_meter_i_hist

        prediction_meter_i_hist_multi, MAPE_hist_multi = hist_forecaster_multi.predict(test_data, previous_days_load_test_data)
        prediction_meter_i_hist_multi = pd.Series(prediction_meter_i_hist_multi)
        prediction_sum_hist_multi += prediction_meter_i_hist_multi


    agregate_load_test_data = agregate_load_data[-1008:-672]/75
    agregate_load_data = agregate_load_data[:-1008]/75
    previous_days_load_test_data = np.array(agregate_load_historical_test_data).reshape(7,336)
    previous_days_load_data = np.array(agregate_load_historical_data).reshape(7, 6144)
    data[:,-1] = agregate_load_data
    test_data[:,-1] = agregate_load_test_data

    forecaster, n_clusters_optimal, MAPEs = cluster_parametrisation_single(data, test_data)
    forecaster_multi, n_clusters_optimal_multi, MAPEs_multi = cluster_parametrisation_multi(data, test_data)
    forecaster_hist, n_clusters_optimal_hist, MAPEs_hist = historical_cluster_parametrisation_single(data, test_data, previous_days_load_data, previous_days_load_test_data)
    forecaster_hist_multi, n_clusters_optimal_hist_multi, MAPEs_hist_multi = historical_cluster_parametrisation_multi(data, test_data, previous_days_load_data, previous_days_load_test_data)

    # //////////////Plotting + plot saving into the correct folders/////////////////////

    n_clusters = range(1, 49, 3)

    plt.plot(n_clusters, MAPEs)
    title = "ECC and Nursery MAPE Model 1"
    path = "Saved_figures/ECC_Nursery_MAPEs/" + title.replace(" ", "_")
    plt.ylabel("MAPE")
    plt.xlabel("Number of clusters")
    plt.title(title)
    plt.savefig(path)
    plt.clf()

    plt.plot(n_clusters, MAPEs_multi)
    title = "ECC and Nursery MAPE Model 2"
    path = "Saved_figures/ECC_Nursery_MAPEs/" + title.replace(" ", "_")
    plt.ylabel("MAPE")
    plt.xlabel("Number of clusters")
    plt.title(title)
    plt.savefig(path)
    plt.clf()

    plt.plot(n_clusters, MAPEs_hist)
    title = "ECC and Nursery Agregate MAPE Model 3"
    path = "Saved_figures/ECC_Nursery_MAPEs/" + title.replace(" ", "_")
    plt.ylabel("MAPE")
    plt.xlabel("Number of clusters")
    plt.title(title)
    plt.savefig(path)
    plt.clf()

    plt.plot(n_clusters, MAPEs_hist_multi)
    title = "ECC and Nursery Agregate MAPE Model 4"
    path = "Saved_figures/ECC_Nursery_MAPEs/" + title.replace(" ", "_")
    plt.ylabel("MAPE")
    plt.xlabel("Number of clusters")
    plt.title(title)
    plt.savefig(path)
    plt.clf()

    #Agregate forecast plots

    test_data = pd.DataFrame(test_data, columns=["Timestamp", "Temperature", "Time", "Weekdays", "Isweekend", "Load"])


    #Model 1
    print("Agregate Model 1 optimal number of clusters: ",n_clusters_optimal)
    prediction, MAPE = forecaster.predict(test_data)
    print("Agregate Model 1 MAPE: ",MAPE)
    plt.plot(test_data['Time'], test_data['Load'], label="Actual")
    plt.plot(test_data['Time'], prediction, label="Predicted")
    plt.legend()
    title = "Agregate Forecast Model 1"
    path = "Saved_figures/ECC_Nursery_forecasts/" + title.replace(" ", "_")
    plt.ylabel("Load (kWh, scaled down by a factor of 75)")
    plt.xlabel("Time (hours)")
    plt.title(title)
    plt.savefig(path)
    plt.clf()

    #Model 2
    print("Agregate Model 2 optimal number of clusters: ", n_clusters_optimal_multi)
    prediction, MAPE = forecaster_multi.predict(test_data)
    print("Agregate Model 2 MAPE: ", MAPE)
    plt.plot(test_data['Time'], test_data['Load'], label="Actual")
    plt.plot(test_data['Time'], prediction, label="Predicted")
    plt.legend()
    title = "Agregate Forecast Model 2"
    path = "Saved_figures/ECC_Nursery_forecasts/" + title.replace(" ", "_")
    plt.ylabel("Load (kWh, scaled down by a factor of 75)")
    plt.xlabel("Time (hours)")
    plt.title(title)
    plt.savefig(path)
    plt.clf()

    # Model 3
    print("Agregate Model 3 optimal number of clusters: ", n_clusters_optimal_hist)
    prediction, MAPE = forecaster_hist.predict(test_data, previous_days_load_test_data)
    print("Agregate Model 3 MAPE: ", MAPE)
    plt.plot(test_data['Time'], test_data['Load'], label="Actual")
    plt.plot(test_data['Time'], prediction, label="Predicted")
    plt.legend()
    title = "Agregate Forecast Model 3"
    path = "Saved_figures/ECC_Nursery_forecasts/" + title.replace(" ", "_")
    plt.ylabel("Load (kWh, scaled down by a factor of 75)")
    plt.xlabel("Time (hours)")
    plt.title(title)
    plt.savefig(path)
    plt.clf()

    #Model 4

    print("Agregate Model 4 optimal number of clusters: ",n_clusters_optimal_hist_multi)
    prediction, MAPE = forecaster_hist_multi.predict(test_data, previous_days_load_test_data)
    print("Agregate Model 4 MAPE: ", MAPE)
    plt.plot(test_data['Time'], test_data['Load'], label="Actual")
    plt.plot(test_data['Time'], prediction, label="Predicted")
    plt.legend()
    title = "Agregate Forecast Model 4"
    path = "Saved_figures/ECC_Nursery_forecasts/" + title.replace(" ", "_")
    plt.ylabel("Load (kWh, scaled down by a factor of 75)")
    plt.xlabel("Time (hours)")
    plt.title(title)
    plt.savefig(path)
    plt.clf()

    #Sum of forecast plots

    plt.plot(test_data['Time'], test_data['Load'], label="Actual")
    plt.plot(test_data['Time'], prediction_sum, label="Predicted")
    MAPE = mean_absolute_percentage_error(test_data['Load'], prediction_sum)
    print("Sum MAPE Model 1: ", MAPE)
    plt.legend()
    title = "Sum of meter forecasts Model 1"
    path = "Saved_figures/ECC_Nursery_forecasts/" + title.replace(" ", "_")
    plt.ylabel("Load (kWh, scaled down by a factor of 75)")
    plt.xlabel("Time (hours)")
    plt.title(title)
    plt.savefig(path)
    plt.clf()

    plt.plot(test_data['Time'], test_data['Load'], label="Actual")
    plt.plot(test_data['Time'], prediction_sum_multi, label="Predicted")
    MAPE = mean_absolute_percentage_error(test_data['Load'], prediction_sum_multi)
    print("Sum hist MAPE Model 2: ", MAPE)
    plt.legend()
    title = "Sum of meter forecasts Model 2"
    path = "Saved_figures/ECC_Nursery_forecasts/" + title.replace(" ", "_")
    plt.ylabel("Load (kWh, scaled down by a factor of 75)")
    plt.xlabel("Time (hours)")
    plt.title(title)
    plt.savefig(path)
    plt.clf()

    plt.plot(test_data['Time'], test_data['Load'], label="Actual")
    plt.plot(test_data['Time'], prediction_sum_hist, label="Predicted")
    MAPE = mean_absolute_percentage_error(test_data['Load'], prediction_sum_hist)
    print("Sum hist MAPE Model 3: ", MAPE)
    plt.legend()
    title = "Sum of meter forecasts Model 3"
    path = "Saved_figures/ECC_Nursery_forecasts/" + title.replace(" ", "_")
    plt.ylabel("Load (kWh, scaled down by a factor of 75)")
    plt.xlabel("Time (hours)")
    plt.title(title)
    plt.savefig(path)
    plt.clf()

    plt.plot(test_data['Time'], test_data['Load'], label="Actual")
    plt.plot(test_data['Time'], prediction_sum_hist_multi, label="Predicted")
    MAPE = mean_absolute_percentage_error(test_data['Load'], prediction_sum_hist_multi)
    print("Sum hist MAPE Model 4: ", MAPE)
    plt.legend()
    title = "Sum of meter forecasts Model 4"
    path = "Saved_figures/ECC_Nursery_forecasts/" + title.replace(" ", "_")
    plt.ylabel("Load (kWh, scaled down by a factor of 75)")
    plt.xlabel("Time (hours)")
    plt.title(title)
    plt.savefig(path)
    plt.clf()

    pred = np.array(prediction_sum_hist_multi)
    load = test_data['Load'].to_numpy()
    error = pred - load
    avg_error = []
    for i in range(48):
        avg_error.append(np.mean([abs(error[i]), abs(error[i + 48]), abs(error[i + 48*2]), abs(error[i + 48*3]), abs(error[i + 48*4]), abs(error[i+48*5]), abs(error[i + 48*6])]))

    plt.plot(range(0,48), avg_error, label = "Average Error, kWh/75")
    plt.legend()
    title = "Sum of meter forecasts Model 4 average absolute half-hourly error"
    path = "Saved_figures/ECC_Nursery_forecasts/" + title.replace(" ", "_")
    plt.ylabel("Load (kWh, scaled down by a factor of 75)")
    plt.xlabel("Time (half hours)")
    plt.title(title)
    plt.savefig(path)
    plt.show()

