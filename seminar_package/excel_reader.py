import pandas as pd
import numpy as np
from numpy import ndarray
from typing import Union, List, Dict
from scipy.stats import norm


def parameter_reader(filename: str) -> Dict[str, object]:
    """
    Method to get all parameters
    @param filename: the database of the Excel file
    @return: the dict of parameters
    """
    return {
        "actual demand in hours": actual_demand_hours(filename),
        "coefficient variation": coefficient_variation(filename),
        "distance matrix": distance_matrix(filename),
        "travel time matrix": travel_time_matrix(filename),
        "driver trailer": driver_trailer(filename),
        "driver trailer matrix": driver_trailer_matrix(filename),
        "forecast matrix": forecast_matrix(filename),
        "forecast matrix minutes": forecast_matrix_minutes(filename),
        "driver cost": driver_cost(filename),
        "min interval": min_interval(filename),
        "max driving": max_driving(filename),
        "number driver": number_driver(filename),
        "number trailer": number_trailer(filename),
        "number customer": number_customer(filename),
        "trailer capacity": trailer_capacity(filename),
        "trailer initial quantity": trailer_initial_quantity(filename),
        "trailer cost": trailer_cost(filename),
        "setup time": setup_time(filename),
        "tank capacity": tank_capacity(filename),
        "tank initial quantity": tank_initial_quantity(filename),
        "safety level": safety_level(filename),
        "time horizon": time_horizon(filename),
        "time windows": time_windows(filename),
        "time windows binary": time_windows_binary(filename)
    }


def distance_matrix(filename: str) -> ndarray:
    """
    Method to return the distance matrix parameters
    @param filename: the database of the Excel file
    @return: the array type of distance matrix
    """
    # read the file and transform to dataframe
    df = pd.read_excel(filename, "Travel distance")

    # extract the distances, reshape the numpy array to matrix
    distances = df['Kilometers'].values
    num_distances = distances.shape[0]
    num_locations = int(num_distances ** 0.5)
    distances = distances.reshape(num_locations, num_locations)

    return distances


def travel_time_matrix(filename: str) -> ndarray:
    """
    Method to return the travel time matrix
    @param filename: the database of the Excel file
    @return: the traveling time matrix
    """
    # read the file and transform to dataframe
    df = pd.read_excel(filename, "Travel time")

    # extract the time, reshape the numpy array to matrix
    times = df['Minutes'].values
    num_times = times.shape[0]
    num_locations = int(num_times ** 0.5)
    times = times.reshape(num_locations, num_locations)
    for i in range(times.shape[0]):
        times[i,i] = 1
    return times


def driver_cost(filename: str) -> Union[float, ndarray]:
    """
    Method to get the driver cost from the database
    Scalar if all equal to each or an array
    @param filename: the database of the Excel file
    @return: the cost of each driver
    """
    # read the file and transform to dataframe
    df = pd.read_excel(filename, "Drivers")

    # extract the cost column and correspond to each driver
    costs = df['Cost per minute'].values

    # if all equal, then a scalar otherwise an array
    # cvxpy recommends to use numpy.float64
    if len(set(costs)) == 1:
        return costs[0]
    return costs


def min_interval(filename: str) -> Union[float, ndarray]:
    """
    Method to get the min interval minutes of each driver
    @param filename: the database of the Excel file
    @return: scalar if all equal or an array
    """
    # read the file and transform to dataframe
    df = pd.read_excel(filename, "Drivers")

    # extract the min interval minutes
    intervals = df['Minimum time between shifts'].values

    # if all equal, return a scalar other an array
    if len(set(intervals)) == 1:
        return intervals[0]
    return intervals


def max_driving(filename: str) -> Union[float, ndarray]:
    """
    Method to get the max driving time of each driver
    @param filename: the database of the Excel file
    @return: scalar if all equal or an array
    """
    # read the file and transform to dataframe
    df = pd.read_excel(filename, "Drivers")

    # extract the min interval minutes
    max_time = df['Maximum driving time'].values

    # if all equal, return a scalar other an array
    if len(set(max_time)) == 1:
        return max_time[0]
    return max_time


def number_driver(filename: str) -> int:
    """
    Method to get the number of the drivers
    @param filename: the database of Excel file
    @return: the number of drivers in a region
    """
    return len(pd.read_excel(filename, "Drivers"))


def number_trailer(filename: str) -> int:
    """
    Method to get the number of the trailers
    @param filename: the database of the Excel file
    @return: the number of trailers
    """
    # read the file and transform into dataframe
    df = pd.read_excel(filename, "Trailers")

    # extract the number of trailers
    return len(df['Trailer index'])


def trailer_capacity(filename: str) -> Union[float, ndarray]:
    """
    Method to get the capacity of the trailers
    @param filename: the database of the Excel file
    @return: scalar of the capacity otherwise array
    """
    # read the file and transform into dataframe
    df = pd.read_excel(filename, "Trailers")

    # extract the capacity for each trailer
    capacity = df['Capacity'].values

    # if all equal, scalar otherwise an array
    if len(set(capacity)) == 1:
        return capacity[0]
    return capacity


def trailer_initial_quantity(filename: str) -> Union[float, ndarray]:
    """
    Method to get the initial quantity of each trailer
    @param filename: the database of the Excel file
    @return: scalar of the initial quantity otherwise array
    """
    # read the file and transform into dataframe
    df = pd.read_excel(filename, "Trailers")

    # extract the capacity for each trailer
    initials = df['Initial quantity'].values

    # if all equal, scalar otherwise an array
    if len(set(initials)) == 1:
        return initials[0]
    return initials


def trailer_cost(filename: str) -> Union[float, ndarray]:
    """
    Method to get the cost of each trailer
    @param filename: the database of the Excel file
    @return: scalar of the cost otherwise array
    """
    # read the file and transform into dataframe
    df = pd.read_excel(filename, "Trailers")

    # extract the capacity for each trailer
    costs = df['Cost per kilometer'].values

    # if all equal, scalar otherwise an array
    if len(set(costs)) == 1:
        return costs[0]
    return costs


def driver_trailer(filename: str) -> ndarray:
    df = pd.read_excel(filename, "Trailer drivers")
    return df.values


def driver_trailer_matrix(filename: str) -> ndarray:
    """
    Method to get the driver trailer matrix
    on whether the driver can steer the trailer or not
    Binary element where 1 means yes and 0 means no
    @param filename: the database of the Excel file
    @return: the binary matrix of driver and trailer
    """
    # initiate an empty matrix
    matrix: List[List[int]] = []

    # read the file and transform into dataframe
    df = pd.read_excel(filename, "Trailer drivers")

    # use the for loop to store the binary value into the matrix
    trailers = df["Trailer index"]
    for i in range(number_driver(filename)):
        driver_trailer_list: List[int] = []
        for j in range(number_trailer(filename)):
            if j == trailers[i]:
                driver_trailer_list.append(1)
            else:
                driver_trailer_list.append(0)
        matrix.append(driver_trailer_list)
    return np.array(matrix)


def get_index_source(filename: str) -> int:
    """
    Method to get the index of the source
    @param filename: the database of the Excel file
    @return: the index of source
    """
    return pd.read_excel(filename, "Sources")["Location index"][0]


def get_index_base(filename: str) -> int:
    """
    Method to get the index of the base
    @param filename: the database of the Excel file
    @return: the index of base
    """
    return pd.read_excel(filename, "Bases")["Location index"][0]


def loading_time(filename: str) -> float:
    """
    Method to get the loading time
    @param filename: the database of the Excel file
    @return: the loading time
    """
    return pd.read_excel(filename, "Sources")["Setup time"][0]


def setup_time(filename: str) -> ndarray:
    """
    Method to get the setup time of each location
    @param filename: the database of the Excel file
    @return: the array containing all loading and unloading time
    """
    # read the file and transform into array
    df = pd.read_excel(filename, "Customers")
    times = df["Setup time"].values

    # insert
    times = np.insert(times, 0, loading_time(filename))
    times = np.insert(times, 0, 0)

    return times


def tank_capacity(filename: str) -> ndarray:
    """
    Method to get the capacity array of each tank
    @param filename: the database of each location
    @return: the array containing the tank capacity of each location
    """
    # read the file and transform into array
    df = pd.read_excel(filename, "Customers")
    capacities = df["Capacity"].values

    # insert
    capacities = np.insert(capacities, 0, 999999999)
    capacities = np.insert(capacities, 0, 0)

    return capacities


def tank_initial_quantity(filename: str) -> ndarray:
    """
    Method to get the initial tank quantity
    @param filename: the database of the Excel file
    @return: the array containing the initial tank quantity
    """
    # read the file and transform into array
    df = pd.read_excel(filename, "Customers")
    initial = df["Initial tank quantity"].values

    # insert
    initial = np.insert(initial, 0, 999999999)
    initial = np.insert(initial, 0, 0)

    return initial


def safety_level(filename: str) -> ndarray:
    """
    Method to get the safety level
    @param filename: the database of the Excel file
    @return: the array containing the safety level
    """
    # read the file and transform into array
    df = pd.read_excel(filename, "Customers")
    levels = df["Safety level"].values

    # insert
    levels = np.insert(levels, 0, 0)
    levels = np.insert(levels, 0, 0)

    return levels


def number_customer(filename: str) -> int:
    """
    Method to get the number of customers
    @param filename: the database of the Excel file
    @return: the number of customers
    """
    return len(pd.read_excel(filename, "Customers"))


def forecast_matrix(filename: str) -> ndarray:
    """
    Method to get the forecast matrix
    @param filename: the database of the Excel file
    @return: the forecast matrix
    """
    # read the file and transform into array
    df = pd.read_excel(filename, "Forecast")
    forecast = df["Forecast"].values

    # reshape the data to get the matrix
    two_locations = np.zeros((2, int(len(forecast)/number_customer(filename))))

    return np.vstack((two_locations,
                      forecast.reshape(number_customer(filename), int(len(forecast)/number_customer(filename)))))


def forecast_matrix_minutes(filename: str) -> ndarray:
    """
    Method to transform the forecast matrix per minute (mostly 0 except at the hours)
    @param filename: the database of the Excel file
    @return: the forecast matrix in minutes
    """
    hourly_forecast = forecast_matrix(filename)
    forecast = np.zeros((number_customer(filename)+2 ,time_horizon(filename) + 1))
    rows, columns = hourly_forecast.shape
    for i in range(rows):
        for j in range(columns):
            forecast[i, (j+1)*60] = hourly_forecast[i, j]
    return forecast


def time_horizon(filename: str) -> int:
    """
    Method to find the length of the time horizon in minutes
    @param filename: the database of the Excel file
    @return the length of the time horizon
    """
    return int(60 * len(pd.read_excel(filename, "Forecast")) / len(pd.read_excel(filename, "Customers")))


def time_windows(filename: str) -> ndarray:
    """
    Method to simply return the time windows of the drivers.
    @param filename: the database of the Excel file
    @return: the original time window matrix
    """
    df = pd.read_excel(filename, "Time windows drivers")
    return df.values


def time_windows_binary(filename: str) -> ndarray:
    """
    Method to create a binary matrix out of the time horizon and time windows of drivers.
    @param filename: the database of the Excel file
    @return: the binary time window matrix 
    """
    time_window = time_windows(filename)
    binary_time_windows = np.zeros((number_driver(filename), time_horizon(filename) + 1))
    for row in time_window:
        for t in range(row[1], row[2]):
            binary_time_windows[row[0], t] = 1
    return binary_time_windows


def coefficient_variation(filename: str) -> ndarray:
    """
    Method to get the demand variance of customers
    @param filename: the database of the Excel file
    @return: the demand variance matrix
    """
    # read the file and transform into array
    df = pd.read_excel(filename, "Customers")
    coefficient = df["Coefficient of variation"].values

    # return the demand standard deviation
    new_coefficient = np.insert(coefficient, 0, [0, 0])
    return new_coefficient


def demand_upperbound_minutes(filename: str, prob: float = 0.95) -> ndarray:
    """
    Method to get the demand upperbound in minutes
    @param filename: data set file
    @param prob: the (prob)th quantile of normal distribution
    @return: demand upperbound in minutes
    """
    mean_minutes = forecast_matrix_minutes(filename)
    coefficient = coefficient_variation(filename)
    z = norm.ppf(prob)

    for i in range(2, number_customer(filename) + 2):
        for j in range(len(mean_minutes[i])):
            if j != 0 and j % 60 == 0:
                mean_minutes[i, j] = mean_minutes[i, j] * coefficient[i] * z + mean_minutes[i, j]
    return mean_minutes


def normal_simulator(filename: str, seed: int = 42) -> ndarray:
    """
    Method to simulate a normal distribution (must set the same seed with actual demand)
    @param seed: the seed to keep the simulator fixed
    @param filename: data set file
    @return: a simulator of demands
    """
    np.random.seed(seed)
    mean_minutes = forecast_matrix_minutes(filename)
    coefficient = coefficient_variation(filename)

    for i in range(2, number_customer(filename) + 2):
        for j in range(len(mean_minutes[i])):
            if j != 0 and j % 60 == 0:
                mean_minutes[i, j] = np.random.normal(mean_minutes[i, j], coefficient[i] * mean_minutes[i, j])
                if mean_minutes[i, j] <= 0:
                    mean_minutes[i, j] = 0
    return mean_minutes


def actual_demand_hours(filename: str, seed: int = 42) -> ndarray:
    """
    Method to simulate to get the actual demand in hours and to fit the JSON output event list checker
    (must set the same seed as the normal distribution simulator)
    @param seed: the seed to keep the simulator fixed
    @param filename: the data of the file
    """
    np.random.seed(seed)
    mean = forecast_matrix(filename)
    coefficient = coefficient_variation(filename)

    for i in range(2, number_customer(filename) + 2):
        for j in range(len(mean[i])):
            mean[i, j] = np.random.normal(mean[i, j], coefficient[i] * mean[i, j])
            if mean[i, j] <= 0:
                mean[i, j] = 0
    return mean
