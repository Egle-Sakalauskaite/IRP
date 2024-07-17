from seminar_package import excel_reader
from seminar_package import heuristics_helper
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def num_of_relevant_customers(file_path):
    """Checks for each data instance how many customers use up all of their medical air before the end of the time horizon"""
    forecast = excel_reader.forecast_matrix(file_path)
    initial = excel_reader.tank_initial_quantity(file_path)
    safety = excel_reader.safety_level(file_path)
    total_used = np.sum(forecast, axis=1)
    initial_available = initial - safety
    return np.sum(initial_available - total_used < 0)

def min_number_of_visits(file_path):
    """Returns a lower bound of the number of visits each location requires durring the time horizon"""
    forecast = excel_reader.forecast_matrix(file_path)
    initial = excel_reader.tank_initial_quantity(file_path)
    safety = excel_reader.safety_level(file_path)
    capacity = excel_reader.tank_capacity(file_path)
    max_trailer =  max(excel_reader.trailer_capacity(file_path))
    current = initial
    results = np.zeros(current.size)

    for t in range(forecast.shape[1]):
        current -= forecast[:, t]

        for i in range(current.size):
            if current[i] < safety[i]:
                current[i] = min(capacity[i], max_trailer)
                results[i] += 1

    return results

def time_of_first_safety_level_reach(file_path):
    """Returns the hour in which each location reaches its safety level for the first time"""
    forecast = excel_reader.forecast_matrix(file_path)
    initial = excel_reader.tank_initial_quantity(file_path)
    safety = excel_reader.safety_level(file_path)
    results = np.zeros(initial.size)
    checked = set()

    for t in range(forecast.shape[1]):
        initial -= forecast[:, t]

        for i in range(initial.size):
            if i not in checked and initial[i] <= safety[i]:
                results[i] = t
                checked.add(i)

    return results

def triangle_inequality_distance(file_path):
    """Checks whether triangle inequality holds for all locations for distance"""
    violated_locations = set()
    distance_matrix = excel_reader.distance_matrix(file_path)
    num_customers = excel_reader.number_customer(file_path)
    for i in range(num_customers):
        for j in range(num_customers):
            if i == j:
                continue
            for k in range(num_customers):
                if k == i or k == j:
                    continue
                if distance_matrix[i,j] > distance_matrix[i,k] + distance_matrix[k,j]:
                    violated_locations.add((i,k,j))

    if not violated_locations:
        print("The triangle inequality holds.")
    else:
        print("Triangle inequality does not hold for the following:")
        print(violated_locations)


def triangle_inequality_time(file_path):
    """Checks whether triangle inequality holds for all locations for travel time"""
    violated_locations = set()
    travel_time_matrix = excel_reader.travel_time_matrix(file_path)
    num_customers = excel_reader.number_customer(file_path)
    for i in range(num_customers):
        for j in range(num_customers):
            if i == j:
                continue
            for k in range(num_customers):
                if k == i or k == j:
                    continue
                if travel_time_matrix[i,j] > travel_time_matrix[i,k] + travel_time_matrix[k,j]:
                    violated_locations.add((i,k,j))
    if not violated_locations:
        print("The triangle inequality holds.")
    else:
        print("Triangle inequality does not hold for the following:")
        print(violated_locations)


def min_interval_check(filename: str) -> bool:
    """
    check if the interval between two consecutive time windows meets the min interval
    @param filename: the database of Excel file
    @return: boolean value whether all intervals satisfy the min interval
    """
    # read the file and get the data frame
    df = pd.read_excel(filename, "Time windows drivers")

    # extract the col of time windows
    start_time = df["Start time"].values
    end_time = df["End time"].values

    # get the min interval constraint value
    min_interval = excel_reader.min_interval(filename)

    all_interval = True
    for i in range(len(start_time) - 1):
        interval = start_time[i + 1] - end_time[i]
        if interval <= 0:
            continue
        else:
            if interval < min_interval:
                all_interval = False
    return all_interval


def interval_show(filename: str) -> List[float]:
    """
    check min interval values
    @param filename: the database of Excel file
    @return: interval values between consecutive time windows
    """
    # read the file and get the data frame
    df = pd.read_excel(filename, "Time windows drivers")

    # extract the col of time windows
    start_time = df["Start time"].values
    end_time = df["End time"].values

    # get the min interval constraint value
    min_interval = excel_reader.min_interval(filename)

    all_interval = []
    for i in range(len(start_time) - 1):
        interval = start_time[i + 1] - end_time[i]
        if interval > 0:
            all_interval.append(interval)
    return all_interval


def max_duration_check(filename: str) -> bool:
    """
    check the max duration constraint
    @param filename: the database of the Excel file
    @return: the boolean value if all interval of a time window meets the constraint
    """
    # read the file and get the data frame
    df = pd.read_excel(filename, "Time windows drivers")

    # extract the col of time windows
    start_time = df["Start time"].values
    end_time = df["End time"].values

    # get the max duration constraint
    duration = excel_reader.max_driving(filename)

    # check if it holds
    bool = True
    for i in range(len(start_time)):
        if end_time[i] - start_time[i] > duration:
            bool = False
    return bool


def duration_show(filename: str) -> List[float]:
    """
    show the values of duration of a time window
    @param filename: the database of the Excel file
    @return: the list of all duration values
    """
    # read the file and get the data frame
    df = pd.read_excel(filename, "Time windows drivers")

    # extract the col of time windows
    start_time = df["Start time"].values
    end_time = df["End time"].values

    return end_time - start_time


def same_trailer_driver_overlap(file_path):
    """Returns true if any two drivers that operate the same trailer are available at the same time"""
    time_windows = excel_reader.time_windows_binary(file_path)
    drivers_trailers = excel_reader.driver_trailer_matrix(file_path)
    for trailer_idx in range(drivers_trailers.shape[1]):
        sharing_drivers = np.where(drivers_trailers[:, trailer_idx] == 1)  
        to_compare = time_windows[sharing_drivers]

        for t in range(time_windows.shape[1]):
            if np.sum(to_compare[:, t] > 1):
                return True

    return False

def plot_availability(file_path):
    """x axis: time horizon, y axis: drivers"""
    df = pd.read_excel(file_path, sheet_name="Time windows drivers")
    time_windows = df.values
    periods = heuristics_helper.periods_in_order(time_windows)
    driver_idx = np.array([0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1])
    periods = np.insert(periods, 0, driver_idx, axis=1)
    print(periods)
    plt.figure(figsize=(10, 4))
    num_drivers = len(df['Driver index'].unique())
    color_palette = [(0.7, 0.3, 0.3), (0.3, 0.3, 0.7)]

    # for i in range(periods.shape[0]):
    #     color = color_palette[periods[i, 0]]
    #     plt.plot([periods[i, 1], periods[i, 2]], [driver_idx, driver_idx], linewidth=2, color=color)

    for i, driver_idx in enumerate(df['Driver index'].unique()):
        driver_data = df[df['Driver index'] == driver_idx]
        for _, row in driver_data.iterrows():
            color = color_palette[i]
            plt.plot([row['Start time'], row["End time"]], [driver_idx, driver_idx], linewidth=2, color=color)

    plt.xlabel('Time')
    plt.ylabel('Driver Index')
    plt.title('Driver Availability Windows')
    plt.yticks(df['Driver index'].unique(), [f'Driver {idx}' for idx in df['Driver index'].unique()])
    plt.grid(True)
    plt.savefig('./output/availability_windows_tiny.png')
    plt.show()

plot_availability("./data/Tinyland.xlsx")