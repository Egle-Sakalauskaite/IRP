from sklearn.discriminant_analysis import StandardScaler
import numpy as np
from collections import deque
from sklearn.cluster import SpectralClustering
import warnings
from seminar_package import excel_reader
# from . import excel_reader
warnings.filterwarnings("ignore", category=UserWarning, message="The spectral clustering API has changed.*")
warnings.filterwarnings("ignore", category=UserWarning, message="Graph is not fully connected")

def update_state(parameters, forecast, initial_trailer_tank, initial_inventory, customers_in_cluster, start_time, result):
    """update the state. Keep in mind that customer cluster inventory has all the locations,
    but to avoid indexing error, only locations of this cluster are updated"""
    setup_time = parameters["setup time"]
    salary = parameters["driver cost"]
    trailer_cost = parameters["trailer cost"]
    distance = parameters["distance matrix"]

    state = {"current tank level": initial_trailer_tank.copy(),
             "current cluster inventories": initial_inventory.copy(),
             "customers": customers_in_cluster.copy(),
             "shift length": 0,
             "shift cost": 0,
             "current location": 0,
             "current time": start_time}
    
    for operation in result["Operations"]:
        origin = state["current location"]
        destination = operation["Location index"]
        current_time = state["current time"]
        arrival_time = operation["Arrival time"]

        state["current tank level"] -= operation["Quantity"]
        state["current cluster inventories"][customers_in_cluster] = consume(state["current cluster inventories"], customers_in_cluster, forecast, current_time, operation["Arrival time"] + setup_time[destination])
        state["current cluster inventories"][destination] += operation["Quantity"]
        state["shift length"] = int(arrival_time + setup_time[destination] - result["Start time"])
        state["shift cost"] = salary*state["shift length"]
        state["shift cost"] += trailer_cost*distance[origin][destination]
        state["current location"] = int(operation["Location index"])
        state["current time"] = int(operation["Arrival time"] + setup_time[destination])

        check_state(parameters, customers_in_cluster, state, result["Trailer index"])
    
    return state

def insert_operation(parameters, result, operation, insert_position):
    """inserts the new operation at specified position and updates arival times for all operations that go after
    if no insert position provided, simply appends the operation as the last one"""
    if insert_position is None or insert_position == len(result["Operations"]):
        result["Operations"].append(operation)
    else:
        new_location = operation["Location index"]
        next_location = result["Operations"][insert_position]["Location index"]
        if insert_position > 0:
            previous_location = result["Operations"][insert_position - 1]["Location index"]
        else:
            previous_location = 0
        result["Operations"].insert(insert_position, operation)
        travel_time = parameters["travel time matrix"]
        setup_time = parameters["setup time"]
        extra_time = travel_time[new_location, next_location] + setup_time[new_location] + travel_time[previous_location, new_location] - travel_time[previous_location, next_location]

        # adjust for all the locations that go after arrival times
        for i in range(insert_position + 1, len(result["Operations"])):
            result["Operations"][i]["Arrival time"] += extra_time
    
    return result

def identify_cluster(clusters, location):
    """checks what cluster does the location belong to"""
    for cluster, locations in clusters.items():
        if location in locations:
            return cluster
        
    return None

def delivery_quantity(parameters, state, customer_inventory, customer):
    """"try delivering total amount that the customer needs before the end of the time horizon,
    if that is too large, deliver everything that is in the trailer or the max amount that fits in customers tank"""
    capacity = parameters["tank capacity"][customer]
    trailer_inventory = state["current tank level"]
    available_space = capacity - customer_inventory
    return min(trailer_inventory, available_space)

def minimal_delivery_quantity(parameters, initial_inventory, trailer, customer):
    """estimate a delivery quantity that does not depend on exact arrival time or current trailer inventory"""
    trailer_capacity = parameters["trailer capacity"][trailer]
    safety = parameters["safety level"][customer]
    minimal_available_space = initial_inventory[customer] - safety
    return min(trailer_capacity, minimal_available_space)


def check_state(parameters, customers_in_cluster, state, trailer):
    """checks if the state is feasible"""
    trailer_capacity = parameters["trailer capacity"]
    travel_durations = parameters["travel time matrix"]
    max_shift_length = parameters["max driving"]
    safety = parameters["safety level"]
    customer_capacity = parameters["tank capacity"]

    if state["current tank level"] > trailer_capacity[trailer]:
        raise ValueError(f"trailer capacity violated: {state['current tank level']} > {trailer_capacity[trailer]}")
    if 0 > state["current tank level"]:
        raise ValueError(f"trailer capacity violated: {state['current tank level']} < 0")
    if not 0 <= state["shift length"] + travel_durations[state["current location"], 0] <= max_shift_length:
        raise ValueError(f"shift length violated: {state['shift length'] + travel_durations[state['current location'], 0]} < {max_shift_length}")
    for customer in customers_in_cluster:
        if np.any(state["current cluster inventories"][customer] < safety[customer]):
            raise ValueError(f"safety level of customer {customer} violated: {state['current cluster inventories'][customer]} < {safety[customer]}")
        if np.any(state["current cluster inventories"][customer] > customer_capacity[customer]):
            raise ValueError(f"capacity of customer {customer} violated: {state['current cluster inventories'][customer]} > {customer_capacity[customer]}")
    
def clustering_customers_travel_cost(parameters, n_clusters, filtered_customers):
    """Clusters customers according to their proximity.
    Returns the clusters as a dictionary with key as the cluster index and value as a list of customer indexes that belong to that cluster"""
    clusters = {}
    if n_clusters == 1:
        clusters[0] = filtered_customers
    elif len(filtered_customers) <= n_clusters:
        for i, customer in enumerate(filtered_customers):
            clusters[i] = [customer]
    else:
        travel_durations = parameters["travel time matrix"]
        distances = parameters["distance matrix"]
        travel_cost = parameters["trailer cost"]
        salary_cost = parameters["driver cost"]
        
        costs_matrix = travel_cost * distances + salary_cost * travel_durations
        filtered_costs_matrix = costs_matrix[filtered_customers[:, None], filtered_customers]
        scaler = StandardScaler()
        costs_normalized = scaler.fit_transform(filtered_costs_matrix)
        spectral_clustering = SpectralClustering(n_clusters=n_clusters, n_neighbors=min(n_clusters, len(filtered_customers)), affinity='nearest_neighbors', random_state=42)
        cluster_labels = spectral_clustering.fit_predict(costs_normalized)
        
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = [filtered_customers[i]]
            else:
                clusters[label].append(filtered_customers[i])
    
    return clusters

def cluster_driver_assignment(clusters, filtered_customers, trailers):
    """Assign clusters to drivers according to the biggest single customer demand and driver's trailer size.
    If multiple trailers can handle such demand, assign according to total cluster demand"""
    cluster_properties = []
    trailers_sorted = sorted(trailers, key=lambda x: x[2], reverse=True)
    result = {}

    for cluster, customers in clusters.items():
        total_demand = 0
        for customer in customers:
            demand = filtered_customers[customer]
            total_demand += demand

        cluster_properties.append((cluster, total_demand))

    cluster_sorted = sorted(cluster_properties, key=lambda x: x[1], reverse=True)

    for i, cluster in enumerate(cluster_sorted):
        result[cluster[0]] = trailers_sorted[i][1]

    return result

def available_trailers(parameters, available_drivers):
    """returns a list of available trailers and their capacities"""
    capacities = parameters["trailer capacity"]
    driver_trailer = parameters["driver trailer"]
    result = []

    for driver in available_drivers:
        trailer = driver_trailer[driver][1]
        result.append((trailer, driver, capacities[trailer]))

    return result

"replace forecasted demand with the upper bound of expected demand"
def ranking_urgency(safety, forecast, inventory, current_time):
    """"Ranks the locations according to their urgency.
    Returns a queue of tuples (customer index, time t at which the safety level will be violated if no refills), customer with highest urgency at the front"""
    urgency_queue = deque()
    added = set()
    total_demands = np.zeros(len(safety))

    for t in range(current_time, forecast.shape[1]):
        inventory -= forecast[:, t]

        for customer in range(2, forecast.shape[0]):
            if customer not in added and inventory[customer] <= safety[customer]:
                urgency_queue.append((customer, t))
                added.add(customer)

    total_demands =  np.clip((safety - inventory), a_min=0,  a_max=None)
    return urgency_queue, total_demands

def filter_customers(parameters, urgency, total_demands, initial_inventory, end_time, SAFETY_TIME):
    """identifies the customers that should be visited before the end of the time horizon"""
    travel_time = parameters["travel time matrix"]
    capacity = parameters["tank capacity"]
    min_available_space = capacity[2:] - initial_inventory[2:]
    min_available_space = np.concatenate((np.zeros(2), min_available_space))
    filtered_customers = {}

    # add customers that will reach their safety level soon
    for customer, deadline in urgency:
        if end_time + travel_time[0, customer] + SAFETY_TIME > deadline:
            filtered_customers[customer] = min_available_space[customer]


    # add customers that have enough space to fill their total demand:
    for customer in range(2, len(total_demands)):
        if 0 < total_demands[customer] <= min_available_space[customer]:
            filtered_customers[customer] = min_available_space[customer]
            
    return filtered_customers

def delivery_time_windows(urgency, filtered_customers):
    """constructs delivery time windows of filtered customers
    according to when their inventory is empty enough, but has not reached the safety level yet"""
    result = {}

    while len(result) < len(filtered_customers):
        customer_urgency = urgency.pop()
        idx = customer_urgency[0]

        if idx in filtered_customers:
            delivery_earliest = filtered_customers[idx][0]
            delivery_latest = customer_urgency[1]
            minimal_amount = filtered_customers[idx][1]
            result[idx] = (delivery_earliest, delivery_latest, minimal_amount)

    return result

def periods_in_order(time_windows):
    """sorts the drivers availability time windows according to their start time,
    if needed trips some time windows to avoid overlapping"""
    # sort according to start time
    start_time_sorted = np.argsort(time_windows[:, 1])
    time_windows_sorted = time_windows[start_time_sorted]

    # retain only unique time windows
    time_windows_unique = np.unique(time_windows_sorted[:, 1:], axis=0)

    # trim some time windows to eliminate overlaps
    # only in tinyland driver 0 shifts get trimmed
    for i in range(time_windows_unique.shape[0] - 1):
        if time_windows_unique[i, 1] > time_windows_unique[i+1, 0]:
            time_windows_unique[i, 1] = time_windows_unique[i+1, 0]
    
    return time_windows_unique

def periods_of_the_day(time_windows, day, time_horizon):
    "Returns the time periods that start on the specified day after 18:00 or on the next day before 18:00"
    all_periods = periods_in_order(time_windows)
    periods_today_evening = all_periods[(all_periods[:, 0] / 60 // 24 == (day - 1)) & (all_periods[:, 0] / 60 % 24 >= 18)]
    periods_tomorrow = all_periods[(all_periods[:, 0] / 60 // 24 == day) & (all_periods[:, 0] / 60 % 24 < 18)]
    filtered_periods = np.concatenate((periods_today_evening, periods_tomorrow), axis=0)
    if not len(filtered_periods) == 0:
        next_period_idx = np.where(all_periods[:, 0] == filtered_periods[-1, 0])[0] + 1
        if next_period_idx < all_periods.shape[0]:
            time_horizon = all_periods[next_period_idx, 0]
    
    if not isinstance(time_horizon, int):
        time_horizon = time_horizon[0]

    return filtered_periods, time_horizon

def available_drivers(time_windows, start_time, end_time):
    """identifies the drivers that are available for the entire given time period"""
    result = set()

    for driver in range(time_windows.shape[0]):
        if np.all(time_windows[driver, start_time:end_time]):
            result.add(driver)

    return result

"""Make the consumption approximate according to the upper bounds of the expected demand"""
def consume(current_inventory, customers_list, forecast, start_time, end_time):
    """performs consumption for a specified time period"""
    for t in range(start_time, end_time):
        current_inventory[customers_list] -= forecast[customers_list, t]

    return current_inventory[customers_list]

def update_initial_trailer(initial_trailer, period_result):
    """updates the trailer inventories according to the operations that were performed in this period"""
    for cluster, shift in period_result.items():
        trailer = shift["Trailer index"]
        for operation in shift["Operations"]:
            initial_trailer[trailer] -= operation["Quantity"]
    return initial_trailer

def add_initial_customers(initial_inventory, period_result):
    """üpdates the customers inventories according to the operations that were performed in this period"""
    for cluster, shift in period_result.items():
        for operation in shift["Operations"]:
            location = operation["Location index"]
            if location  == 1:
                continue
            initial_inventory[location] += operation["Quantity"]
    return initial_inventory

def check_feasibility(solution, parameters, real_demand):
    """checks if solution is feasible and returns its total cost"""
    distance = parameters["distance matrix"]
    travel_time = parameters["travel time matrix"]
    setup_time = parameters["setup time"]
    driver_trailer = parameters["driver trailer"]
    cost = 0

    minutes_below_safety, minutes_below_0 = check_feasibility_inventories(solution, parameters, real_demand)

    for shift in solution:
        trailer = shift["Trailer index"]
        driver = shift["Driver index"]

        if driver_trailer[driver][1] != trailer:
            raise ValueError(f"driver {driver} is riding a wrong trailer {trailer}")

        current_time = shift["Start time"]
        current_location = 0

        for operation in shift["Operations"]:
            next_location = operation["Location index"]
            next_time = operation["Arrival time"]
            time = current_time + setup_time[current_location] + travel_time[current_location, next_location]
            
            if time > next_time:
                raise ValueError(f"travel time violated for operation {operation}")
            
            cost += parameters["trailer cost"]*distance[current_location, next_location]
            current_location = next_location
            current_time = next_time
        
        shift_length = current_time + setup_time[current_location] + travel_time[current_location, 0] - shift["Start time"]

        if shift_length > parameters["max driving"]:
            raise ValueError(f"Maximum driving time violated for shift {shift}")

        cost += parameters["driver cost"]*shift_length
        cost += parameters["trailer cost"]*distance[current_location,0]

    return cost, minutes_below_safety, minutes_below_0
        
def check_feasibility_inventories(solution, parameters, real_demand):
    """checks if customer and triler inventory constraints are violated"""
    current_inventory = parameters["tank initial quantity"].astype(np.float64)
    current_trailer_tank = parameters["trailer initial quantity"].astype(np.float64)
    forecast = real_demand
    minutes_below_safety = np.zeros(len(current_inventory))
    minutes_below_0 = np.zeros(len(current_inventory))

    for t in range(parameters["time horizon"]):
        current_inventory -= forecast[:, t]

        for shift in solution:
            trailer = shift["Trailer index"]

            for operation in shift["Operations"]:
                if operation["Arrival time"] == t:
                    location = operation["Location index"]
                    quantity = operation["Quantity"]
                    current_inventory[location] += quantity
                    current_trailer_tank[trailer] -= quantity

        for customer in range(len(current_inventory)):
            if current_inventory[customer] < parameters["safety level"][customer]:
                minutes_below_safety[customer] += 1
            if current_inventory[customer] < 0:
                minutes_below_0[customer] += 1

        if np.any(current_inventory < parameters["safety level"]):
            # raise ValueError(f"safety level violated at time {t}")
            print(f"safety level violated at time {t}")
        if np.any(current_inventory < 0):
            # raise ValueError(f"customer inventory dropped below 0 at time {t}")
            print(f"customer inventory dropped below 0 at time {t}")
        if np.any(current_inventory > parameters["tank capacity"]):
            # raise ValueError(f"customer capacity violated at time {t}")
            print(f"customer capacity violated at time {t}")
        if np.any(current_trailer_tank < 0):
            # raise ValueError(f"negative trailer inventory at time {t}")
            print(f"negative trailer inventory at time {t}")
        if np.any(current_trailer_tank > parameters["trailer capacity"]):
            # raise ValueError(f"trailer capacity violated at time {t}")
            print(f"trailer capacity violated at time {t}")

    return minutes_below_safety, minutes_below_0

    # parameters = excel_reader.parameter_reader(file)
    # current_inventory = parameters["tank initial quantity"].astype(np.float64)
    # current_trailer_tank = parameters["trailer initial quantity"].astype(np.float64)
    # capacity_customer = parameters["tank capacity"]
    # capacity_trailer = parameters["trailer capacity"]
    # minutes_below_safety = np.zeros(len(current_inventory))
    #
    # for t in range(parameters["time horizon"]):
    #     current_inventory -= real_demand[:, t]
    #
    #     for shift in solution:
    #         trailer = shift["Trailer index"]
    #         available_space_trailer = capacity_trailer[trailer] - current_trailer_tank[trailer]
    #
    #         for operation in shift["Operations"]:
    #             if operation["Arrival time"] == t:
    #                 location = operation["Location index"]
    #                 quantity = operation["Quantity"]
    #                 available_space_location = capacity_customer[location] - current_inventory[location]
    #
    #                 if location == 1 and quantity > available_space_trailer:
    #                     current_trailer_tank[trailer] -= available_space_trailer
    #                 elif quantity > available_space_location:
    #                     current_inventory[location] += available_space_location
    #                     current_trailer_tank[trailer] -= available_space_location
    #                 else:
    #                     current_inventory[location] += quantity
    #                     current_trailer_tank[trailer] -= quantity
    #
    #
    #     for customer in range(len(current_inventory)):
    #         if current_inventory[customer] < parameters["safety level"][customer]:
    #             minutes_below_safety += 1
    #
    #
    #     if np.any(current_inventory < 0):
    #         raise ValueError(f"customer inventory dropped below 0 at time {t}")
    #     if np.any(current_inventory > parameters["tank capacity"]):
    #         raise ValueError(f"customer capacity violated at time {t}")
    #     if np.any(current_trailer_tank < 0):
    #         raise ValueError(f"negative trailer inventory at time {t}")
    #     if np.any(current_trailer_tank > parameters["trailer capacity"]):
    #         raise ValueError(f"trailer capacity violated at time {t}")
    #
    # return minutes_below_safety


def check_below_0(solution, parameters, real_demand):
    """checks if customer and trailer inventory constraints are violated"""
    current_inventory = parameters["tank initial quantity"].astype(np.float64)
    current_trailer_tank = parameters["trailer initial quantity"].astype(np.float64)
    forecast = real_demand
    minutes_below_safety = np.zeros(len(current_inventory))
    minutes_below_0 = np.zeros(len(current_inventory))

    for t in range(parameters["time horizon"]):
        current_inventory -= forecast[:, t]

        for shift in solution:
            trailer = shift["Trailer index"]

            for operation in shift["Operations"]:
                if operation["Arrival time"] == t:
                    location = operation["Location index"]
                    quantity = operation["Quantity"]
                    current_inventory[location] += quantity
                    current_trailer_tank[trailer] -= quantity

        if np.any(current_inventory < 0):
            return True

    return False