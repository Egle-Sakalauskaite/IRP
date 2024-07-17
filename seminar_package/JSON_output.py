import json
from seminar_package import excel_reader
# from . import excel_reader
# import excel_reader
from seminar_package import dummy
# from . import dummy
# import dummy
from sortedcontainers import SortedDict
import numpy as np

precision = (10 ** - 9)

def write_to_JSON(python_dict, filename):
    """
    Takes a list of dictionaries, whereeEach dictionary represents a shift. Look at Canvas for format!
    Afterwards, it generates a JSON file in the desired format.

    @param python_dict: The solution as a list of dicionaries
    @param filename: How to name the file (excluding the extension '.json').
    """
    file_path = "./json_solutions/"
    file_path += filename
    file_path += ".json"
    with open(file_path, 'w') as json_file:
        json.dump(python_dict, json_file, indent=4)


def total_cost(solution, parameter):
    """
    Calculates the total cost of a solution, irrespective of feasibility.
    @param solution: A Python dictionary that has the required JSON format.
    @param parameter: The dictionary containing all the parameters from excel_reader.
    @return the total cost
    """
    driver_cost = parameter.get("driver cost")
    trailer_cost = parameter.get("trailer cost")
    setup_times = parameter.get("setup time")
    travel_times = parameter.get("travel time matrix")
    trailers = range(parameter.get("number trailer"))
    distances = parameter.get("distance matrix")

    total_hours = 0
    total_km = 0

    for shift in solution:
        start_time = shift["Start time"]
        last_operation = shift["Operations"][len(shift["Operations"]) - 1]
        end_time = last_operation["Arrival time"]
        last_location = last_operation["Location index"]
        end_time += setup_times[last_location]
        end_time += travel_times[last_location, 0] # 0 is the base
        total_hours +=  (end_time - start_time)

    for trailer in trailers:
        events_trailer = event_list_trailer(solution, parameter, trailer)
        previous_location = 0
        for time, event in events_trailer.items():
            current_location = event[0]
            total_km += distances[previous_location, current_location]
            previous_location = current_location
        #Each trailer must return to the base one last time. previous_location should be the last visited location
        total_km += distances[previous_location , 0]
    
    return driver_cost * total_hours + trailer_cost * total_km

def additional_info(solution):
    """
    Calculates the total quantity delivered of a solution, number of visits to the source and the number of visits to customers, irrespective of feasibility.
    @param solution: A Python dictionary that has the required JSON format.
    @return the total cost
    """
    n_source_visits = 0
    n_customer_visits = 0
    total_delivered_quantity = 0
    n_shifts = 0
    for shift in solution:
        n_shifts += 1
        for operation in shift["Operations"]:
            if operation["Location index"] == 1:
                n_source_visits += 1
            else:
                n_customer_visits += 1
                total_delivered_quantity += operation["Quantity"]
    return total_delivered_quantity, n_source_visits, n_customer_visits, n_shifts
    
            

def feasibility(solution, parameter, demand):
    """
    Checks feasibility and calculates the number of minutes below safety level in one go.
    @param solution: A Python dictionary that has the required JSON format.
    @param parameter: The dictionary containing all the parameters from excel_reader.
    @param demand: the matrix of either expected demand or simulated demand. Must be in hours!
    @return: a list that keeps track of the minutes of a customer being below safety level
    """
    correct1 = driver_trailer_correct(solution, parameter)
    correct2 = trailer_inventory(solution, parameter)
    minutes_below_safety, correct3 = customer_inventory(solution, parameter, demand)
    correct4 = arrival_time_correctness(solution, parameter)
    correct5 = driver_time_window(solution, parameter)
    correct6 = driver_max_driving(solution, parameter)
    feasible = correct1 and correct2 and correct3 and correct4 and correct5 and correct6
    return minutes_below_safety, feasible


def driver_time_window(solution, parameter):
    """
    Checks whether the driver starts at and returns to the base within time window.
    @param solution: A Python dictionary that has the required JSON format.
    @param parameter: The dictionary containing all the parameters from excel_reader.
    """
    time_windows = parameter.get("time windows")
    setup_times = parameter.get("setup time")
    travel_times = parameter.get("travel time matrix")
    correct = True
    for shift in solution:
        driver = shift["Driver index"]
        start_time = shift["Start time"]
        last_operation = shift["Operations"][len(shift["Operations"]) - 1]
        end_time = last_operation["Arrival time"]
        last_location = last_operation["Location index"]
        end_time += setup_times[last_location]
        end_time += travel_times[last_location, 0] # 0 is the base
        fits_in_window = False # Becomes true if it fits in one window
        for row in time_windows:
            if row[0] == driver:
                if row[1] <= start_time and row[2] >= end_time:
                    fits_in_window = True
                    break
        if not fits_in_window:
            # print(f"!!! Driver {driver} starting at time {start_time} does not fit in a time window. !!!")
            correct = False
    # if correct:
    #     print("All shifts fit within the drivers' time windows.")
    return correct


def driver_max_driving(solution, parameter):
    """
    Checks whether the driver does not drive longer than the maximum duration (Only relevant for TinyLand).
    @param solution: A Python dictionary that has the required JSON format.
    @param parameter: The dictionary containing all the parameters from excel_reader.
    """
    setup_times = parameter.get("setup time")
    travel_times = parameter.get("travel time matrix")
    correct = True
    for shift in solution:
        driver = shift["Driver index"]
        start_time = shift["Start time"]
        last_operation = shift["Operations"][len(shift["Operations"]) - 1]
        end_time = last_operation["Arrival time"]
        last_location = last_operation["Location index"]
        end_time += setup_times[last_location]
        end_time += travel_times[last_location, 0] # 0 is the base
        duration = end_time - start_time
        if duration > parameter.get("max driving"):
            # print(f"!!! Driver {driver} starting at {start_time} drives for longer than the max duration. !!!")
            correct = False
    # if correct:
    #     print("No driver drives longer than the maximum duration.")
    return correct


def driver_trailer_correct(solution, parameter):
    """
    Checks whether each driver drives the correct trailer.
    @param solution: A Python dictionary that has the required JSON format.
    @param parameter: The dictionary containing all the parameters from excel_reader.
    """
    driver_trailer_binary = parameter.get("driver trailer matrix")
    #A scuffed way to get a simple pair of driver-trailers from the binary matrix
    driver_trailer_pair = {}
    correct = True
    for index, value in np.ndenumerate(driver_trailer_binary):
        if value == 1:
            driver_trailer_pair[index[0]] = index[1]   
    for shift in solution:
        driver = shift["Driver index"]
        trailer = shift["Trailer index"]
        if not driver_trailer_pair[driver] == trailer:
            # print(f"!!! Driver {driver} is driving Trailer {trailer} wrongly. !!!")
            correct = False
    # if correct:
    #     print("All drivers are driving the correct trailer")
    return correct


def trailer_inventory(solution, parameter):
    """
    Checks whether trailer inventory is always nonnegative and below capacity.
    @param solution: A Python dictionary that has the required JSON format.
    @param parameter: The dictionary containing all the parameters from excel_reader.
    """
    trailer_capacities = parameter.get("trailer capacity").astype(np.float64)
    trailer_initial = parameter.get("trailer initial quantity").astype(np.float64)
    trailers = range(parameter.get("number trailer"))
    correct = True
    for trailer in trailers:
        quantity = trailer_initial[trailer]
        events_list = event_list_trailer(solution, parameter, trailer)
        for time, event in events_list.items(): # time is the key of the dictionary
            quantity -= event[1]
            if quantity < 0 and 0 - quantity > precision:
                # print(f"!!! Trailer {trailer} reaches below 0 inventory at time {time} with quantity {quantity} !!!")
                correct = False
            if quantity > trailer_capacities[trailer] and quantity - trailer_capacities[trailer] > precision:
                # print(f"!!! Trailer {trailer} reaches above capacity at time {time} with {quantity - trailer_capacities[trailer]} excess !!!")
                correct = False
    # if correct:
    #     print("The inventory of the trailers stay within bounds.")
    return correct


def customer_inventory(solution, parameter, demand):
    """
    Checks whether customer inventory is always above safety level and below capacity.
    @param solution: A Python dictionary that has the required JSON format.
    @param parameter: The dictionary containing all the parameters from excel_reader.
    @param demand: The matrix of either expected demand or simulated demand.
    @return: A list of number of minutes that the customer is below safety level
    """
    customers = range(2, parameter.get("number customer") + 2) # Exclude 0 Base and 1 Source
    tank_capacities = parameter.get("tank capacity").astype(np.float64)
    tank_initial = parameter.get("tank initial quantity").astype(np.float64)
    safety_levels = parameter.get("safety level")
    minutes_below_safety = np.zeros(len(safety_levels))
    correct = True
    for customer in customers:
        inventory = tank_initial[customer]
        safety = safety_levels[customer]
        capacity = tank_capacities[customer]
        events_list = event_list_customer(solution, parameter, customer, demand)
        below_safety = False #Assume every customer is above safety at the start (Duh)
        start_below = 0 # The moment it falls below safety
        for time, quantity in events_list.items(): #time is key, quantity is value in dictionary
            inventory += quantity
            if inventory < safety:
                if not below_safety: #Only print the first time it goes below
                    # print(f"! Customer {customer} reaches below safety level at time {time}. !")
                    start_below = time
                below_safety = True
            else:
                if below_safety: # Only print when return back above safety
                    # print(f"! Customer {customer} is above safety level again at time {time}. !")
                    minutes_below_safety[customer] += (time - start_below)
                below_safety = False
            if time == parameter.get("time horizon") and below_safety:
                minutes_below_safety[customer] += (time - start_below)
            if inventory > capacity and inventory - capacity > precision:
                # print(f"!!! Customer {customer} reaches above capacity at time {time} with {inventory - capacity} excess. !!!")
                correct = False
            if inventory < 0:
                # print(f"!!! Customer {customer} reaches below 0 inventory at time {time}. !!!")
                correct = False
    # if correct:
    #     print("The customers' inventories always stay within bounds.")
    return minutes_below_safety, correct


def arrival_time_correctness(solution, parameter):
    """
    Checks whether the list of operations are consistent in terms of travel time. It is allowed to arrive later than expected, but not earlier.
    @param solution: A Python dictionary that has the required JSON format.
    @param parameter: The dictionary containing all the parameters from excel_reader.
    """
    trailers = range(parameter.get("number trailer"))
    setup_times = parameter.get("setup time")
    travel_times = parameter.get("travel time matrix")
    correct = True
    for trailer in trailers:
        events_trailer = event_list_trailer(solution, parameter, trailer)
        previous_time = 0
        previous_location = 0
        for time, event in events_trailer.items():
            current_time = time
            current_location = event[0]
            time_elapsed = current_time - previous_time
            if previous_location == current_location: # The reason for this if statement is that the diagonals are 1.
                travel_time = 0
            else:
                travel_time = travel_times[previous_location, current_location]
            time_needed = setup_times[previous_location] + travel_time
             #The setup time of the previous location, as you first arrive and then set up.
            if time_needed > time_elapsed:
                # print(f"!!! Trailer {trailer} arrives earlier than possible at time {current_time} in location {current_location} !!!")
                correct = False
    # if correct:
    #     print("All trailers have consistent arrival times.")
    return correct


def event_list_trailer(solution, parameter, trailer):
    """
    Creats an events list in the form of an SortedDict for the trailers, ordered by the key (time). The values are pairs (location, quantity). This function also checks for simultaneity of trailers.
    @param solution: A Python dictionary that has the required JSON format.
    @param parameter: The dictionary containing all the parameters from excel_reader.
    @return The events list in the form of an SortedDict
    """
    setup_times = parameter.get("setup time")
    travel_times = parameter.get("travel time matrix")
    event_list = SortedDict()
    end_shift = float('-inf')
    for shift in solution:
        if len(shift["Operations"]) == 0:
            continue
        if shift["Trailer index"] == trailer:
            start_shift = shift["Start time"]
            if end_shift > start_shift: #end_shift is the end of the previous shift
                raise AssertionError(f"!!! Trailer {trailer} has two simultaneous shifts around {start_shift}. !!!")
            event_list[start_shift] = (0,0) #Location 0 is the base
            last_operation = shift["Operations"][len(shift["Operations"]) - 1]
            end_shift = last_operation["Arrival time"]
            last_location = last_operation["Location index"]
            end_shift += setup_times[last_location]
            end_shift += travel_times[last_location, 0] # 0 is the base
            for operation in shift["Operations"]:
                time = operation["Arrival time"]
                if time in event_list:
                    raise AssertionError(f"!!! Trailer {trailer} has two simultaneous operations at time {time}. First is at {event_list[time][0]}, second is at {operation['Location index']} !!!")
                else:
                    location = operation["Location index"]
                    quantity = operation["Quantity"]
                    event_list[time] = (location, quantity)
    return event_list
    

def event_list_customer(solution, parameter, customer, demand):
    """
    Creats an events list in the form of an SortedDict for the customers, ordered by the key (time). The values are quantities.
    @param solution: A Python dictionary that has the required JSON format.
    @param parameter: The dictionary containing all the parameters from excel_reader.
    @param demand: The matrix of either the expected demand or simulated demand.
    @return The events list in the form of an SortedDict
    """
    events_customer = SortedDict()
    
    #Add all refills
    trailers = range(parameter.get("number trailer"))
    for trailer in trailers:
        events_trailer = event_list_trailer(solution, parameter, trailer)
        for time, event in events_trailer.items():
            location = event[0]
            quantity = event[1]
            if location == customer:
                events_customer[time] = quantity
    #Add all forecasts (negative)
    forecasts_customer = demand[customer]
    for hour, forecast in enumerate(forecasts_customer):
        minute = 60*(hour + 1)
        if minute in events_customer:
            events_customer[minute] -= forecast
        else:
            events_customer[minute] = -forecast
        
    return events_customer


#Main to test
# dummy_solution = dummy.generate_solution()
# dummy_param = dummy.parameter_generator()
# total_cost = total_cost(dummy_solution, dummy_param)
# feasibility(dummy_solution, dummy_param)
# print(total_cost)
#End of main
