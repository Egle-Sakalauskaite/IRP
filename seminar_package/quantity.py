import copy
import numpy as np


def simple_quantities_whole(wrong_solution, parameters, real_demand):
    """
    Takes an infeasible solution and readjusts the quantities delivered, while keeping the routes the same. The adjustment is simple
    as we simply decrease the quantity if any capacity is violated. The solution is the entire time horizon.
    @param wrong_solution: the python list of dictionaries that is generated from heuristic
    @param parameters: the dictionary containing all parameters from excel_reader
    @param real_demand: the simulated demand per minute.
    @return: the solution in the same format with adjusted quantities.
    """
    solution = copy.deepcopy(wrong_solution)
    current_inventory = parameters["tank initial quantity"].astype(np.float64).copy()
    current_trailer_tank = parameters["trailer initial quantity"].astype(np.float64).copy()
    capacity_customer = parameters["tank capacity"].astype(np.float64)
    capacity_trailer = parameters["trailer capacity"].astype(np.float64)

    for t in range(parameters["time horizon"]):
        current_inventory -= real_demand[:, t]

        for shift in solution:
            trailer = shift["Trailer index"]
            available_space_trailer = capacity_trailer[trailer] - current_trailer_tank[trailer]

            for operation in shift["Operations"]:                
                if operation["Arrival time"] == t:
                    location = operation["Location index"]
                    quantity = operation["Quantity"]
                    available_space_location = capacity_customer[location] - current_inventory[location]

                    if location == 1:
                        quantity = -available_space_trailer
                        operation["Quantity"] = quantity
                        current_trailer_tank[trailer] -= quantity

                    if location >= 2:
                        if quantity > available_space_location:
                            # print(f"Decreased quantity because of customer capacity at customer {location} at time {t}")
                            quantity = available_space_location
                        operation["Quantity"] = quantity
                        current_inventory[location] += quantity
                        current_trailer_tank[trailer] -= quantity
        
    return solution

def simple_quantities_daily(partial_solution, parameters, current_inventory, current_trailer_tank, real_demand, start_time, end_time):
    """
    Takes a part of a wrong solution and readjusts the quantities delivered, while keeping the routes the same. 
    @param partial_solution: part of the wrong solution that spans one day. 
    @param parameters: the dictionary containing all parameters from excel_reader
    @param current_inventory: The current inventory of customers at the start of the partial solution.
    @param current_trailer_tank: The current tank level of trailers at the start of the partial solution
    @param real_demand: the simulated demand per minute.
    @param start_time: The start of the day. Mainly used to determine the demand subtraction.
    @param end_time: The end of the day. Mainly used to determine the demand subtraction.
    @return solution: the solution in the same format with adjusted quantities.
    @return current_inventory: the inventory levels of customers after adjusting the quantities.
    @return current_trailer_tank: the tank level of trailers after adjusting the quantities.
    """
    solution = copy.deepcopy(partial_solution)
    current_inventory = current_inventory.astype(np.float64).copy()
    current_trailer_tank = current_trailer_tank.astype(np.float64).copy()
    capacity_customer = parameters["tank capacity"].astype(np.float64)
    capacity_trailer = parameters["trailer capacity"].astype(np.float64)

    for t in range(start_time, end_time):
        current_inventory -= real_demand[:, t]

        for shift in solution:
            trailer = shift["Trailer index"]
            available_space_trailer = capacity_trailer[trailer] - current_trailer_tank[trailer]

            for operation in shift["Operations"]:
                if operation["Arrival time"] == t:
                    location = operation["Location index"]
                    quantity = operation["Quantity"]
                    available_space_location = capacity_customer[location] - current_inventory[location]

                    if location == 1:
                        quantity = -available_space_trailer
                        # print("Decreased quantity at source because of trailer capacity")
                        operation["Quantity"] = quantity
                        current_trailer_tank[trailer] -= quantity

                    if location >= 2:
                        if quantity > available_space_location:
                            # print(f"Decreased quantity because of customer capacity at customer {location} at time {t}")
                            quantity = available_space_location

                        operation["Quantity"] = quantity
                        current_inventory[location] += quantity
                        current_trailer_tank[trailer] -= quantity
        
    return solution, current_inventory, current_trailer_tank


def empty_quantities_whole(wrong_solution, parameters, real_demand):
    """
    Takes an infeasible solution and readjusts the quantities delivered, while keeping the routes the same. It tries to empty the trailer as much as possible at the end of the horizon and right before visiting a source. The solution is the entire time horizon.
    @param wrong_solution: the python list of dictionaries that is generated from heuristic
    @param parameters: the dictionary containing all parameters from excel_reader
    @param real_demand: the simulated demand per minute.
    @return: the solution in the same format with adjusted quantities.
    """
    solution = copy.deepcopy(wrong_solution)
    current_inventory = parameters["tank initial quantity"].astype(np.float64).copy()
    current_trailer_tank = parameters["trailer initial quantity"].astype(np.float64).copy()
    capacity_customer = parameters["tank capacity"].astype(np.float64)
    capacity_trailer = parameters["trailer capacity"].astype(np.float64)
    trailers_final_operation = []
    trailers_prev_operation = []
    prev_op_available_space = np.zeros(len(capacity_trailer)) # Keeps track how much available space there was in a trailer's previous operation.
    saved_quantities = np.zeros(len(capacity_trailer))

    for trailer in range(len(capacity_trailer)):
        trailers_prev_operation.append({}) 
        for shift in solution[::-1]:
            if shift["Trailer index"] == trailer:
                trailers_final_operation.append(shift["Operations"][-1])
                break

    for t in range(parameters["time horizon"]):
        current_inventory -= real_demand[:, t]

        for shift in solution:
            trailer = shift["Trailer index"]
            available_space_trailer = capacity_trailer[trailer] - current_trailer_tank[trailer]

            for operation in shift["Operations"]:                
                if operation["Arrival time"] == t:
                    location = operation["Location index"]
                    quantity = operation["Quantity"]
                    available_space_location = capacity_customer[location] - current_inventory[location]

                    if location == 1:
                        #Adds more quantity to the previous operation
                        if not trailers_prev_operation[trailer] == {}:
                            prev_location = trailers_prev_operation[trailer]["Location index"] 
                            to_add = min(current_trailer_tank[trailer], prev_op_available_space[trailer])
                            # if to_add > 0:
                                # print(f"Increased quantity right before visiting source at time {trailers_prev_operation[trailer]['Arrival time']}")
                            trailers_prev_operation[trailer]["Quantity"] += to_add
                            current_inventory[prev_location] += to_add
                            current_trailer_tank[trailer] -= to_add
                            available_space_trailer = capacity_trailer[trailer] - current_trailer_tank[trailer]
                        quantity = -available_space_trailer
                        operation["Quantity"] = quantity
                        # if -quantity > available_space_trailer:
                        #     print("Decreased quantity at source because of trailer capacity")
                        current_trailer_tank[trailer] -= quantity
                        saved_quantities[trailer] = 0

                    if location >= 2:
                        #Check whether the operation is the very last operation that trailer will do.
                        if (location == trailers_final_operation[trailer]["Location index"] and t == trailers_final_operation[trailer]["Arrival time"]):
                            quantity = min(available_space_location, current_trailer_tank[trailer])
                            # print(f"Trailer {trailer} dumps quantity {quantity} at the end.")
                        elif quantity > available_space_location:
                            # print(f"Decreased quantity because of customer capacity at customer {location} at time {t}")
                            saved_quantities[trailer] += quantity - available_space_location
                            quantity = available_space_location
                        elif saved_quantities[trailer] > 0 and quantity < available_space_location:
                            # print(f"Increased quantity because of saved quantity at customer {location} at time {t}")
                            if quantity + saved_quantities[trailer] <= available_space_location:
                                quantity += saved_quantities[trailer]
                                saved_quantities[trailer] = 0
                            else:
                                saved_quantities[trailer] = quantity + saved_quantities[trailer] - available_space_location
                                quantity = available_space_location
                        operation["Quantity"] = quantity
                        current_inventory[location] += quantity
                        current_trailer_tank[trailer] -= quantity

                    # Update the last operation a trailer has done.
                    trailers_prev_operation[trailer] = operation
                    prev_op_available_space[trailer] = capacity_customer[location] - current_inventory[location]
    return solution

def empty_quantities_daily(partial_solution, parameters, current_inventory, current_trailer_tank, real_demand, start_time, end_time):
    """
    Takes a part of a wrong solution and readjusts the quantities delivered, while keeping the routes the same. It tries to empty the trailer as much as possible right before visiting a source.
    @param partial_solution: part of the wrong solution that spans one day. 
    @param parameters: the dictionary containing all parameters from excel_reader
    @param current_inventory: The current inventory of customers at the start of the partial solution.
    @param current_trailer_tank: The current tank level of trailers at the start of the partial solution
    @param real_demand: the simulated demand per minute.
    @param start_time: The start of the day. Mainly used to determine the demand subtraction.
    @param end_time: The end of the day. Mainly used to determine the demand subtraction.
    @return solution: the solution in the same format with adjusted quantities.
    @return current_inventory: the inventory levels of customers after adjusting the quantities.
    @return current_trailer_tank: the tank level of trailers after adjusting the quantities.
    """
    solution = copy.deepcopy(partial_solution)
    current_inventory = current_inventory.astype(np.float64).copy()
    current_trailer_tank = current_trailer_tank.astype(np.float64).copy()
    capacity_customer = parameters["tank capacity"].astype(np.float64)
    capacity_trailer = parameters["trailer capacity"].astype(np.float64)
    trailers_prev_operation = []
    prev_op_available_space = np.zeros(len(capacity_trailer)) 
    saved_quantities = np.zeros(len(capacity_trailer))

    for trailer in range(len(capacity_trailer)):
        trailers_prev_operation.append({})

    for t in range(start_time, end_time):
        current_inventory -= real_demand[:, t]

        for shift in solution:
            trailer = shift["Trailer index"]
            available_space_trailer = capacity_trailer[trailer] - current_trailer_tank[trailer]

            for operation in shift["Operations"]:
                if operation["Arrival time"] == t:
                    location = operation["Location index"]
                    quantity = operation["Quantity"]
                    available_space_location = capacity_customer[location] - current_inventory[location]

                    if location == 1:
                        #Adds more quantity to the previous operation
                        if not trailers_prev_operation[trailer] == {}:
                            prev_location = trailers_prev_operation[trailer]["Location index"] 
                            to_add = min(current_trailer_tank[trailer], prev_op_available_space[trailer])
                            # if to_add > 0:
                            #     print(f"Increased quantity right before visiting source at time {trailers_prev_operation[trailer]['Arrival time']}")
                            trailers_prev_operation[trailer]["Quantity"] += to_add
                            current_inventory[prev_location] += to_add
                            current_trailer_tank[trailer] -= to_add
                            available_space_trailer = capacity_trailer[trailer] - current_trailer_tank[trailer]

                        # if -quantity > available_space_trailer:
                        #     print("Decreased quantity at source because of trailer capacity")
                        quantity = -available_space_trailer
                        operation["Quantity"] = quantity
                        current_trailer_tank[trailer] -= quantity
                        saved_quantities[trailer] = 0

                    if location >= 2:
                        if quantity > available_space_location:
                            # print(f"Decreased quantity because of customer capacity at customer {location} at time {t}")
                            saved_quantities[trailer] += quantity - available_space_location
                            quantity = available_space_location

                        elif saved_quantities[trailer] > 0 and quantity < available_space_location:
                            # print(f"Increased quantity because of saved quantity at customer {location} at time {t}")
                            if quantity + saved_quantities[trailer] <= available_space_location:
                                quantity += saved_quantities[trailer]
                                saved_quantities[trailer] = 0
                            else:
                                saved_quantities[trailer] = quantity + saved_quantities[trailer] - available_space_location
                                quantity = available_space_location

                        operation["Quantity"] = quantity
                        current_inventory[location] += quantity
                        current_trailer_tank[trailer] -= quantity
                    # Update the last operation a trailer has done.
                    trailers_prev_operation[trailer] = operation
                    prev_op_available_space[trailer] = capacity_customer[location] - current_inventory[location]

        
    return solution, current_inventory, current_trailer_tank