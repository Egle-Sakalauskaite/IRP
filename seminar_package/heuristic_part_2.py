import numpy as np
from collections import deque
import copy
from seminar_package import heuristics_helper_part_2 as hh
from seminar_package import excel_reader
# import JSON_output
# import heuristics_helper_part_2 as hh
# import excel_reader
# from . import JSON_output
# from . import heuristics_helper_part_2 as hh
# from . import excel_reader
import time

def heuristic_main(parameters, upper_bound_forecast, SAFETY_TIME, MINIMUM_DELIVERY_QUANTITY_PROPORTION):
    # initialization
    # parameters = excel_reader.parameter_reader(FILE)
    time_periods = hh.periods_in_order(parameters["time windows"])
    "consumption forecast should be replaced with upper bounds estimates"
    # upper_bound_forecast = excel_reader.demand_upperbound_minutes(FILE, ALPHA)
    "use upper bounds to estimate consumption"
    initial_inventory = hh.consume(parameters["tank initial quantity"], range(parameters["number customer"] + 2), upper_bound_forecast, 0, time_periods[0, 0])
    initial_trailer_tank = parameters["trailer initial quantity"].astype(np.float64)
    trailer_capacity = parameters["trailer capacity"].astype(np.float64)
    travel_time = parameters["travel time matrix"]
    setup_time = parameters["setup time"]
    safety = parameters["safety level"]
    solution = []
    cost = None
    minutes_below_safety = None

    for period_idx in range(time_periods.shape[0]):
    # for period_idx in range(8):
        # print(f"time period: {time_periods[period_idx,:]}")
        # initial inventory: the inventory level at the beginning of the shift
        # NOT the inventory at the beginning of the time horizon

        # start_time: start of the shift
        # shift_end_time: end of the shift
        # end_time: beginning of the next shift
        start_time: int = time_periods[period_idx, 0]
        shift_end_time: int = time_periods[period_idx, 1]
        end_time: int

        if period_idx < time_periods.shape[0] - 1:
            end_time = time_periods[period_idx + 1, 0]
        else:
            end_time = parameters["time horizon"]

        # print(f"next iteration starts at: {end_time}")
        # all the clients queued according to their safety level reaching time: urgency(customer id, safety level reach time)
        urgency: deque[tuple[int, int]]
        # a set of customers which will reach their safety level before the end of the time horizon
        urgency, total_demands = hh.ranking_urgency(safety, upper_bound_forecast, initial_inventory.copy(), start_time)
        # print(f"urgency: {urgency}")
        # print(f"total demands: {total_demands}")

        # filtered customers: urgent and empty enough filteresd customers
        # filtered_customers{customer_idx: min_available_space}
        filtered_customers: dict[int, float]
        filtered_customers = hh.filter_customers(parameters, urgency, total_demands, initial_inventory, end_time, SAFETY_TIME)
        # print(f"should be visited: {filtered_customers}")

        # no customers that cannot be served in the next shift: do not schedule any trips this shift
        if len(filtered_customers) == 0:
            "use upper bounds to estimate consumption"
            initial_inventory = hh.consume(initial_inventory, range(parameters["number customer"] + 2), upper_bound_forecast, start_time, end_time)
            # print("Skipping iteration")
            continue
        
        # available trailers: [(trailer idx, driver idx, trailer capacity)]
        # delivery_time_windows{customer: (earliest, latest, minimal quantity)}
        # clusters{cluster idx: [customer idx]}
        # clusters_drivers{cluster idx: driver idx}
        available_drivers: set[int] = hh.available_drivers(parameters["time windows binary"], start_time, shift_end_time)
        available_trailers: list[tuple[int, int, float]] = hh.available_trailers(parameters, available_drivers)
        # print(f"trailers drivers and capacities: {available_trailers}")
        n_clusters: int = len(available_drivers)
        clusters: dict[int: list[int]] = hh.clustering_customers_travel_cost(parameters, n_clusters, np.array(list(filtered_customers.keys())))
        # print(f"clustering {n_clusters}:")
        # print(clusters)
        clusters_drivers: dict[int: int] = hh.cluster_driver_assignment(clusters, filtered_customers, available_trailers)
        period_result: dict[int: dict] = {}
        state: dict[int: dict] = {}
        routed: set[int] = set()

        # initialize the solution
        for cluster in clusters:
            driver = clusters_drivers[cluster]
            trailer = parameters["driver trailer"][driver][1]
            customers = clusters[cluster]

            state[cluster] = {"current tank level": initial_trailer_tank[trailer],
                            "current cluster inventories": initial_inventory,
                            "customers": customers,
                            "shift length": 0,
                            "shift cost": 0,
                            "current location": 0,
                            "current time": start_time}
            
            period_result[cluster] = {"Driver index": driver,
                                    "Trailer index": trailer,
                                    "Start time": start_time, 
                                    "Operations": []}
        
        # insert the customers into the routes
        while len(routed) < len(filtered_customers):
            if len(urgency) == 0:
                break

            next_to_insert = urgency.popleft()[0]
            # print(f"Next location to insert: {next_to_insert}")
            cluster = hh.identify_cluster(clusters, next_to_insert)
            # print(f"cluster: {cluster}")

            if cluster is None:
                # print(f"customer {next_to_insert} is not in any cluster")
                continue

            customers_in_cluster = clusters[cluster]
            trailer = period_result[cluster]["Trailer index"]
            minimal_delivery_quantity = hh.minimal_delivery_quantity(parameters, initial_inventory, trailer, next_to_insert)
            # print(f"minimal delivery quantity: {minimal_delivery_quantity}")

            # Before attemping to insert, consider first a trip to the source
            if state[cluster]["current tank level"] < minimal_delivery_quantity*MINIMUM_DELIVERY_QUANTITY_PROPORTION:
                # print("scheduling a trip to the source...")
                arrival_time = state[cluster]["current time"] + travel_time[state[cluster]["current location"], 1]
                quantity = trailer_capacity[trailer] - state[cluster]["current tank level"]

                operation = {"Location index": 1,
                            "Arrival time": arrival_time,
                            "Quantity": -quantity}
                
                # print(f"Trailer inventory before the operation: {state[cluster]["current tank level"]}")
                period_result_copied = copy.deepcopy(period_result[cluster])
                period_result_copied = hh.insert_operation(parameters, period_result_copied, operation, None)

                try:
                    state[cluster] = hh.update_state(parameters, upper_bound_forecast, initial_trailer_tank[trailer], initial_inventory, customers_in_cluster, start_time, period_result_copied)
                except ValueError as e:
                    # print("=====INFEASIBLE!=====")
                    # print(str(e))
                    continue
                else:
                    period_result[cluster] = period_result_copied
                    state[cluster] = hh.update_state(parameters, upper_bound_forecast, initial_trailer_tank[trailer], initial_inventory, customers_in_cluster, start_time, period_result[cluster])
                    # print(f"operation's quantity: {operation["Quantity"]}")
                    # print(f"Trailer inventory after the operation: {state[cluster]["current tank level"]}")


            best_feasible_result = None
            best_cost = 9999999999999999
            # print(f"current operations: {period_result[cluster]["Operations"]}")

            # try inserting into each position, select a feasible position with the least cost
            for insert_position in range(len(period_result[cluster]["Operations"]) + 1):
                # print(f"Attempting to insert at: {insert_position}")

                period_result_copied = copy.deepcopy(period_result[cluster])

                if len(period_result_copied["Operations"]) == 0 or insert_position == 0:
                    previous_location = 0
                    previous_arrival_time = start_time
                else:
                    previous_location = period_result_copied["Operations"][insert_position - 1]["Location index"]
                    previous_arrival_time = period_result_copied["Operations"][insert_position - 1]["Arrival time"]
                
                arrival_time = previous_arrival_time + setup_time[previous_location] + travel_time[previous_location, next_to_insert]
                "use upper bounds to estimate consumption"
                current_inventory = hh.consume(initial_inventory.copy(), next_to_insert, upper_bound_forecast, start_time, arrival_time - 1)
                delivery_quantity = hh.delivery_quantity(parameters, state[cluster], current_inventory, next_to_insert)
                
                if delivery_quantity == 0:
                    continue

                operation = {"Location index": next_to_insert,
                            "Arrival time": arrival_time,
                            "Quantity": delivery_quantity}
                
                period_result_copied = hh.insert_operation(parameters, period_result_copied, operation, insert_position)
                # print(f"Trailer inventory before operation: {state[cluster]["current tank level"]}")

                try:
                    state[cluster] = hh.update_state(parameters, upper_bound_forecast, initial_trailer_tank[trailer], initial_inventory, customers_in_cluster, start_time, period_result_copied)
                except ValueError as e:
                    # print("=====INFEASIBLE!=====")
                    # print(str(e))
                    pass
                else:
                    if state[cluster]["shift cost"] < best_cost:
                        best_feasible_result = period_result_copied
                        best_cost = state[cluster]["shift cost"]

            if best_feasible_result is not None:
                period_result[cluster] = best_feasible_result
                state[cluster] = hh.update_state(parameters, upper_bound_forecast, initial_trailer_tank[trailer], initial_inventory, customers_in_cluster, start_time, period_result[cluster])
                routed.add(next_to_insert)
                # print(f"operation's quantity: {operation["Quantity"]}")
                # print(f"Trailer inventory after operation: {state[cluster]["current tank level"]}")
            # else:
                # print(f"No feasible position found... location {next_to_insert} cannot be added to this shift")

            # print(f"current operations: {period_result[cluster]["Operations"]}")


        # inventory of locations that did not get considered in either of the clusters
        # must be reduced according to the consumption for (start_time, end_time)
        initial_trailer_tank = hh.update_initial_trailer(initial_trailer_tank.copy(), period_result)
        initial_inventory = hh.consume(initial_inventory.copy(), range(parameters["number customer"] + 2), upper_bound_forecast, start_time, end_time)
        initial_inventory = hh.add_initial_customers(initial_inventory.copy(), period_result)

        for clust, shift in period_result.items():
            if len(shift["Operations"]) > 0:
                solution.append(shift)

    # if(len(routed) < len(filtered_customers)):
        # print(F"CUSTOMERS {set(filtered_customers).difference(routed)} REMAIN UNROUTED AT THE END")

    # print(solution)

    # real_demand = excel_reader.normal_simulator(FILE)

    # try:
    #     cost, minutes_below_safety = hh.check_feasibility(solution, FILE, real_demand)
    #     print(f"OBJECTIVE VALUE: {cost}")
    # except ValueError as e:
    #     print(str(e))

    return solution

# data instances
# file_path_tiny = ".\\data\\Tinyland.xlsx"
# file_path_middle = ".\\data\\Middleland.xlsx"
# file_path_large = ".\\data\\Largeland.xlsx"
#
# # hyper parameters
# FILE = file_path_tiny
# ALPHA = 0.6
# SAFETY_TIME = 0
# SAFETY_QUANTITY = 1
# MINIMUM_DELIVERY_QUANTITY_PROPORTION = 0.8

# start_time = time.time()
# objective, safety_violation, solution = heuristic_main(FILE, ALPHA, SAFETY_TIME, SAFETY_QUANTITY, MINIMUM_DELIVERY_QUANTITY_PROPORTION)
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time} seconds")
# print(objective)

# n_shifts = len(solution)
# n_customer_visits = 0
# n_source_visits = 0
# total_delivered_quantity = 0

# for shift in solution:
#     for operation in shift["Operations"]:
#         if operation["Location index"] == 1:
#             n_source_visits += 1
#         else:
#             n_customer_visits += 1
#             total_delivered_quantity += operation["Quantity"]


# print(f"shifts: {n_shifts}")
# print(f"customer visits: {n_customer_visits}")
# print(f"source visits: {n_source_visits}")
# print(f"total quantity delivered: {total_delivered_quantity}")

# def convert_int64(integer):
#     if isinstance(integer, np.int64):
#         return int(integer)
#     return integer

# for shift in solution:
#     for key, value in shift.items():
#         if key == "Operations":
#             for operation in shift["Operations"]:
#                 for key2, value2 in operation.items():
#                     operation[key2] = convert_int64(value2)
#         else:
#             shift[key] = convert_int64(value)

# JSON_output.write_to_JSON(solution, "heuristic_tiny_2")

# objectives = []

# for i in range(50, 61):
#     for j in range(0, 2500, 300):
#         objective, safety_violations, solution = heuristic_main(FILE, i / 100, j, SAFETY_QUANTITY, MINIMUM_DELIVERY_QUANTITY_PROPORTION)
#         if objective is not None:
#             objectives.append({"configuration": (i / 100, j), "objective value": objective, "safety violated min": np.sum(safety_violations)})
#
# print("OBJECTOVES:")
# print(objectives)


# Alternative names: heuristic_middle heuristic_large
# parameters = excel_reader.parameter_reader(FILE)
# JSON_output.feasibility(solution, parameters)
# JSON_output.total_cost(solution, parameters)A