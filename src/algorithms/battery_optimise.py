import pandas as pd
import numpy as np

import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

from pyomo.environ import (
    ConcreteModel, 
    Set, 
    Param, 
    Var, 
    Constraint, 
    Objective, 
    maximize, 
    SolverFactory, 
    Any
)
from pyutilib.services import register_executable, registered_executable
register_executable(name='glpsol')

def battery_optimisation(datetime, spot_price, max_battery_capacity=1, initial_capacity=0, max_battery_power=0.25, efficiency=0.9, include_revenue=True, solver: str='glpk'):
    """
    Determine the optimal charge and discharge behavior of a battery.
    Assuming pure foresight of future spot prices over every 
    15 mins period to maximise the revenue.
    PS: Assuming no degradation to the battery over the timeline and battery cannot
        charge and discharge concurrently.
    ----------
    Parameters
    ----------
    datetime        : a list of time stamp
    spot_price      : a list of spot price of the corresponding time stamp
    max_battery_capacity : the maximum capacity of the battery
    initial_capacity  : the initial capacity of the battery
    max_battery_power : the maximum power of the battery
    efficiency      : the efficiency of the battery
    include_revenue : a boolean indicates if return results should include revenue calculation
    solver          : the name of the desire linear programming solver (eg. 'glpk', 'mosek', 'gurobi')

    Returns
    ----------
    A dataframe that contains battery's opening capacity for each 15 mins period, spot price
    of each 15 mins period and battery's raw power for each 15 mins priod
    """
    # Battery's technical specification
    MIN_BATTERY_CAPACITY = 0
    MAX_BATTERY_CAPACITY = max_battery_capacity
    MAX_BATTERY_POWER = max_battery_power
    INITIAL_CAPACITY = initial_capacity # Default initial capacity will assume to be 0
    EFFICIENCY = efficiency
    MLF = 1 # Marginal Loss Factor
    
    df = pd.DataFrame({'datetime': datetime, 'spot_price': spot_price}).reset_index(drop=True)
    df['period'] = df.index
    initial_period = 0
    final_period = df.index[-1]
    
    # Define model and solver
    battery = ConcreteModel()
    opt = SolverFactory(solver)

    # defining components of the objective model
    # battery parameters
    battery.Period = Set(initialize=list(df.period), ordered=True)
    battery.Price = Param(initialize=list(df.spot_price), within=Any)

    # battery varaibles
    battery.Capacity = Var(battery.Period, bounds=(MIN_BATTERY_CAPACITY, MAX_BATTERY_CAPACITY))
    battery.Charge_power = Var(battery.Period, bounds=(0, MAX_BATTERY_POWER))
    battery.Discharge_power = Var(battery.Period, bounds=(0, MAX_BATTERY_POWER))

    # Set constraints for the battery
    # Defining the battery objective (function to be maximise)
    def maximise_profit(battery):
        rev = sum(df.spot_price[i] * (battery.Discharge_power[i] / 4 * EFFICIENCY) * MLF for i in battery.Period)
        cost = sum(df.spot_price[i] * (battery.Charge_power[i] / 4) / MLF for i in battery.Period)
        return rev - cost

    # Make sure the battery does not charge above the limit
    def over_charge(battery, i):
        return battery.Charge_power[i] <= (MAX_BATTERY_CAPACITY - battery.Capacity[i]) * 4 / EFFICIENCY

    # Make sure the battery discharge the amount it actually has
    def over_discharge(battery, i):
        return battery.Discharge_power[i] <= battery.Capacity[i] * 4

    # Make sure the battery do not discharge when price are not positive
    def negative_discharge(battery, i):
        # if the spot price is not positive, suppress discharge
        if battery.Price.extract_values_sparse()[None][i] <= 0:
            return battery.Discharge_power[i] == 0

        # otherwise skip the current constraint    
        return Constraint.Skip

    # Defining capacity rule for the battery
    def capacity_constraint(battery, i):
        # Assigning battery's starting capacity at the beginning
        if i == battery.Period.first():
            return battery.Capacity[i] == INITIAL_CAPACITY
        # if not update the capacity normally    
        return battery.Capacity[i] == (battery.Capacity[i-1] 
                                        + (battery.Charge_power[i-1] / 4 * EFFICIENCY) 
                                        - (battery.Discharge_power[i-1] / 4))

    # Set constraint and objective for the battery
    battery.capacity_constraint = Constraint(battery.Period, rule=capacity_constraint)
    battery.over_charge = Constraint(battery.Period, rule=over_charge)
    battery.over_discharge = Constraint(battery.Period, rule=over_discharge)
    battery.negative_discharge = Constraint(battery.Period, rule=negative_discharge)
    battery.objective = Objective(rule=maximise_profit, sense=maximize)

    # Maximise the objective
    opt.solve(battery, tee=False)

    # unpack results
    charge_power, discharge_power, capacity, spot_price = ([] for i in range(4))
    for i in battery.Period:
        charge_power.append(battery.Charge_power[i].value)
        discharge_power.append(battery.Discharge_power[i].value)
        capacity.append(battery.Capacity[i].value)
        spot_price.append(battery.Price.extract_values_sparse()[None][i])

    result = pd.DataFrame({'datetime':datetime, 'spot_price':spot_price, 'charge_power':charge_power,
                           'discharge_power':discharge_power, 'opening_capacity':capacity})
    
    # make sure it does not discharge & charge at the same time
    if not len(result[(result.charge_power != 0) & (result.discharge_power != 0)]) == 0:
        print('Ops! The battery discharges & charges concurrently, the result has been returned')
        return result
    
    # convert columns charge_power & discharge_power to power
    result['power'] = np.where((result.charge_power > 0), 
                                -result.charge_power, 
                                result.discharge_power)
    
    # calculate market dispatch
    result['market_dispatch'] = np.where(result.power < 0,
                                         result.power / 4,
                                         result.power / 4 * EFFICIENCY)
    
    result = result[['datetime', 'spot_price', 'power', 'market_dispatch', 'opening_capacity']]
    
    # calculate revenue
    if include_revenue:
        result['revenue'] = np.where(result.market_dispatch < 0, 
                              result.market_dispatch * result.spot_price / MLF,
                              result.market_dispatch * result.spot_price * MLF)
    
    return result