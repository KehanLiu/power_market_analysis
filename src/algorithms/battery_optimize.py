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
    Any,
    Binary,
    NonNegativeReals
)

def battery_optimisation(
        datetime, spot_price, max_battery_capacity=1, initial_capacity=0, end_capacity=0, max_battery_power=0.25, 
        efficiency=0.9, daily_max_charging_circles=2, include_revenue=True, solver: str='glpk'):
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
    end_capacity      : the end capacity of the battery
    max_battery_power : the maximum power of the battery
    efficiency      : the efficiency of the battery
    daily_max_charging_circles : the maximum number of charging circles in a day
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
    END_CAPACITY = end_capacity
    EFFICIENCY = efficiency
    DAILY_MAX_CHARGING_CIRCLES = daily_max_charging_circles
    MLF = 1 # Marginal Loss Factor
    
    df = pd.DataFrame({'datetime': datetime, 'spot_price': spot_price}).reset_index(drop=True)
    df['period'] = df.index
    initial_period = 0
    final_period = df.index[-1]
    
    # Define model and solver
    battery = ConcreteModel()
    solver = SolverFactory(solver)

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
    
    # Make sure the battery does not charge more than the daily max charging circles everyday
    def daily_charging_limit(battery, i):
        # Get the datetime for the current period
        current_date = df.datetime[i].date()
        # Get all periods for the current date
        daily_periods = [p for p in battery.Period if df.datetime[p].date() == current_date]
        # Sum up all charging for the day (each charge power / 4 represents the energy charged in MWh)
        daily_charging = sum(battery.Charge_power[p] / 4 for p in daily_periods)
        # Ensure total daily charging doesn't exceed max capacity times daily max circles
        return daily_charging <= MAX_BATTERY_CAPACITY * DAILY_MAX_CHARGING_CIRCLES
    
    # Make sure the battery has the defined end capacity at the end of the optimisation period
    def end_capacity_constraint(battery, i):
        # Only apply constraint to the final period
        if i == battery.Period.last():
            return battery.Capacity[i] == END_CAPACITY
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
    battery.daily_charging_limit = Constraint(battery.Period, rule=daily_charging_limit)
    battery.end_capacity_constraint = Constraint(battery.Period, rule=end_capacity_constraint)
    battery.objective = Objective(rule=maximise_profit, sense=maximize)

    # Maximise the objective
    solver.solve(battery, tee=False)

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

def optimize_battery_operation_every_week(
    da_price_data,
    max_battery_capacity,
    initial_capacity,
    end_capacity,
    max_battery_power,
    efficiency,
    daily_max_charging_circles,
    include_revenue,
    result_csv_path,
    solver='glpk'
):
    """
    Optimise the battery operation for every week in the given data
    """
    # Convert datetime to datetime64[ns]
    da_price_data['datetime'] = da_price_data['datetime'].astype('datetime64[ns]')
    
    # Group by week
    da_price_data['year'] = da_price_data['datetime'].dt.year
    da_price_data['week'] = da_price_data['datetime'].dt.isocalendar().week

    # Optimise for each week for every year
    for year in da_price_data['year'].unique():
        year_data = da_price_data[da_price_data['year'] == year]
        total_yearly_result_df = pd.DataFrame()
        for week in year_data['week'].unique():
            week_data = year_data[year_data['week'] == week]
            result = battery_optimisation(
                week_data['datetime'], 
                week_data['da_price'], 
                max_battery_capacity=max_battery_capacity,
                initial_capacity=initial_capacity,
                end_capacity=end_capacity,
                max_battery_power=max_battery_power,
                efficiency=efficiency,
                daily_max_charging_circles=daily_max_charging_circles,
                include_revenue=include_revenue,    
                solver=solver
            )
            result.to_csv(f'{result_csv_path}/battery_operation_{year}_{week}.csv', index=False)
            revenue = result['revenue'].sum()
            print(f'Year: {year}, Week: {week}, Revenue: {revenue}')
            total_yearly_result_df = pd.concat([total_yearly_result_df, result])
            print(f'Total revenue for year {year}: {total_yearly_result_df.revenue.sum()}')
        total_yearly_result_df.to_csv(f'{result_csv_path}/battery_operation_{year}.csv', index=False)

def optimize_battery_operation_every_day(
    da_price_data_for_single_day,
    max_battery_capacity,
    initial_capacity,
    end_capacity,
    max_battery_power,
    efficiency,
    daily_max_charging_circles,
    include_revenue,
    result_csv_path = None,
    solver='glpk'
):
    """
    Optimise the battery operation for every single day in the given data
    """
    result = battery_optimisation(
        da_price_data_for_single_day['datetime'], 
        da_price_data_for_single_day['da_price'], 
        max_battery_capacity=max_battery_capacity,
        initial_capacity=initial_capacity,
        end_capacity=end_capacity,
        max_battery_power=max_battery_power,
        efficiency=efficiency,
        daily_max_charging_circles=daily_max_charging_circles,
        include_revenue=include_revenue,
        solver=solver
    )
    if result_csv_path:
        date = da_price_data_for_single_day['datetime'].date()
        result.to_csv(f'{result_csv_path}/battery_operation_{date}.csv', index=False)
    return result

def twenty_four_hours_optimization(
    multi_market_price_data, 
    max_battery_capacity,
    initial_capacity,
    end_capacity,
    max_battery_power,
    efficiency,
    daily_max_charging_circles, 
    start_hour=0,
    calculate_daily_revenue=True, 
    ):
    """
    Optimise the battery operation for every 24 hours based on the multi-market price data
    """
    # optimize the battery operation for the every 24 hours based on the da price first
    every_day_da_price = multi_market_price_data.groupby(multi_market_price_data['datetime'].dt.date)
    result_total = pd.DataFrame()
    for date, single_day_df in every_day_da_price:
        result_single_day = optimize_battery_operation_every_day(
            single_day_df,
            max_battery_capacity,
            initial_capacity,
            end_capacity,
            max_battery_power,
            efficiency,
            daily_max_charging_circles,
            include_revenue=True
            )
        if calculate_daily_revenue:
            result_single_day['total_daily_revenue'] = result_single_day['revenue'].sum()
        # Reset index before concatenation
        result_single_day = result_single_day.reset_index(drop=True)
        # Concatenate with ignore_index=True
        result_total = pd.concat([result_total, result_single_day], ignore_index=True)
    return result_total

def multi_market_optimization(multi_market_price_data, safty_fcr_bid_coefficient):
    result_total_single_day = twenty_four_hours_optimization(multi_market_price_data)
    mm_result = multi_market_price_data.merge(result_total_single_day, on='datetime', how='left')
    # calculate the revenue for each day
    mm_result['target_fcr_price'] = mm_result['total_daily_revenue'] / 6
    mm_result['scaled_fcr_price'] = mm_result['target_fcr_price'] * safty_fcr_bid_coefficient
    mm_result['success_fcr_bid'] = mm_result['scaled_fcr_price'] >= mm_result['GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]']
    # if fcr bid failed, do twenty_four_hours_optimization on failed timeslots of that day
    mm_result_failed = mm_result[mm_result['success_fcr_bid'] == False]
    result_total_failed_fcr_bid = twenty_four_hours_optimization(mm_result_failed, calculate_daily_revenue=False)
    # add a real_ prefix to every column of result_total_failed_fcr_bid, except for datetime
    result_total_failed_fcr_bid.columns = ['datetime' if col == 'datetime' else 'real_' + col 
                                         for col in result_total_failed_fcr_bid.columns]
    # merge result_total_failed_fcr_bid and mm_result
    result_total = mm_result.merge(result_total_failed_fcr_bid, on='datetime', how='left')
    result_total.loc[result_total.success_fcr_bid == True, 'real_revenue'] = result_total.loc[result_total.success_fcr_bid == True, 'scaled_fcr_price'] / 16
    # ffill real_opening_capacity
    result_total['real_opening_capacity'] = result_total['real_opening_capacity'].ffill()
    return result_total