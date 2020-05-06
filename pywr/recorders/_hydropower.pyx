from ._recorders import NumpyArrayNodeRecorder, Aggregator
import numpy as np
cimport numpy as np
import pandas as pd
import datetime
import warnings


cpdef double hydropower_calculation(double flow, double water_elevation, double turbine_elevation, double efficiency,
                                    double flow_unit_conversion=1.0, double energy_unit_conversion=1e-6,
                                    double density=1000.0):
    """
    Calculate the total power produced using the hydropower equation.
    
   
    Parameters
    ----------
    flow : double 
        Flow rate of water through the turbine. Should be converted using `flow_unit_conversion` to 
        units of $m^3£ per day (not per second).
    water_elevation : double
        Elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives
        the working head of the turbine.
    turbine_elevation : double
        Elevation of the turbine itself. The difference between the `water_elevation` and this value gives
        the working head of the turbine.
    efficiency : double
        An efficiency scaling factor for the power output of the turbine.
    flow_unit_conversion : double (default=1.0)
        A factor used to transform the units of flow to be compatible with the equation here. This
        should convert flow to units of $m^3/second$
    energy_unit_conversion : double (default=1e-6)
        A factor used to transform the units of power. Defaults to 1e-6 to return $MW$. 
    density : double (default=1000)
        Density of water in $kg/m^{-3}$.
        
    Returns
    -------
    power : double
        Hydropower production rate in units of energy per day.
    
    Notes
    -----
    The hydropower calculation uses the following equation.
    
    .. math:: P = \rho * g * \deltaH * q
    
    The flow rate in should be converted to units of :math:`m^3` per second using the `flow_unit_conversion` parameter.    
    
    """
    cdef double head
    cdef double power
    cdef double q

    head = water_elevation - turbine_elevation
    if head < 0.0:
        head = 0.0

    # Convert flow to correct units (typically to m3/day)
    q = flow * flow_unit_conversion
    # Power
    power = density * q * 9.81 * head * efficiency

    return power * energy_unit_conversion


cdef class HydropowerRecorder(NumpyArrayNodeRecorder):
    """ Calculates the power production using the hydropower equation

    This recorder saves an array of the hydrpower production in each timestep. It can be converted to a dataframe
    after a model run has completed. It does not calculate total energy production.

    Parameters
    ----------

    water_elevation_parameter : Parameter instance (default=None)
        Elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives
        the working head of the turbine.
    turbine_elevation : double
        Elevation of the turbine itself. The difference between the `water_elevation` and this value gives
        the working head of the turbine.
    efficiency : float (default=1.0)
        The efficiency of the turbine.
    density : float (default=1000.0)
        The density of water.
    flow_unit_conversion : float (default=1.0)
        A factor used to transform the units of flow to be compatible with the equation here. This
        should convert flow to units of :math:`m^3/second`
    energy_unit_conversion : float (default=1e-6)
        A factor used to transform the units of total energy. Defaults to 1e-6 to return :math:`MW`.

    Notes
    -----
    The hydropower calculation uses the following equation.

    .. math:: P = \\rho * g * \\delta H * q

    The flow rate in should be converted to units of :math:`m^3` per second using the `flow_unit_conversion` parameter.

    Head is calculated from the given `water_elevation_parameter` and `turbine_elevation` value. If water elevation
    is given then head is the difference in elevation between the water and the turbine. If water elevation parameter
    is `None` then the head is simply the turbine elevation.


    See Also
    --------
    TotalHydroEnergyRecorder
    pywr.parameters.HydropowerTargetParameter

    """
    def __init__(self, model, node, water_elevation_parameter=None, turbine_elevation=0.0, efficiency=1.0, density=1000,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        super(HydropowerRecorder, self).__init__(model, node, **kwargs)

        self.water_elevation_parameter = water_elevation_parameter
        self.turbine_elevation = turbine_elevation
        self.efficiency = efficiency
        self.density = density
        self.flow_unit_conversion = flow_unit_conversion
        self.energy_unit_conversion = energy_unit_conversion

    property water_elevation_parameter:
        def __get__(self):
            return self._water_elevation_parameter
        def __set__(self, parameter):
            if self._water_elevation_parameter:
                self.children.remove(self._water_elevation_parameter)
            self.children.add(parameter)
            self._water_elevation_parameter = parameter        

    cpdef after(self):
        cdef int i
        cdef double q, head, power
        cdef Timestep ts = self.model.timestepper.current
        cdef ScenarioIndex scenario_index
        flow = self.node.flow

        for scenario_index in self.model.scenarios.combinations:

            if self._water_elevation_parameter is not None:
                head = self._water_elevation_parameter.get_value(scenario_index)
                if self.turbine_elevation is not None:
                    head -= self.turbine_elevation
            elif self.turbine_elevation is not None:
                head = self.turbine_elevation
            else:
                raise ValueError('One or both of storage_node or level must be set.')

            # -ve head is not valid
            head = max(head, 0.0)
            # Get the flow from the current node
            q = self._node._flow[scenario_index.global_id]
            power = hydropower_calculation(q, head, 0.0, self.efficiency, density=self.density,
                                             flow_unit_conversion=self.flow_unit_conversion,
                                             energy_unit_conversion=self.energy_unit_conversion)

            self._data[ts.index, scenario_index.global_id] = power

    @classmethod
    def load(cls, model, data):
        from pywr.parameters import load_parameter
        node = model._get_node_from_ref(model, data.pop("node"))
        if "water_elevation_parameter" in data:
            water_elevation_parameter = load_parameter(model, data.pop("water_elevation_parameter"))
        else:
            water_elevation_parameter = None

        return cls(model, node, water_elevation_parameter=water_elevation_parameter, **data)
HydropowerRecorder.register()


cdef class HydropowerRecorderWithVaribaleTailwater(NumpyArrayNodeRecorder):
    """ Calculates the power production using the hydropower equation

    This recorder saves an array of the hydrpower production in each timestep. It can be converted to a dataframe
    after a model run has completed. It does not calculate total energy production.

    Parameters
    ----------

    water_elevation_parameter : Parameter instance (default=None)
        Elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives
        the working head of the turbine.
    turbine_elevation_parameter : Parameter instance (default=None)
        Elevation of the turbine itself. The difference between the `water_elevation` and this value gives
        the working head of the turbine. It is recommended to use 'InterpolatedLevelParameter'.
    efficiency : float (default=1.0)
        The efficiency of the turbine.
    density : float (default=1000.0)
        The density of water.
    flow_unit_conversion : float (default=1.0)
        A factor used to transform the units of flow to be compatible with the equation here. This
        should convert flow to units of :math:`m^3/second`
    energy_unit_conversion : float (default=1e-6)
        A factor used to transform the units of total energy. Defaults to 1e-6 to return :math:`MW`.

    Notes
    -----
    The hydropower calculation uses the following equation.

    .. math:: P = \\rho * g * \\delta H * q

    The flow rate in should be converted to units of :math:`m^3` per second using the `flow_unit_conversion` parameter.

    Head is calculated from the given `water_elevation_parameter` and `turbine_elevation` value. If water elevation
    is given then head is the difference in elevation between the water and the turbine. If water elevation parameter
    is `None` then the head is simply the turbine elevation.


    See Also
    --------
    TotalHydroEnergyRecorder
    pywr.parameters.HydropowerTargetParameter

    """
    def __init__(self, model, node, water_elevation_parameter=None, turbine_elevation_parameter=None, efficiency=1.0, density=1000,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        super(HydropowerRecorderWithVaribaleTailwater, self).__init__(model, node, **kwargs)

        self.water_elevation_parameter = water_elevation_parameter
        self.turbine_elevation_parameter = turbine_elevation_parameter
        self.efficiency = efficiency
        self.density = density
        self.flow_unit_conversion = flow_unit_conversion
        self.energy_unit_conversion = energy_unit_conversion

    property water_elevation_parameter:
        def __get__(self):
            return self._water_elevation_parameter
        def __set__(self, parameter):
            if self._water_elevation_parameter:
                self.children.remove(self._water_elevation_parameter)
            self.children.add(parameter)
            self._water_elevation_parameter = parameter

    property turbine_elevation_parameter:
        def __get__(self):
            return self._turbine_elevation_parameter
        def __set__(self, parameter):
            if self._turbine_elevation_parameter:
                self.children.remove(self._turbine_elevation_parameter)
            self.children.add(parameter)
            self._turbine_elevation_parameter = parameter         

    cpdef after(self):
        cdef int i
        cdef double q, head, power
        cdef Timestep ts = self.model.timestepper.current
        cdef ScenarioIndex scenario_index
        flow = self.node.flow

        for scenario_index in self.model.scenarios.combinations:

            if self._water_elevation_parameter is not None:
                head = self._water_elevation_parameter.get_value(scenario_index)
                if self.turbine_elevation_parameter is not None:                    
                    head -= self.turbine_elevation_parameter.get_value(scenario_index)
            elif self.turbine_elevation_parameter is not None:
                head = self.turbine_elevation_parameter.get_value(scenario_index)
            else:
                raise ValueError('One or both of storage_node or level must be set.')

            # -ve head is not valid
            head = max(head, 0.0)
            # Get the flow from the current node
            q = self._node._flow[scenario_index.global_id]
            power = hydropower_calculation(q, head, 0.0, self.efficiency, density=self.density,
                                             flow_unit_conversion=self.flow_unit_conversion,
                                             energy_unit_conversion=self.energy_unit_conversion)

            self._data[ts.index, scenario_index.global_id] = power

    @classmethod
    def load(cls, model, data):
        from pywr.parameters import load_parameter
        node = model._get_node_from_ref(model, data.pop("node"))
        if "water_elevation_parameter" in data:
            water_elevation_parameter = load_parameter(model, data.pop("water_elevation_parameter"))
        else:
            water_elevation_parameter = None

        if "turbine_elevation_parameter" in data:
            turbine_elevation_parameter = load_parameter(model, data.pop("turbine_elevation_parameter"))
        else:
            turbine_elevation_parameter = None

        return cls(model, node, water_elevation_parameter=water_elevation_parameter,turbine_elevation_parameter=turbine_elevation_parameter, **data)
HydropowerRecorderWithVaribaleTailwater.register()


cdef class TotalHydroEnergyRecorder(BaseConstantNodeRecorder):
    """ Calculates the total energy production using the hydropower equation from a model run.

    This recorder saves the total energy production in each scenario during a model run. It does not save a timeseries
    or power, but rather total energy.

    Parameters
    ----------

    water_elevation_parameter : Parameter instance (default=None)
        Elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives
        the working head of the turbine.
    turbine_elevation : double
        Elevation of the turbine itself. The difference between the `water_elevation` and this value gives
        the working head of the turbine.
    efficiency : float (default=1.0)
        The efficiency of the turbine.
    density : float (default=1000.0)
        The density of water.
    flow_unit_conversion : float (default=1.0)
        A factor used to transform the units of flow to be compatible with the equation here. This
        should convert flow to units of :math:`m^3/day`
    energy_unit_conversion : float (default=1e-6)
        A factor used to transform the units of total energy. Defaults to 1e-6 to return :math:`MJ`.

    Notes
    -----
    The hydropower calculation uses the following equation.

    .. math:: P = \\rho * g * \\delta H * q

    The flow rate in should be converted to units of :math:`m^3` per day using the `flow_unit_conversion` parameter.

    Head is calculated from the given `water_elevation_parameter` and `turbine_elevation` value. If water elevation
    is given then head is the difference in elevation between the water and the turbine. If water elevation parameter
    is `None` then the head is simply the turbine elevation.


    See Also
    --------
    HydropowerRecorder
    pywr.parameters.HydropowerTargetParameter

    """
    def __init__(self, model, node, water_elevation_parameter=None, turbine_elevation=0.0, efficiency=1.0, density=1000,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        super(TotalHydroEnergyRecorder, self).__init__(model, node, **kwargs)

        self.water_elevation_parameter = water_elevation_parameter
        self.turbine_elevation = turbine_elevation
        self.efficiency = efficiency
        self.density = density
        self.flow_unit_conversion = flow_unit_conversion
        self.energy_unit_conversion = energy_unit_conversion

    property water_elevation_parameter:
        def __get__(self):
            return self._water_elevation_parameter
        def __set__(self, parameter):
            if self._water_elevation_parameter:
                self.children.remove(self._water_elevation_parameter)
            self.children.add(parameter)
            self._water_elevation_parameter = parameter

    cpdef after(self):
        """ Calculate the  """
        cdef int i
        cdef double q, head, power
        cdef Timestep ts = self.model.timestepper.current
        cdef double days = ts.days
        cdef ScenarioIndex scenario_index
        flow = self.node.flow

        for scenario_index in self.model.scenarios.combinations:

            if self._water_elevation_parameter is not None:
                head = self._water_elevation_parameter.get_value(scenario_index)
                if self.turbine_elevation is not None:
                    head -= self.turbine_elevation
            elif self.turbine_elevation is not None:
                head = self.turbine_elevation
            else:
                raise ValueError('One or both of storage_node or level must be set.')

            # -ve head is not valid
            head = max(head, 0.0)
            # Get the flow from the current node
            q = self._node._flow[scenario_index.global_id]
            power = hydropower_calculation(q, head, 0.0, self.efficiency, density=self.density,
                                             flow_unit_conversion=self.flow_unit_conversion,
                                             energy_unit_conversion=self.energy_unit_conversion)

            self._values[scenario_index.global_id] += power * days *24

    @classmethod
    def load(cls, model, data):
        from pywr.parameters import load_parameter
        node = model._get_node_from_ref(model, data.pop("node"))
        if "water_elevation_parameter" in data:
            water_elevation_parameter = load_parameter(model, data.pop("water_elevation_parameter"))
        else:
            water_elevation_parameter = None

        return cls(model, node, water_elevation_parameter=water_elevation_parameter, **data)
TotalHydroEnergyRecorder.register()


cdef class TotalHydroEnergyRecorderWithVaribaleTailwater(BaseConstantNodeRecorder):
    """ Calculates the total energy production using the hydropower equation from a model run.

    This recorder saves the total energy production in each scenario during a model run. It does not save a timeseries
    or power, but rather total energy.

    Parameters
    ----------

    water_elevation_parameter : Parameter instance (default=None)
        Elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives
        the working head of the turbine.
    turbine_elevation_parameter : Parameter instance (default=None)
        Elevation of the turbine itself. The difference between the `water_elevation` and this value gives
        the working head of the turbine.
    efficiency : float (default=1.0)
        The efficiency of the turbine.
    density : float (default=1000.0)
        The density of water.
    flow_unit_conversion : float (default=1.0)
        A factor used to transform the units of flow to be compatible with the equation here. This
        should convert flow to units of :math:`m^3/second`
    energy_unit_conversion : float (default=1e-6)
        A factor used to transform the units of total energy. Defaults to 1e-6 to return :math:`MWh`.

    Notes
    -----
    The hydropower calculation uses the following equation.

    .. math:: P = \\rho * g * \\delta H * q

    The flow rate in should be converted to units of :math:`m^3` per second using the `flow_unit_conversion` parameter.

    Head is calculated from the given `water_elevation_parameter` and `turbine_elevation_parameter` value. If water elevation
    is given then head is the difference in elevation between the water and the turbine. If water elevation parameter
    is `None` then the head is simply the turbine elevation.


    See Also
    --------
    HydropowerRecorder
    pywr.parameters.HydropowerTargetParameter

    """
    def __init__(self, model, node, water_elevation_parameter=None, turbine_elevation_parameter=None, efficiency=1.0, density=1000,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        super(TotalHydroEnergyRecorderWithVaribaleTailwater, self).__init__(model, node, **kwargs)

        self.water_elevation_parameter = water_elevation_parameter
        self.turbine_elevation_parameter = turbine_elevation_parameter
        self.efficiency = efficiency
        self.density = density
        self.flow_unit_conversion = flow_unit_conversion
        self.energy_unit_conversion = energy_unit_conversion

    property water_elevation_parameter:
        def __get__(self):
            return self._water_elevation_parameter
        def __set__(self, parameter):
            if self._water_elevation_parameter:
                self.children.remove(self._water_elevation_parameter)
            self.children.add(parameter)
            self._water_elevation_parameter = parameter

    property turbine_elevation_parameter:
        def __get__(self):
            return self._turbine_elevation_parameter
        def __set__(self, parameter):
            if self._turbine_elevation_parameter:
                self.children.remove(self._turbine_elevation_parameter)
            self.children.add(parameter)
            self._turbine_elevation_parameter = parameter 

    cpdef after(self):
        """ Calculate the  """
        cdef int i
        cdef double q, head, power
        cdef Timestep ts = self.model.timestepper.current
        cdef double days = ts.days
        cdef ScenarioIndex scenario_index
        flow = self.node.flow

        for scenario_index in self.model.scenarios.combinations:

            if self._water_elevation_parameter is not None:
                head = self._water_elevation_parameter.get_value(scenario_index)
                if self.turbine_elevation_parameter is not None:                    
                    head -= self.turbine_elevation_parameter.get_value(scenario_index)
            elif self.turbine_elevation_parameter is not None:
                head = self.turbine_elevation_parameter.get_value(scenario_index)
            else:
                raise ValueError('One or both of storage_node or level must be set.')

            # -ve head is not valid
            head = max(head, 0.0)
            # Get the flow from the current node
            q = self._node._flow[scenario_index.global_id]
            power = hydropower_calculation(q, head, 0.0, self.efficiency, density=self.density,
                                             flow_unit_conversion=self.flow_unit_conversion,
                                             energy_unit_conversion=self.energy_unit_conversion)

            self._values[scenario_index.global_id] += power * days * 24

    @classmethod
    def load(cls, model, data):
        from pywr.parameters import load_parameter
        node = model._get_node_from_ref(model, data.pop("node"))
        if "water_elevation_parameter" in data:
            water_elevation_parameter = load_parameter(model, data.pop("water_elevation_parameter"))
        else:
            water_elevation_parameter = None

        if "turbine_elevation_parameter" in data:
            turbine_elevation_parameter = load_parameter(model, data.pop("turbine_elevation_parameter"))
        else:
            turbine_elevation_parameter = None

        return cls(model, node, water_elevation_parameter=water_elevation_parameter,turbine_elevation_parameter=turbine_elevation_parameter, **data)
TotalHydroEnergyRecorderWithVaribaleTailwater.register()


cdef class AnnualHydroEnergyRecorder(Recorder):
    """Abstract class for recording cumulative annual differences between actual flow and max_flow.

    This abstract class can be subclassed to calculate statistics of differences between cumulative
    annual actual flow and max_flow on multiple nodes. The abstract class records the cumulative
    actual flow and max_flow from multiple nodes and provides an internal data attribute on which
    to store a derived statistic. A reset day and month control the day on which the cumulative
    data is reset to zero.

    Parameters
    ----------
    model : `pywr.core.Model`
    nodes : iterable of `pywr.core.Node`
        Iterable of Node instances to record.
    reset_month, reset_day : int
        The month and day in which the cumulative actual and max_flow are reset to zero.

    Notes
    -----
    If the first time-step of a simulation does not align with `reset_day` and `reset_month` then
    the first period of the model will be less than one year in length.
    """

    def __init__(self, model, nodes, water_elevation_parameter=None, turbine_elevation_parameter=None, efficiency=1.0, density=1000,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, reset_day=1, reset_month=1, **kwargs):
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        super().__init__(model, **kwargs)
        self.nodes = [n for n in nodes]

        #self.water_elevation_parameter = water_elevation_parameter
        self.water_elevation_parameter = list(water_elevation_parameter)
        self.turbine_elevation_parameter = list(turbine_elevation_parameter)
        self.efficiency = efficiency
        self.density = density
        self.flow_unit_conversion = flow_unit_conversion
        self.energy_unit_conversion = energy_unit_conversion
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        # Validate the reset day and month
        # date will raise a ValueError if invalid. We use a non-leap year to ensure
        # 29th February is an invalid reset day.
        datetime.date(1999, reset_month, reset_day)

        self.reset_day = reset_day
        self.reset_month = reset_month

        for p in self.water_elevation_parameter:
            p.parents.add(self)

        for p in self.turbine_elevation_parameter:
            p.parents.add(self)

#    property water_elevation_parameter:
#        def __get__(self):
#            return self._water_elevation_parameter
#        def __set__(self, parameter):
#            if self._water_elevation_parameter:
#                self.children.remove(self._water_elevation_parameter)
#            self.children.add(parameter)
#            self._water_elevation_parameter = parameter

    cpdef setup(self):
        cdef int ncomb = len(self.model.scenarios.combinations)
        cdef int nts = len(self.model.timestepper)

        start = self.model.timestepper.start
        end_year = self.model.timestepper.end.year
        nyears = end_year - start.year + 1
        if start.day != self.reset_day and start.month != self.reset_month:
            nyears += 1

        self._data = np.zeros((nyears, ncomb,), np.float64)
        self._annual_energy = np.zeros_like(self._data)
        self._current_year_index = 0
        self.switch = np.zeros_like(self._data)

    cpdef reset(self):
        self._data[...] = 0
        self._annual_energy[...] = 0
        self.switch[...] = 0

        self._current_year_index = -1
        self._last_reset_year = -1

    cpdef before(self):

        cdef Timestep ts = self.model.timestepper.current
        if ts.year != self._last_reset_year:
            # I.e. we're in a new year and ...
            # ... we're at or past the reset month/day
            if ts.month > self.reset_month or \
                    (ts.month == self.reset_month and ts.day >= self.reset_day):
                self._current_year_index += 1
                self._last_reset_year = ts.year

            if self._current_year_index < 0:
                # reset date doesn't align with the start of the model
                self._current_year_index = 0

    property data:
        def __get__(self):
            return np.array(self._data, dtype=np.float64)

    property current_data:
        def __get__(self):
            return np.array(self._data[self._current_year_index, :], dtype=np.float64)

    cpdef after(self):

        cdef double q, head, power
        cdef double energy_temp
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self.model.timestepper.current
        cdef double days = ts.days
        cdef int i = self._current_year_index
        cdef int j
        cdef nodes_length = range(0,len(self.nodes),1)

        for scenario_index in self.model.scenarios.combinations:
            j = scenario_index.global_id
            if self.switch[i, j] == 1:
                self._annual_energy[i, j] = 0
                self.switch[i, j] = 0

            for node_index in nodes_length:
                head = self.water_elevation_parameter[node_index].get_value(scenario_index)
                head -= self.turbine_elevation_parameter[node_index].get_value(scenario_index)
                head = max(head, 0.0)
                # Get the flow from the current node
                q = self.nodes[node_index].flow[scenario_index.global_id]
                power = hydropower_calculation(q, head, 0.0, self.efficiency, density=self.density,
                                                flow_unit_conversion=self.flow_unit_conversion,
                                                energy_unit_conversion=self.energy_unit_conversion)

                energy_temp = power * days * 24

                self._annual_energy[i, j] += energy_temp

            if ts.month == 12 and ts.day == 31:
                self.switch[i, j] = 1

            self._data[i, j] = self._annual_energy[i, j]   
        
        return 0

    cpdef double[:] values(self):
        """Compute a value for each scenario using `temporal_agg_func`.
        """
        return self._temporal_aggregator.aggregate_2d(self._data, axis=0, ignore_nan=self.ignore_nan)

    def to_dataframe_annual(self):
        """ Return a `pandas.DataFrame` of the recorder data

        This DataFrame contains a MultiIndex for the columns with the recorder name
        as the first level and scenario combination names as the second level. This
        allows for easy combination with multiple recorder's DataFrames
        """
        index = np.asarray(range(self.model.timestepper.start.year,self.model.timestepper.end.year+1,1),dtype=np.float64)
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self._data), index=index, columns=sc_index)

    @classmethod
    def load(cls, model, data):
        from pywr.parameters import load_parameter
        nodes = [model._get_node_from_ref(model, node_name) for node_name in data.pop('nodes')]
        if "water_elevation_parameter" in data:
            water_elevation_parameter = [load_parameter(model, p) for p in data.pop("water_elevation_parameter")]
        else:
            water_elevation_parameter = None

        if "turbine_elevation_parameter" in data:
            turbine_elevation_parameter = [load_parameter(model, p) for p in data.pop("turbine_elevation_parameter")]
        else:
            turbine_elevation_parameter = None    

        return cls(model, nodes, water_elevation_parameter = water_elevation_parameter, turbine_elevation_parameter = turbine_elevation_parameter, **data)

AnnualHydroEnergyRecorder.register()


cdef class AnnualNonHydroEnergyRecorder(Recorder):
    """Abstract class for recording cumulative annual differences between actual flow and max_flow.

    This abstract class can be subclassed to calculate statistics of differences between cumulative
    annual actual flow and max_flow on multiple nodes. The abstract class records the cumulative
    actual flow and max_flow from multiple nodes and provides an internal data attribute on which
    to store a derived statistic. A reset day and month control the day on which the cumulative
    data is reset to zero.

    Parameters
    ----------
    model : `pywr.core.Model`
    nodes : iterable of `pywr.core.Node`
        Iterable of Node instances to record.
    reset_month, reset_day : int
        The month and day in which the cumulative actual and max_flow are reset to zero.

    Notes
    -----
    If the first time-step of a simulation does not align with `reset_day` and `reset_month` then
    the first period of the model will be less than one year in length.
    """

    def __init__(self, model, nodes, water_elevation_parameter=None, turbine_elevation_parameter=None, non_hydro_capacity_parameter_MW=None, energy_demand_parameter_MWh_per_day=None, efficiency=1.0, density=1000,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, reset_day=1, reset_month=1, **kwargs):
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        super().__init__(model, **kwargs)
        self.nodes = [n for n in nodes]

        self.water_elevation_parameter = list(water_elevation_parameter)
        self.turbine_elevation_parameter = list(turbine_elevation_parameter)
        self.non_hydro_capacity_parameter_MW = non_hydro_capacity_parameter_MW
        self.energy_demand_parameter_MWh_per_day = energy_demand_parameter_MWh_per_day
        self.efficiency = efficiency
        self.density = density
        self.flow_unit_conversion = flow_unit_conversion
        self.energy_unit_conversion = energy_unit_conversion
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        # Validate the reset day and month
        # date will raise a ValueError if invalid. We use a non-leap year to ensure
        # 29th February is an invalid reset day.
        datetime.date(1999, reset_month, reset_day)

        self.reset_day = reset_day
        self.reset_month = reset_month

        for p in self.water_elevation_parameter:
            p.parents.add(self)

        for p in self.turbine_elevation_parameter:
            p.parents.add(self)

    property non_hydro_capacity_parameter_MW:
        def __get__(self):
            return self._non_hydro_capacity_parameter_MW
        def __set__(self, parameter):
            if self._non_hydro_capacity_parameter_MW:
                self.children.remove(self._non_hydro_capacity_parameter_MW)
            self.children.add(parameter)
            self._non_hydro_capacity_parameter_MW = parameter

    property energy_demand_parameter_MWh_per_day:
        def __get__(self):
            return self._energy_demand_parameter_MWh_per_day
        def __set__(self, parameter):
            if self._energy_demand_parameter_MWh_per_day:
                self.children.remove(self._energy_demand_parameter_MWh_per_day)
            self.children.add(parameter)
            self._energy_demand_parameter_MWh_per_day = parameter


    cpdef setup(self):
        cdef int ncomb = len(self.model.scenarios.combinations)
        cdef int nts = len(self.model.timestepper)

        start = self.model.timestepper.start
        end_year = self.model.timestepper.end.year
        nyears = end_year - start.year + 1
        if start.day != self.reset_day and start.month != self.reset_month:
            nyears += 1

        self._data = np.zeros((nyears, ncomb,), np.float64)
        self._annual_hydro_energy = np.zeros_like(self._data)
        self._annual_non_hydro_energy = np.zeros_like(self._data)
        self._annual_energy_demand = np.zeros_like(self._data)
        self._current_year_index = 0
        self.switch = np.zeros_like(self._data)

    cpdef reset(self):
        self._data[...] = 0
        self._annual_hydro_energy[...] = 0
        self._annual_non_hydro_energy[...] = 0
        self._annual_energy_demand[...] = 0
        self.switch[...] = 0

        self._current_year_index = -1
        self._last_reset_year = -1

    cpdef before(self):

        cdef Timestep ts = self.model.timestepper.current
        if ts.year != self._last_reset_year:
            # I.e. we're in a new year and ...
            # ... we're at or past the reset month/day
            if ts.month > self.reset_month or \
                    (ts.month == self.reset_month and ts.day >= self.reset_day):
                self._current_year_index += 1
                self._last_reset_year = ts.year

            if self._current_year_index < 0:
                # reset date doesn't align with the start of the model
                self._current_year_index = 0

    property data:
        def __get__(self):
            return np.array(self._data, dtype=np.float64)

    property current_data:
        def __get__(self):
            return np.array(self._data[self._current_year_index, :], dtype=np.float64)

    cpdef after(self):

        cdef double q, head, power
        cdef double energy_temp
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self.model.timestepper.current
        cdef double days = ts.days
        cdef int i = self._current_year_index
        cdef int j
        cdef nodes_length = range(0,len(self.nodes),1)

        for scenario_index in self.model.scenarios.combinations:
            j = scenario_index.global_id
            if self.switch[i, j] == 1:
                self._annual_hydro_energy[i, j] = 0
                self._annual_energy_demand[i, j] = 0
                self._annual_non_hydro_energy[i, j] = 0
                self.switch[i, j] = 0

            non_hydro_capacity = self.non_hydro_capacity_parameter_MW.get_value(scenario_index)
            energy_demand = self.energy_demand_parameter_MWh_per_day.get_value(scenario_index)

            total_hydro = 0
            for node_index in nodes_length:
                head = self.water_elevation_parameter[node_index].get_value(scenario_index)
                head -= self.turbine_elevation_parameter[node_index].get_value(scenario_index)
                head = max(head, 0.0)
                # Get the flow from the current node
                q = self.nodes[node_index].flow[scenario_index.global_id]
                power = hydropower_calculation(q, head, 0.0, self.efficiency, density=self.density,
                                                flow_unit_conversion=self.flow_unit_conversion,
                                                energy_unit_conversion=self.energy_unit_conversion)

                energy_hydro = power * days * 24
                total_hydro += energy_hydro
                self._annual_hydro_energy[i, j] += energy_hydro

            energy_non_hydro = min(max(energy_demand * days - total_hydro,0),non_hydro_capacity* days * 24)
            self._annual_non_hydro_energy[i, j] += energy_non_hydro
            self._annual_energy_demand[i, j] += energy_demand * days

            if ts.month == 12 and ts.day == 31:
                self.switch[i, j] = 1
                
            self._data[i, j] = self._annual_non_hydro_energy[i, j]

        return 0

    cpdef double[:] values(self):
        """Compute a value for each scenario using `temporal_agg_func`.
        """
        return self._temporal_aggregator.aggregate_2d(self._data, axis=0, ignore_nan=self.ignore_nan)

    def to_dataframe_annual(self):
        """ Return a `pandas.DataFrame` of the recorder data

        This DataFrame contains a MultiIndex for the columns with the recorder name
        as the first level and scenario combination names as the second level. This
        allows for easy combination with multiple recorder's DataFrames
        """
        index = np.asarray(range(self.model.timestepper.start.year,self.model.timestepper.end.year+1,1),dtype=np.float64)
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self._data), index=index, columns=sc_index)

    @classmethod
    def load(cls, model, data):
        from pywr.parameters import load_parameter
        nodes = [model._get_node_from_ref(model, node_name) for node_name in data.pop('nodes')]
        if "water_elevation_parameter" in data:
            water_elevation_parameter = [load_parameter(model, p) for p in data.pop("water_elevation_parameter")]
        else:
            water_elevation_parameter = None

        if "turbine_elevation_parameter" in data:
            turbine_elevation_parameter = [load_parameter(model, p) for p in data.pop("turbine_elevation_parameter")]
        else:
            turbine_elevation_parameter = None

        if "non_hydro_capacity_parameter_MW" in data:
            non_hydro_capacity_parameter_MW = load_parameter(model, data.pop("non_hydro_capacity_parameter_MW"))
        else:
            non_hydro_capacity_parameter_MW = None

        if "energy_demand_parameter_MWh_per_day" in data:
            energy_demand_parameter_MWh_per_day = load_parameter(model, data.pop("energy_demand_parameter_MWh_per_day"))
        else:
            energy_demand_parameter_MWh_per_day = None

        return cls(model, nodes, water_elevation_parameter = water_elevation_parameter, turbine_elevation_parameter = turbine_elevation_parameter,non_hydro_capacity_parameter_MW = non_hydro_capacity_parameter_MW, energy_demand_parameter_MWh_per_day = energy_demand_parameter_MWh_per_day, **data)

AnnualNonHydroEnergyRecorder.register()
