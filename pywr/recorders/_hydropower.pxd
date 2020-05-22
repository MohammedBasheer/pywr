from ._recorders cimport NumpyArrayNodeRecorder, BaseConstantNodeRecorder, Recorder, Aggregator
from pywr.parameters._parameters cimport Parameter
from .._core cimport Timestep, Scenario, ScenarioIndex


cdef class HydropowerRecorder(NumpyArrayNodeRecorder):
    cdef Parameter _water_elevation_parameter
    cdef public double turbine_elevation
    cdef public double flow_unit_conversion
    cdef public double energy_unit_conversion
    cdef public double density
    cdef public double efficiency


cdef class TotalHydroEnergyRecorder(BaseConstantNodeRecorder):
    cdef Parameter _water_elevation_parameter
    cdef public double turbine_elevation
    cdef public double flow_unit_conversion
    cdef public double energy_unit_conversion
    cdef public double density
    cdef public double efficiency


cdef class AnnualHydroEnergyRecorder(Recorder):
    cdef public list nodes
    cdef public list water_elevation_parameter
    cdef public list turbine_elevation_parameter
    cdef public double flow_unit_conversion
    cdef public double energy_unit_conversion
    cdef public double density
    cdef public double efficiency
    cdef public int reset_day
    cdef public int reset_month
    cdef double[:, :] _data
    cdef double[:, :] _annual_energy
    cdef int _current_year_index
    cdef int _last_reset_year
    cdef Aggregator _temporal_aggregator
    cdef double[:, :] switch


cdef class AnnualNonHydroEnergyRecorder(Recorder):
    cdef public list nodes
    cdef public list water_elevation_parameter
    cdef public list turbine_elevation_parameter
    cdef Parameter _non_hydro_capacity_parameter_MW
    cdef Parameter _energy_demand_parameter_MWh_per_day
    cdef public double flow_unit_conversion
    cdef public double energy_unit_conversion
    cdef public double density
    cdef public double efficiency
    cdef public double system_losses
    cdef public int reset_day
    cdef public int reset_month
    cdef double[:, :] _data
    cdef double[:, :] _annual_hydro_energy
    cdef double[:, :] _annual_non_hydro_energy
    cdef double[:, :] _annual_energy_demand
    cdef int _current_year_index
    cdef int _last_reset_year
    cdef Aggregator _temporal_aggregator
    cdef double[:, :] switch
    

cdef class HydropowerRecorderWithVaribaleTailwater(NumpyArrayNodeRecorder):
    cdef Parameter _water_elevation_parameter
    cdef Parameter _turbine_elevation_parameter
    cdef public double flow_unit_conversion
    cdef public double energy_unit_conversion
    cdef public double density
    cdef public double efficiency


cdef class TotalHydroEnergyRecorderWithVaribaleTailwater(BaseConstantNodeRecorder):
    cdef Parameter _water_elevation_parameter
    cdef Parameter _turbine_elevation_parameter
    cdef public double flow_unit_conversion
    cdef public double energy_unit_conversion
    cdef public double density
    cdef public double efficiency
    