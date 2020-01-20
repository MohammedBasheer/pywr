from ._recorders cimport NumpyArrayNodeRecorder, BaseConstantNodeRecorder, Recorder
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
    cdef public list turbine_elevation
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
    