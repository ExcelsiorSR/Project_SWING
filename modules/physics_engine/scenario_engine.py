# =============================================
#              MODULE IMPORTS
# =============================================

import pandapower as pp
import random

# =============================================
#              FNCTIONAL MODULE
# =============================================

def apply_load_multiplier(net, multiplier: float):
    """
    Scales the active and reactive power of all loads in the network.
    """
    
    net.load['p_mw'] = net.load['p_mw'] * multiplier

    net.load['q_mvar'] = net.load['q_mvar'] * multiplier
    
    return net

def derate_lines(net, derate_multiplier: float):
    """
    Reduces the maximum thermal current capacity of all transmission lines.
    """
    net.line['max_i_ka'] = net.line['max_i_ka'] * derate_multiplier
    
    return net

def simulate_transformer_aging(asset_registry, trafo_id: int, severity: float = 1.0):
    """
    Simulates the chemical breakdown of transformer cellulose and oil.
    Increases the DGA (Dissolved Gas Analysis) score and reduces RUL.
    """
    if trafo_id in asset_registry["transformers"]:
        trafo = asset_registry["transformers"][trafo_id]
        
        # Exponential degradation curve based on severity
        trafo["dga_score"] += (random.uniform(0.5, 2.0) * severity)
        trafo["rul_days"] -= (random.uniform(1.0, 3.0) * severity)
        
        # Threshold checks based on IEEE Std C57.104
        if trafo["dga_score"] > 80.0:
            trafo["fault_class"] = "HIGH_ENERGY_ARCING"
        elif trafo["dga_score"] > 50.0:
            trafo["fault_class"] = "THERMAL_FAULT"
            
        # Floor the RUL at 0
        trafo["rul_days"] = max(0.0, trafo["rul_days"])
        
    return asset_registry

def simulate_generator_wear(asset_registry, gen_id: int, wear_rate: float = 1.0):
    """
    Simulates mechanical bearing wear in synchronous generators.
    Increases vibration amplitude and maps to ISO 10816 states.
    """
    if gen_id in asset_registry["generators"]:
        gen = asset_registry["generators"][gen_id]
        
        gen["vibration_amp"] += (random.uniform(0.1, 0.5) * wear_rate)
        gen["rul_days"] -= (random.uniform(0.8, 2.5) * wear_rate)
        
        # Threshold checks based on ISO 10816
        if gen["vibration_amp"] > 7.5:
            gen["fault_class"] = "CRITICAL_UNBALANCE"
        elif gen["vibration_amp"] > 4.5:
            gen["fault_class"] = "BEARING_WEAR"
            
        gen["rul_days"] = max(0.0, gen["rul_days"])
        
    return asset_registry

def simulate_breaker_degradation(asset_registry, breaker_id: int, friction_factor: float = 1.0):
    """
    Simulates trip coil mechanism friction and latch delay in HVCBs.
    """
    if breaker_id in asset_registry["circuit_breakers"]:
        cb = asset_registry["circuit_breakers"][breaker_id]
        
        cb["tccs_delay_ms"] += (random.uniform(1.0, 4.0) * friction_factor)
        cb["rul_days"] -= (random.uniform(0.5, 2.0) * friction_factor)
        
        # 50ms delay is considered a critical failure in clearing faults
        if cb["tccs_delay_ms"] > 50.0: 
            cb["fault_class"] = "MECHANICAL_JAM"
        elif cb["tccs_delay_ms"] > 30.0:
            cb["fault_class"] = "SLUGGISH_OPERATION"
            
        cb["rul_days"] = max(0.0, cb["rul_days"])
        
    return asset_registry

def simulate_relay_degradation(asset_registry, relay_id: int, noise_factor: float = 1.0):
    """
    Simulates the degradation of numerical relay logic integrity.
    Mimics memory corruption or CT saturation errors.
    """
    if relay_id in asset_registry["relays"]:
        relay = asset_registry["relays"][relay_id]
        
        # Decreases logic integrity percentage
        relay["logic_integrity"] -= (random.uniform(0.5, 2.0) * noise_factor)
        
        if relay["logic_integrity"] < 50.0:
            relay["fault_class"] = "CRITICAL_LOGIC_FAILURE"
        elif relay["logic_integrity"] < 80.0:
            relay["fault_class"] = "MINOR_CALIBRATION_ERROR"
            
        relay["logic_integrity"] = max(0.0, relay["logic_integrity"])
        
    return asset_registry


# ==========================================================
#               EXTENDED SCENARIO EVENTS
# ==========================================================

def simulate_generator_trip(net, asset_registry, gen_id: int, trip_type: str = "FORCED_OUTAGE"):
    """Forces a generator out of service and stamps the event onto the asset registry."""
    if gen_id in net.gen.index:
        net.gen.at[gen_id, 'in_service'] = False
    else:
        raise ValueError(f"Generator {gen_id} not found in network.")

    if gen_id in asset_registry["generators"]:
        gen_record = asset_registry["generators"][gen_id]
        gen_record["fault_class"] = trip_type
        gen_record["rul_days"] = 0.0

    return net, asset_registry


def simulate_weather_event(net, asset_registry, event_type: str = "HEATWAVE", severity: float = 1.0):
    """Applies a grid-wide weather stress scenario (HEATWAVE / STORM / COLD_SNAP)."""
    if event_type == "HEATWAVE":
        derate_factor = max(0.5, 1.0 - (0.10 * severity))
        net = derate_lines(net, derate_multiplier=derate_factor)
        for trafo_id in asset_registry["transformers"]:
            asset_registry = simulate_transformer_aging(asset_registry, trafo_id, severity=1.5 * severity)

    elif event_type == "STORM":
        derate_factor = max(0.3, 1.0 - (0.25 * severity))
        net = derate_lines(net, derate_multiplier=derate_factor)
        for cb_id in asset_registry["circuit_breakers"]:
            asset_registry = simulate_breaker_degradation(asset_registry, cb_id, friction_factor=2.0 * severity)

    elif event_type == "COLD_SNAP":
        net = apply_load_multiplier(net, multiplier=1.0 + (0.20 * severity))

    else:
        raise ValueError(f"Unknown weather event_type: '{event_type}'. Use HEATWAVE, STORM, or COLD_SNAP.")

    return net, asset_registry


def ramp_down_renewables(net, ramp_percent: float, renewable_gen_ids: list = None):
    """Curtails renewable generator output by ramp_percent (0-100)."""
    if not (0 <= ramp_percent <= 100):
        raise ValueError("ramp_percent must be between 0 and 100.")

    target_ids = renewable_gen_ids if renewable_gen_ids is not None else list(net.gen.index)

    for gid in target_ids:
        if gid in net.gen.index:
            current_mw = net.gen.at[gid, 'p_mw']
            net.gen.at[gid, 'p_mw'] = current_mw * (1 - ramp_percent / 100.0)

    return net