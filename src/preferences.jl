module LuxPreferences

using Preferences: load_preference, has_preference, set_preferences!
using UUIDs: UUID

using ..Lux: Lux

const LuxUUID = UUID("b2108857-7c20-44ae-9111-449ecde12c47")

# Nested AD
const AUTOMATIC_NESTED_AD_SWITCHING = load_preference(
    LuxUUID, "automatic_nested_ad_switching", true
)

# GPU-Aware MPI
const MPI_CUDA_AWARE = load_preference(LuxUUID, "cuda_aware_mpi", false)
const MPI_ROCM_AWARE = load_preference(LuxUUID, "rocm_aware_mpi", false)

# Eltype Auto Conversion
const ELTYPE_MISMATCH_HANDLING = load_preference(
    LuxUUID, "eltype_mismatch_handling", "none"
)

function __init__()
    if ELTYPE_MISMATCH_HANDLING ∉ ("none", "warn", "convert", "error")
        error(
            "Invalid value for `eltype_mismatch_handling` preference: ",
            ELTYPE_MISMATCH_HANDLING,
            ". Valid choices are: (\"none\", \"warn\", \"convert\", \"error\")",
        )
    end
end

# Dispatch Doctor
function set_dispatch_doctor_preferences!(package, mode::String)
    @assert mode in ("disable", "warn", "error") "Invalid value for `mode`: $mode. Valid \
                                                  choices are: (\"disable\", \"warn\", \
                                                  \"error\")"
    if has_preference(package, "dispatch_doctor")
        orig_pref = load_preference(package, "dispatch_doctor")
        if orig_pref == mode
            @info "Dispatch Doctor preference for $(package) is already set to $mode."
            return nothing
        end
    end
    set_preferences!(package, "instability_check" => mode; force=true)
    @info "Dispatch Doctor preference for $(package) set to $mode. Please restart Julia \
           for this change to take effect."
    return nothing
end

end

# Dispatch Doctor
"""
    set_dispatch_doctor_preferences!(mode::String)
    set_dispatch_doctor_preferences!(; luxcore::String="disable", luxlib::String="disable")

Set the dispatch doctor preference for `LuxCore` and `LuxLib` packages.

`mode` can be `"disable"`, `"warn"`, or `"error"`. For details on the different modes, see
the [DispatchDoctor.jl](https://astroautomata.com/DispatchDoctor.jl/dev/) documentation.

If the preferences are already set, then no action is taken. Otherwise the preference is
set. For changes to take effect, the Julia session must be restarted.
"""
function set_dispatch_doctor_preferences!(mode::String)
    return set_dispatch_doctor_preferences!(; luxcore=mode, luxlib=mode)
end

function set_dispatch_doctor_preferences!(;
    luxcore::String="disable", luxlib::String="disable"
)
    LuxPreferences.set_dispatch_doctor_preferences!(LuxCore, luxcore)
    LuxPreferences.set_dispatch_doctor_preferences!(LuxLib, luxlib)
    return nothing
end
