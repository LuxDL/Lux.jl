module LuxPreferences

using ArgCheck: @argcheck
using Preferences: load_preference, has_preference, set_preferences!, @load_preference

using ..Lux: Lux

macro load_preference_with_choices(pref, default, choices)
    msg1 = "Invalid value for `$(pref)` preference: "
    msg2 = ". Valid choices are: $(choices)"
    return esc(quote
        val = load_preference($(Lux), $(pref), $(default))
        val ∉ $(choices) && error($(msg1) * string(val) * $(msg2))
        val
    end)
end

# Nested AD
const AUTOMATIC_NESTED_AD_SWITCHING = @load_preference("automatic_nested_ad_switching",
    true)

# GPU-Aware MPI
const MPI_CUDA_AWARE = @load_preference("cuda_aware_mpi", false)
const MPI_ROCM_AWARE = @load_preference("rocm_aware_mpi", false)

# Eltype Auto Conversion
const ELTYPE_MISMATCH_HANDLING = @load_preference_with_choices("eltype_mismatch_handling",
    "none", ("none", "warn", "convert", "error"))

# Dispatch Doctor
function set_dispatch_doctor_preferences!(package, mode::String)
    @argcheck mode in ("disable", "warn", "error")
    if has_preference(package, "dispatch_doctor")
        orig_pref = load_preference(package, "dispatch_doctor")
        if orig_pref == mode
            @info "Dispatch Doctor preference for $(package) is already set to $mode."
            return
        end
    end
    set_preferences!(package, "instability_check" => mode; force=true)
    @info "Dispatch Doctor preference for $(package) set to $mode. Please restart Julia \
           for this change to take effect."
    return
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
        luxcore::String="disable", luxlib::String="disable")
    LuxPreferences.set_dispatch_doctor_preferences!(LuxCore, luxcore)
    LuxPreferences.set_dispatch_doctor_preferences!(LuxLib, luxlib)
    return
end
