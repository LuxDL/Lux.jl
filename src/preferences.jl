# Macro to deprecate a preference
macro deprecate_preference(old_pref, new_pref, default)
    msg1 = "Preference `$(old_pref)` is deprecated and will be removed in a future \
            release. Use `$(new_pref)` instead."
    msg2 = "Both `$(old_pref)` and `$(new_pref)` preferences are set. Please remove \
            `$(old_pref)`."
    return esc(quote
        if has_preference($(__module__), $(old_pref))
            Base.depwarn($msg1, $(Meta.quot(Symbol(__module__))))
            has_preference($(__module__), $(new_pref)) && error($msg2)
            load_preference($(__module__), $(old_pref), $(default))
        else
            load_preference($(__module__), $(new_pref), $(default))
        end
    end)
end

macro load_preference_with_choices(pref, default, choices)
    msg1 = "Invalid value for `$(pref)` preference: "
    msg2 = ". Valid choices are: $(choices)"
    return esc(quote
        val = load_preference($(__module__), $(pref), $(default))
        val ∉ $(choices) && error($(msg1) * string(val) * $(msg2))
        val
    end)
end

# Nested AD
const AUTOMATIC_NESTED_AD_SWITCHING = @deprecate_preference("DisableAutomaticNestedADSwitching",
    "automatic_nested_ad_switching", true)

# GPU-Aware MPI
const MPI_CUDA_AWARE = @deprecate_preference("LuxDistributedMPICUDAAware", "cuda_aware_mpi",
    false)
const MPI_ROCM_AWARE = @deprecate_preference("LuxDistributedMPIROCMAware", "rocm_aware_mpi",
    false)

# Eltype Auto Conversion
const ELTYPE_MISMATCH_HANDLING = @load_preference_with_choices("eltype_mismatch_handling",
    "none", ("none", "warn", "convert", "error"))
