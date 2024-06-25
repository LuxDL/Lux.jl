# Macro to deprecate a preference
macro deprecate_preference(old_pref, new_pref, default)
    msg1 = "Preference `$(old_pref)` is deprecated and will be removed in a future \
            release. Use `$(new_pref)` instead."
    msg2 = "Both `$(old_pref)` and `$(new_pref)` preferences are set. Please remove \
            `$(old_pref)`."
    return esc(quote
        old_pref_val = load_preference($(__module__), $(old_pref), missing)
        if old_pref_val !== missing
            Base.depwarn($msg1, $(Meta.quot(Symbol(__module__))))
            new_pref_val = load_preference($(__module__), $(new_pref), missing)
            new_pref_val !== missing && error($msg2)
            old_pref_val
        else
            load_preference($(__module__), $(new_pref), $(default))
        end
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
