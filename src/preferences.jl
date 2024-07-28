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
const AUTOMATIC_NESTED_AD_SWITCHING = @load_preference("automatic_nested_ad_switching",
    true)

# GPU-Aware MPI
const MPI_CUDA_AWARE = @load_preference("cuda_aware_mpi", false)
const MPI_ROCM_AWARE = @load_preference("rocm_aware_mpi", false)

# Eltype Auto Conversion
const ELTYPE_MISMATCH_HANDLING = @load_preference_with_choices("eltype_mismatch_handling",
    "none", ("none", "warn", "convert", "error"))
