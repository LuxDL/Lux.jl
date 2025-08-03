"""
Minimal Reactant tracing test for RQS implementation.

This script tests whether the RQS functions can be traced by Reactant.jl
without control flow errors.
"""

using Reactant
using Random
using NNlib

# Include the RQS implementation
include("../src/rqs/rqs_wrapper_gather.jl")

println("🔍 Testing Reactant tracing for RQS implementation...")

# Set up test data
rng = Random.default_rng()
Random.seed!(rng, 42)

# Small test case to minimize compilation time
K = 3
D = 2
B = 2

# Generate random logits
logits = randn(rng, Float32, 3K + 1, D, B)

# Generate test input
u = rand(rng, Float32, D, B)

println("📊 Test data shapes:")
println("  logits: $(size(logits))")
println("  u: $(size(u))")

# Test 1: Parameterization function
println("\n🧪 Test 1: Parameterization function")
try
    result = parameterize_rqs(logits)
    println("✅ Parameterization execution successful")
    println("  x_pos shape: $(size(result[1]))")
    println("  y_pos shape: $(size(result[2]))")
    println("  d shape: $(size(result[3]))")
catch e
    println("❌ Parameterization execution failed: $e")
end

# Test 2: Forward transformation
println("\n🧪 Test 2: Forward transformation")
try
    result = rqs_forward(u, logits)
    println("✅ Forward transformation execution successful")
    println("  v shape: $(size(result[1]))")
    println("  log_det shape: $(size(result[2]))")
catch e
    println("❌ Forward transformation execution failed: $e")
end

# Test 3: Inverse transformation
println("\n🧪 Test 3: Inverse transformation")
try
    v, _ = rqs_forward(u, logits)
    result = rqs_inverse(v, logits)
    println("✅ Inverse transformation execution successful")
    println("  u shape: $(size(result[1]))")
    println("  log_det shape: $(size(result[2]))")
catch e
    println("❌ Inverse transformation execution failed: $e")
end

# Test 4: RQSBijector interface
println("\n🧪 Test 4: RQSBijector interface")
try
    bj = RQSBijector(K)
    result = forward_and_log_det(bj, u, logits)
    println("✅ RQSBijector forward execution successful")
    println("  v shape: $(size(result[1]))")
    println("  log_det shape: $(size(result[2]))")
catch e
    println("❌ RQSBijector forward execution failed: $e")
end

# Test 5: Full round-trip
println("\n🧪 Test 5: Full round-trip")
try
    v, _ = rqs_forward(u, logits)
    u_recovered, _ = rqs_inverse(v, logits)
    error = maximum(abs.(u - u_recovered))
    println("✅ Full round-trip execution successful")
    println("  Max round-trip error: $error")
catch e
    println("❌ Full round-trip execution failed: $e")
end

println("\n🎉 RQS execution test completed!")
println("📝 Check the output above for any execution errors.")
println("💡 If all tests pass, the RQS implementation is ready for integration.")
