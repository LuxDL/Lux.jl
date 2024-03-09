using Documenter, DocumenterVitepress, Pkg
using Lux, LuxCore, LuxLib, WeightInitializers, Boltz
using LuxTestUtils, LuxDeviceUtils
using LuxAMDGPU, LuxCUDA

deployconfig = Documenter.auto_detect_deploy_system()
Documenter.post_status(deployconfig; type="pending", repo="github.com/LuxDL/Lux.jl.git")

pages = ["Lux.jl" => "index.md",
    "Ecosystem" => "ecosystem.md",
    "Getting Started" => [
        "Introduction" => "introduction/index.md", "Overview" => "introduction/overview.md",
        "Resources" => "introduction/resources.md",
        "Citation" => "introduction/citation.md"],
    "Tutorials" => ["Overview" => "tutorials/index.md",
        "Beginner" => [
            "tutorials/beginner/1_Basics", "tutorials/beginner/2_PolynomialFitting",
            "tutorials/beginner/3_SimpleRNN", "tutorials/beginner/4_SimpleChains"],
        "Intermediate" => ["tutorials/intermediate/5_NeuralODE",
            "tutorials/intermediate/6_BayesianNN", "tutorials/intermediate/7_HyperNet"],
        "Advanced" => ["tutorials/advanced/8_GravitationalWaveForm"]],
    "Manual" => [
        "manual/interface.md", "manual/debugging.md", "manual/dispatch_custom_input.md",
        "manual/freezing_model_parameters.md", "manual/gpu_management.md",
        "manual/migrate_from_flux.md", "manual/weight_initializers.md"],
    "API Reference" => ["index.md",
        "Lux" => ["api/Lux/layers.md", "api/Lux/utilities.md",
            "api/Lux/contrib.md", "api/Lux/switching_frameworks.md"],
        "Accelerator Support" => [
            "api/Accelerator_Support/LuxAMDGPU.md", "api/Accelerator_Support/LuxCUDA.md",
            "api/Accelerator_Support/LuxDeviceUtils.md"],
        "Building Blocks" => [
            "api/Building_Blocks/LuxCore.md", "api/Building_Blocks/LuxLib.md",
            "api/Building_Blocks/WeightInitializers.md"],
        "Domain Specific Modeling" => ["api/Domain_Specific_Modeling/Boltz.md"],
        "Testing Functionality" => ["api/Testing/LuxTestUtils.md"]]]

makedocs(; sitename="Lux",
    authors="Avik Pal et al.",
    clean=true,
    doctest=true,
    modules=[Lux, LuxCore, LuxLib, WeightInitializers, Boltz,
        LuxTestUtils, LuxDeviceUtils, LuxAMDGPU, LuxCUDA],
    linkcheck=true,
    linkcheck_ignore=["https://turing.ml/stable/",
        "https://turing.ml/stable/tutorials/03-bayesian-neural-network/"],
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="github.com/LuxDL/Lux.jl", devbranch="dev", deploy_url="lux.csail.mit.edu"),
    draft=true,  # FIXME
    source="src",
    build=joinpath(@__DIR__, "build"),
    pages=pages)

deploydocs(; repo="github.com/LuxDL/Lux.jl.git",
    push_preview=true, target="build", devbranch="main")
