using Documenter, DocumenterVitepress, Pkg
using Lux, LuxCore, LuxLib, WeightInitializers, Boltz
using LuxTestUtils, LuxDeviceUtils
using LuxCUDA

#! format: off

pages = [
    "Lux.jl" => "index.md",
    "Ecosystem" => "ecosystem.md",
    "Getting Started" => [
        "Introduction" => "introduction/index.md",
        "Overview" => "introduction/overview.md",
        "Resources" => "introduction/resources.md",
        "Citation" => "introduction/citation.md"
    ],
    "Tutorials" => [
        "Overview" => "tutorials/index.md",
        "Beginner" => [
            "tutorials/beginner/1_Basics.md",
            "tutorials/beginner/2_PolynomialFitting.md",
            "tutorials/beginner/3_SimpleRNN.md",
            "tutorials/beginner/4_SimpleChains.md"
        ],
        "Intermediate" => [
            "tutorials/intermediate/1_NeuralODE.md",
            "tutorials/intermediate/2_BayesianNN.md",
            "tutorials/intermediate/3_HyperNet.md"
        ],
        "Advanced" => [
            "tutorials/advanced/1_GravitationalWaveForm.md"
        ]
    ],
    "Manual" => [
        "manual/interface.md",
        "manual/debugging.md",
        "manual/dispatch_custom_input.md",
        "manual/freezing_model_parameters.md",
        "manual/gpu_management.md",
        "manual/migrate_from_flux.md",
        "manual/weight_initializers.md",
        "manual/distributed_utils.md",
        "manual/nested_autodiff.md"
    ],
    "API Reference" => [
        "Lux" => [
            "api/Lux/layers.md",
            "api/Lux/autodiff.md",
            "api/Lux/utilities.md",
            "api/Lux/contrib.md",
            "api/Lux/interop.md",
            "api/Lux/distributed_utils.md",
        ],
        "Accelerator Support" => [
            "api/Accelerator_Support/LuxCUDA.md",
            "api/Accelerator_Support/LuxDeviceUtils.md"
        ],
        "Building Blocks" => [
            "api/Building_Blocks/LuxCore.md",
            "api/Building_Blocks/LuxLib.md",
            "api/Building_Blocks/WeightInitializers.md"
        ],
        "Domain Specific Modeling" => [
            "api/Domain_Specific_Modeling/Boltz.md",
            "api/Domain_Specific_Modeling/Boltz_Layers.md",
            "api/Domain_Specific_Modeling/Boltz_Vision.md",
            "api/Domain_Specific_Modeling/Boltz_Private.md"
        ],
        "Testing Functionality" => [
            "api/Testing_Functionality/LuxTestUtils.md"
        ]
    ]
]

#! format: on

deploy_config = Documenter.auto_detect_deploy_system()
deploy_decision = Documenter.deploy_folder(deploy_config; repo="github.com/LuxDL/Lux.jl",
    devbranch="main", devurl="dev", push_preview=true)

makedocs(; sitename="Lux.jl Documentation",
    authors="Avik Pal et al.",
    clean=true,
    doctest=false,  # We test it in the CI, no need to run it here
    modules=[Lux, LuxCore, LuxLib, WeightInitializers,
        Boltz, LuxTestUtils, LuxDeviceUtils, LuxCUDA],
    linkcheck=true,
    repo="https://github.com/LuxDL/Lux.jl/blob/{commit}{path}#{line}",
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="github.com/LuxDL/Lux.jl", devbranch="main", devurl="dev",
        deploy_url="https://lux.csail.mit.edu", deploy_decision),
    draft=false,
    warnonly=:linkcheck,  # Lately it has been failing quite a lot but those links are actually fine
    pages)

deploydocs(; repo="github.com/LuxDL/Lux.jl.git",
    push_preview=true, target="build", devbranch="main")
