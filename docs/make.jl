using Documenter, DocumenterVitepress, Pkg
using Lux, LuxCore, LuxLib, WeightInitializers, Boltz
using LuxTestUtils, LuxDeviceUtils
using LuxAMDGPU, LuxCUDA

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
        "manual/weight_initializers.md"
    ],
    "API Reference" => [
        "Lux" => [
            "api/Lux/layers.md",
            "api/Lux/utilities.md",
            "api/Lux/contrib.md",
            "api/Lux/switching_frameworks.md"
        ],
        "Accelerator Support" => [
            "api/Accelerator_Support/LuxAMDGPU.md",
            "api/Accelerator_Support/LuxCUDA.md",
            "api/Accelerator_Support/LuxDeviceUtils.md"
        ],
        "Building Blocks" => [
            "api/Building_Blocks/LuxCore.md",
            "api/Building_Blocks/LuxLib.md",
            "api/Building_Blocks/WeightInitializers.md"
        ],
        "Domain Specific Modeling" => [
            "api/Domain_Specific_Modeling/Boltz.md"
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
    doctest=true,
    modules=[Lux, LuxCore, LuxLib, WeightInitializers, Boltz,
        LuxTestUtils, LuxDeviceUtils, LuxAMDGPU, LuxCUDA],
    linkcheck=true,
    linkcheck_ignore=["https://turing.ml/stable/",
        "https://turing.ml/stable/tutorials/03-bayesian-neural-network/"],
    repo="https://github.com/LuxDL/Lux.jl/blob/{commit}{path}#{line}",
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="github.com/LuxDL/Lux.jl", devbranch="main", devurl="dev",
        deploy_url="https://lux.csail.mit.edu", deploy_decision),
    draft=false,
    pages=pages)

deploydocs(; repo="github.com/LuxDL/Lux.jl.git",
    push_preview=true, target="build", devbranch="main")
