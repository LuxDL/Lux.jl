using Documenter, DocumenterVitepress, Pkg
using Lux, LuxCore, LuxLib, WeightInitializers, NNlib
using LuxTestUtils, MLDataDevices
using LuxCUDA

using Optimisers # for some docstrings

#! format: off

pages = [
    "Lux.jl" => "index.md",
    "Getting Started" => [
        "Introduction" => "introduction/index.md",
        "Overview" => "introduction/overview.md",
        "Resources" => "introduction/resources.md",
        "Updating to v1" => "introduction/updating_to_v1.md",
        "Citation" => "introduction/citation.md"
    ],
    "Tutorials" => [
        "Overview" => "tutorials/index.md",
        "Beginner" => [
            "tutorials/beginner/1_Basics.md",
            "tutorials/beginner/2_PolynomialFitting.md",
            "tutorials/beginner/3_SimpleRNN.md",
            "tutorials/beginner/4_SimpleChains.md",
            "tutorials/beginner/5_OptimizationIntegration.md"
        ],
        "Intermediate" => [
            "tutorials/intermediate/1_NeuralODE.md",
            "tutorials/intermediate/2_BayesianNN.md",
            "tutorials/intermediate/3_HyperNet.md",
            "tutorials/intermediate/4_PINN2DPDE.md",
            "tutorials/intermediate/5_ConditionalVAE.md",
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
        "manual/nested_autodiff.md",
        "manual/compiling_lux_models.md",
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
            "api/Accelerator_Support/MLDataDevices.md"
        ],
        "NN Primitives" => [
            "api/NN_Primitives/LuxLib.md",
            "api/NN_Primitives/NNlib.md",
            "api/NN_Primitives/ActivationFunctions.md"
        ],
        "Building Blocks" => [
            "api/Building_Blocks/LuxCore.md",
            "api/Building_Blocks/WeightInitializers.md"
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

makedocs(;
    sitename="Lux.jl Docs",
    authors="Avik Pal et al.",
    clean=true,
    doctest=false,  # We test it in the CI, no need to run it here
    modules=[
        Lux, LuxCore, LuxLib, WeightInitializers, LuxTestUtils, MLDataDevices, NNlib
    ],
    linkcheck=true,
    linkcheck_ignore=[
        "http://www.iro.umontreal.ca/~lisa/publications2/index.php/attachments/single/205"
    ],
    repo="https://github.com/LuxDL/Lux.jl/blob/{commit}{path}#{line}",
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="github.com/LuxDL/Lux.jl", devbranch="main", devurl="dev",
        deploy_url="https://lux.csail.mit.edu", deploy_decision),
    draft=false,
    pages
)

deploydocs(;
    repo="github.com/LuxDL/Lux.jl.git",
    push_preview=true, target="build", devbranch="main"
)
