using Documenter, DocumenterVitepress, Pkg
using Lux, LuxCore, LuxLib, WeightInitializers, Boltz
using LuxTestUtils, LuxDeviceUtils
using LuxAMDGPU, LuxCUDA

deployconfig = Documenter.auto_detect_deploy_system()
Documenter.post_status(deployconfig; type="pending", repo="github.com/LuxDL/Lux.jl.git")

makedocs(; sitename="Lux",
    authors="Avik Pal et al.",
    clean=true,
    doctest=true,
    modules=[Lux, LuxCore, LuxLib, WeightInitializers, Boltz,
        LuxTestUtils, LuxDeviceUtils, LuxAMDGPU, LuxCUDA],
    linkcheck=true,
    linkcheck_ignore=["https://turing.ml/stable/",
        "https://turing.ml/stable/tutorials/03-bayesian-neural-network/"],
    format=DocumenterVitepress.MarkdownVitepress(; repo = "github.com/LuxDL/Lux.jl", devbranch = "dev", deploy_url = "lux.csail.mit.edu"),
    draft=false,
    source="src",
    build=joinpath(@__DIR__, "build"))

deploydocs(; repo="github.com/LuxDL/Lux.jl.git",
    push_preview=true, target="build", devbranch="main")
