include("./DocumenterVitepress/DocumenterVitepress.jl")
using .DocumenterVitepress

using Documenter, Pkg
using Lux, LuxCore, LuxLib, WeightInitializers, Boltz
using LuxTestUtils, LuxDeviceUtils
using LuxAMDGPU, LuxCUDA

deployconfig = Documenter.auto_detect_deploy_system()
Documenter.post_status(deployconfig; type="pending", repo="github.com/LuxDL/Lux.jl.git")

makedocs(; sitename="Lux", authors="Avik Pal et al.", clean=true, doctest=true,
    modules=[Lux, LuxCore, LuxLib, WeightInitializers, Boltz, LuxTestUtils, LuxDeviceUtils,
        LuxAMDGPU, LuxCUDA], checkdocs=:all, format=DocumenterVitepress.MarkdownVitepress(),
    draft=false, strict=[:doctest, :linkcheck, :parse_error, :example_block, :missing_docs],
    source="src", build=joinpath(@__DIR__, "build"))

deploydocs(; repo="github.com/LuxDL/Lux.jl.git", push_preview=true,
    target="page/.vitepress/dist", devbranch="main")
