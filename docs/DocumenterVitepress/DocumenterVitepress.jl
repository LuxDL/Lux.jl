"""
    DocumenterVitepress

Similar to DocumentationMarkdown.jl but designed to work with
[vitepress](https://vitepress.dev/).
"""
module DocumenterVitepress

using Documenter: Documenter
using Documenter.Utilities: Selectors

const ASSETS = normpath(joinpath(@__DIR__, "..", "assets"))

include("writer.jl")

export MarkdownVitepress

# Selectors interface in Documenter.Writers, for dispatching on different writers
abstract type MarkdownFormat <: Documenter.Writers.FormatSelector end

Selectors.order(::Type{MarkdownFormat}) = 0.0
Selectors.matcher(::Type{MarkdownFormat}, fmt, _) = isa(fmt, MarkdownVitepress)
Selectors.runner(::Type{MarkdownFormat}, fmt, doc) = render(doc, fmt)

end
