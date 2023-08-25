import Documenter: Anchors, Builder, Documents, Expanders, Documenter, Utilities

import ANSIColoredPrinters

# import Markdown as MarkdownStdlib
module _Markdown
import Markdown
end
const MarkdownStdlib = _Markdown.Markdown

struct MarkdownVitepress <: Documenter.Writer
end

# return the same file with the extension changed to .md
mdext(f) = string(splitext(f)[1], ".md")

function render(doc::Documents.Document, settings::MarkdownVitepress=MarkdownVitepress())
    @info "DocumenterMarkdownVitepress: rendering MarkdownVitepress pages."
    copy_assets(doc)
    mime = MIME"text/plain"()
    for (src, page) in doc.blueprint.pages
        open(mdext(page.build), "w") do io
            for elem in page.elements
                node = page.mapping[elem]
                render(io, mime, node, page, doc)
            end
        end
    end
end

function copy_assets(doc::Documents.Document)
    @debug "copying assets to build directory."
    assets = ASSETS
    if isdir(assets)
        builddir = joinpath(doc.user.build, "assets")
        isdir(builddir) || mkdir(builddir)
        for each in readdir(assets)
            src = joinpath(assets, each)
            dst = joinpath(builddir, each)
            ispath(dst) && @warn "DocumenterMarkdownVitepress: overwriting '$dst'."
            cp(src, dst; force=true)
        end
    else
        @warn "DocumenterMarkdownVitepress: no assets directory found."
    end
end

function render(io::IO, mime::MIME"text/plain", vec::Vector, page, doc)
    for each in vec
        render(io, mime, each, page, doc)
    end
end

function render(io::IO, mime::MIME"text/plain", anchor::Anchors.Anchor, page, doc)
    println(io, "\n<a id='", lstrip(Anchors.fragment(anchor), '#'), "'></a>")
    return render(io, mime, anchor.object, page, doc)
end

## Documentation Nodes.

function render(io::IO, mime::MIME"text/plain", node::Documents.DocsNodes, page, doc)
    for node in node.nodes
        render(io, mime, node, page, doc)
    end
end

function render(io::IO, mime::MIME"text/plain", node::Documents.DocsNode, page, doc)
    # Docstring header based on the name of the binding and it's category.
    println(io,
        "<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>")
    anchor = "<a id='$(node.anchor.id)' href='#$(node.anchor.id)'>#</a>"
    header = "&nbsp;<b><u>$(node.object.binding)</u></b> &mdash; <i>$(Utilities.doccat(node.object))</i>."
    println(io, anchor, header, "\n\n")
    # Body. May contain several concatenated docstrings.
    renderdoc(io, mime, node.docstr, page, doc)
    return println(io, "</div>\n<br>")
end

function renderdoc(io::IO, mime::MIME"text/plain", md::MarkdownStdlib.MD, page, doc)
    if haskey(md.meta, :results)
        # The `:results` field contains a vector of `Docs.DocStr` objects associated with
        # each markdown object. The `DocStr` contains data such as file and line info that
        # we need for generating correct source links.
        for (markdown, result) in zip(md.content, md.meta[:results])
            render(io, mime, dropheaders(markdown), page, doc)
            # When a source link is available then print the link.
            url = Utilities.url(doc.internal.remote, doc.user.repo, result)
            if url !== nothing
                link = "<a target='_blank' href='$url' class='documenter-source'>source</a><br>"
                println(io, "\n", link, "\n")
            end
        end
    else
        # Docstrings with no `:results` metadata won't contain source locations so we don't
        # try to print them out. Just print the basic docstring.
        render(io, mime, dropheaders(md), page, doc)
    end
end

function renderdoc(io::IO, mime::MIME"text/plain", other, page, doc)
    # TODO: properly support non-markdown docstrings at some point.
    return render(io, mime, other, page, doc)
end

## Index, Contents, and Eval Nodes.

function render(io::IO, ::MIME"text/plain", index::Documents.IndexNode, page, doc)
    for (object, _, page, mod, cat) in index.elements
        page = mdext(page)
        url = string("#", Utilities.slugify(object))
        println(io, "- [`", object.binding, "`](", url, ")")
    end
    return println(io)
end

function render(io::IO, ::MIME"text/plain", contents::Documents.ContentsNode, page, doc)
    for (count, path, anchor) in contents.elements
        path = mdext(path)
        header = anchor.object
        url = string(path, Anchors.fragment(anchor))
        link = MarkdownStdlib.Link(header.text, url)
        level = Utilities.header_level(header)
        print(io, "    "^(level - 1), "- ")
        MarkdownStdlib.plaininline(io, link)
        println(io)
    end
    return println(io)
end

function render(io::IO, mime::MIME"text/plain", node::Documents.EvalNode, page, doc)
    return node.result === nothing ? nothing : render(io, mime, node.result, page, doc)
end

function render(io::IO, mime::MIME"text/plain", mcb::Documents.MultiCodeBlock, page, doc)
    return render(io, mime, Documents.join_multiblock(mcb), page, doc)
end

# Select the "best" representation for Markdown output.
using Base64: base64decode
function render(io::IO, mime::MIME"text/plain", d::Documents.MultiOutput, page, doc)
    return foreach(x -> Base.invokelatest(render, io, mime, x, page, doc), d.content)
end
function render(io::IO, mime::MIME"text/plain", d::Dict{MIME, Any}, page, doc)
    filename = String(rand('a':'z', 7))
    if haskey(d, MIME"text/markdown"())
        println(io, d[MIME"text/markdown"()])
    elseif haskey(d, MIME"text/html"())
        println(io, d[MIME"text/html"()])
    elseif haskey(d, MIME"image/svg+xml"())
        # NOTE: It seems that we can't simply save the SVG images as a file and include them
        # as browsers seem to need to have the xmlns attribute set in the <svg> tag if you
        # want to include it with <img>. However, setting that attribute is up to the code
        # creating the SVG image.
        println(io, d[MIME"image/svg+xml"()])
    elseif haskey(d, MIME"image/png"())
        write(joinpath(dirname(page.build), "$(filename).png"),
            base64decode(d[MIME"image/png"()]))
        println(io,
            """
    ![]($(filename).png)
    """)
    elseif haskey(d, MIME"image/webp"())
        write(joinpath(dirname(page.build), "$(filename).webp"),
            base64decode(d[MIME"image/webp"()]))
        println(io,
            """
    ![]($(filename).webp)
    """)
    elseif haskey(d, MIME"image/jpeg"())
        write(joinpath(dirname(page.build), "$(filename).jpeg"),
            base64decode(d[MIME"image/jpeg"()]))
        println(io,
            """
    ![]($(filename).jpeg)
    """)
    elseif haskey(d, MIME"image/gif"())
        write(joinpath(dirname(page.build), "$(filename).gif"),
            base64decode(d[MIME"image/gif"()]))
        println(io,
            """
    ![]($(filename).gif)
    """)
    elseif haskey(d, MIME"text/plain"())
        text = d[MIME"text/plain"()]
        out = repr(MIME"text/plain"(), ANSIColoredPrinters.PlainTextPrinter(IOBuffer(text)))
        render(io, mime, MarkdownStdlib.Code(out), page, doc)
    else
        error("this should never happen.")
    end
    return nothing
end

## Basic Nodes. AKA: any other content that hasn't been handled yet.

function render(io::IO, ::MIME"text/plain", other, page, doc)
    println(io)
    linkfix = ".md#"
    return println(io, replace(MarkdownStdlib.plain(other), linkfix => "#"))
end

render(io::IO, ::MIME"text/plain", str::AbstractString, page, doc) = print(io, str)

# Metadata Nodes get dropped from the final output for every format but are needed throughout
# rest of the build and so we just leave them in place and print a blank line in their place.
render(io::IO, ::MIME"text/plain", node::Documents.MetaNode, page, doc) = println(io, "\n")

function render(io::IO, ::MIME"text/plain", raw::Documents.RawNode, page, doc)
    return raw.name === :html ? println(io, raw.text, "\n") : nothing
end

## Markdown Utilities.

# Remove all header nodes from a markdown object and replace them with bold font.
# Only for use in `text/plain` output, since we'll use some css to make these less obtrusive
# in the HTML rendering instead of using this hack.
function dropheaders(md::MarkdownStdlib.MD)
    out = MarkdownStdlib.MD()
    out.meta = md.meta
    out.content = map(dropheaders, md.content)
    return out
end
function dropheaders(h::MarkdownStdlib.Header)
    return MarkdownStdlib.Paragraph([MarkdownStdlib.Bold(h.text)])
end
dropheaders(v::Vector) = map(dropheaders, v)
dropheaders(other) = other
