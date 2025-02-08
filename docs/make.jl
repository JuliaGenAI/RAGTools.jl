using RAGTools
using Documenter

DocMeta.setdocmeta!(RAGTools, :DocTestSetup, :(using RAGTools); recursive = true)

makedocs(;
    modules = [RAGTools],
    authors = "J S <49557684+svilupp@users.noreply.github.com> and contributors",
    sitename = "RAGTools.jl",
    format = Documenter.HTML(;
        canonical = "https://svilupp.github.io/RAGTools.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "Example" => "example.md",
        "Interface" => "interface.md",
        "API Reference" => "api_reference.md"
    ]
)

deploydocs(;
    repo = "github.com/svilupp/RAGTools.jl",
    devbranch = "main"
)
