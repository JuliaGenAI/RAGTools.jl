using RAGTools
using Documenter

DocMeta.setdocmeta!(RAGTools, :DocTestSetup, :(using RAGTools); recursive = true)

makedocs(;
    modules = [RAGTools],
    authors = "J S <49557684+svilupp@users.noreply.github.com> and contributors",
    sitename = "RAGTools.jl",
    format = Documenter.HTML(;
        canonical = "https://github.com/JuliaGenAI/RAGTools.jl",
        edit_link = "main",
        assets = String[],
        size_threshold = 5 * 2^20
    ),
    pages = [
        "Home" => "index.md",
        "Example" => "example.md",
        "Interface" => "interface.md",
        "API Reference" => "api_reference.md"
    ]
)

deploydocs(;
    repo = "github.com/JuliaGenAI/RAGTools.jl",
    devbranch = "main"
)
