using RAG
using Documenter

DocMeta.setdocmeta!(RAG, :DocTestSetup, :(using RAG); recursive = true)

makedocs(;
    modules = [RAG],
    authors = "J S <49557684+svilupp@users.noreply.github.com> and contributors",
    sitename = "RAG.jl",
    format = Documenter.HTML(;
        canonical = "https://github.com/JuliaGenAI/RAG.jl",
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
    repo = "github.com/JuliaGenAI/RAG.jl",
    devbranch = "main"
)
