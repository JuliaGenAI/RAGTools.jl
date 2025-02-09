using RAG
using Test
using SparseArrays, LinearAlgebra, Unicode, Random
using PromptingTools
using PromptingTools.AbstractTrees
using Snowball
using JSON3, HTTP
using Aqua
const PT = PromptingTools
const RT = RAG

@testset "RAG.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(RAG)
    end
    @testset "Core" begin
        include("utils.jl")
        include("types.jl")
        include("preparation.jl")
        include("rank_gpt.jl")
        include("retrieval.jl")
        include("generation.jl")
        include("annotation.jl")
        include("evaluation.jl")
    end
end
