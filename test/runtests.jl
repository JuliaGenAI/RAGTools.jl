using RAGTools
using Test
using SparseArrays, LinearAlgebra, Unicode, Random
using PromptingTools
using PromptingTools.AbstractTrees
using Snowball
using JSON3, HTTP
using Aqua
const PT = PromptingTools
const RT = RAGTools

# Set a dummy API key for tests using CustomOpenAISchema with mock HTTP servers
# Required since OpenAI.jl now throws an error when api_key is empty
# We set the PT global directly since it's loaded at module load time from preferences/ENV
if isempty(PT.OPENAI_API_KEY)
    PT.OPENAI_API_KEY = "test-api-key-for-mock-server"
end

@testset "RAGTools.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(RAGTools)
    end
    @testset "Core" begin
        include("utils.jl")
        include("types/candidate_chunks.jl")
        include("types/document_term_matrix.jl")
        include("types/index.jl")
        include("types/rag_result.jl")
        include("preparation.jl")
        include("rank_gpt.jl")
        include("retrieval.jl")
        include("generation.jl")
        include("annotation.jl")
        include("evaluation.jl")
    end
end
