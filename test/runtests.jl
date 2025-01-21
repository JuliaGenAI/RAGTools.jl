using RAGTools
using Test
using Aqua

@testset "RAGTools.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(RAGTools)
    end
    # No tests yet
end
