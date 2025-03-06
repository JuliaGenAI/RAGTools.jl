# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Fixed

## [0.3.0]

### Added
- Added `ReciprocalRankFusionReranker` and associated `rerank` method for hybrid retrieval (MultiIndex with embeddings and keywords indices referring to the same chunks).

## [0.2.1]

### Fixed
- Fixed `find_closest` to pass kwargs to `bm25` to allow for normalization of scores
- Fixed a bug in `ChunkEmbeddingsIndex` where users couldn't create a bitpacked index with `embeddings` of type `BitMatrix` (to use `finder=BitPackedCosineSimilarity()`)

## [0.2.0]

### Added
- Progress bar for `get_embeddings` given it can take a while for large documents (new dependency `ProgressMeter.jl`, but extremely lightweight). Make sure to set kwargs to `embedder_kwargs=(;verbose=true)` to see it.

### Updated
- Increased PromptingTools compat to v0.73

## [0.1.1]

### Fixed
- Fixed `preprocess_tokens`, `get_tags`, and `get_embeddings` to not trigger package extension checks (leftover from carve out from PromptingTools)
- Clean up docs references to `PromptingTools.Experimental.RAGTools`

## [0.1.0]

### Added
- Initial release of RAGTools.jl, simple carve-out of module RAGTools.jl from PromptingTools.jl.
