module StatsBaseExt

using InfluenceFunctions: InfluenceFunctions
using Statistics: Statistics
using StatsBase: StatsBase
using StatsFuns: StatsFuns

include("weights.jl")
include("moments.jl")
include("scalarstats.jl")
include("cov.jl")
include("counts.jl")
include("partialcor.jl")

end
