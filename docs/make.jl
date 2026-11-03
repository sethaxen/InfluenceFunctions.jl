using InfluenceFunctions
using Documenter

DocMeta.setdocmeta!(InfluenceFunctions, :DocTestSetup, :(using InfluenceFunctions); recursive = true)

makedocs(;
    modules = [InfluenceFunctions],
    authors = "Seth Axen <seth@sethaxen.com> and contributors",
    sitename = "InfluenceFunctions.jl",
    format = Documenter.HTML(;
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)
