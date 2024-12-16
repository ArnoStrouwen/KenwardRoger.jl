using KenwardRoger
using Documenter

DocMeta.setdocmeta!(KenwardRoger, :DocTestSetup, :(using KenwardRoger); recursive=true)

makedocs(;
    modules=[KenwardRoger],
    authors="Arno Strouwen",
    sitename="KenwardRoger.jl",
    format=Documenter.HTML(;
        canonical="https://ArnoStrouwen.github.io/KenwardRoger.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/ArnoStrouwen/KenwardRoger.jl", devbranch="master")
