
using Pkg

## CONFIG

const _external_deps_dev = [
    "SoleBase" => "https://github.com/aclai-lab/SoleBase.jl#dev",
    "SoleData" => "https://github.com/aclai-lab/SoleData.jl#dev",
    "SoleLogics" => "https://github.com/aclai-lab/SoleLogics.jl#dev",
    "SoleModels" => "https://github.com/aclai-lab/SoleModels.jl#dev",
]

## UTILS

function to_package_spec(p::Union{Pair{<:AbstractString,<:AbstractString},Tuple{<:AbstractString,<:AbstractString}})
    name, s = p
    if occursin("#", s)
        url, rev = split(s, '#')
        Pkg.PackageSpec(;url = url, rev = rev)
    else
        Pkg.PackageSpec(;url = s)
    end
end
function safe_remove(p::AbstractString)
    try
        Pkg.rm(Pkg.PackageSpec(p))
    catch ex
    end
end

## START
Pkg.activate(dirname(PROGRAM_FILE))

safe_remove.(map(x -> x[1], [_external_deps_dev...]))

for (name, s) in _external_deps_dev
    Pkg.add(to_package_spec((name, s),))
    Pkg.develop(name)
end

function append_to_file(filename::AbstractString, s::AbstractString)
    f = open(filename, "a+")
    write(f, s)
    close(f)
end

append_to_file(".gitignore", "Project.toml\n")

Pkg.instantiate()
Pkg.resolve()
Pkg.precompile()
