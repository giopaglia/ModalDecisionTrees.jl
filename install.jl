
import Pkg

## CONFIG

const _script_version = 1

const recommended_version = VersionNumber(1, 8, 3)

const _external_deps = [
]

const _external_deps_dev = [
    "SoleBase" => "https://github.com/aclai-lab/SoleBase.jl#dev",
    "SoleData" => "https://github.com/aclai-lab/SoleData.jl#dev",
    "SoleLogics" => "https://github.com/aclai-lab/SoleLogics.jl#dev",
    "SoleModels" => "https://github.com/aclai-lab/SoleModels.jl#dev",
]

const _python_deps = [
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "statsmodels",
    "matplotlib",
]

# binary to search => package name
const _system_deps = [
    "cmake" => "cmake",
    "python3" => "python + python-is-python3",
    "unzip" => "unzip"
]

## UTILS

function _question(q::AbstractString, a::Dict{Char,<:Any}, default::Char)
    while true
        print("$(q) [$(join([k == default ? uppercase(k) : k for k in keys(a)], '/'))] ")

        run(`stty raw`)
        u = read(stdin, Char)
        run(`stty cooked`)
        println()

        if u == '\x0D' # ENTER
            u = default
        end

        u = lowercase(u)

        if haskey(a, u)
            return a[u]()
        end
    end
end
function question(q::AbstractString, default::Pair{Char,<:Any}, a::Pair{Char,<:Any}...)
    return _question(q, Dict(lowercase(default[1]) => default[2], [lowercase(c[1]) => c[2] for c in a]...), default[1])
end

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

function install_python_dep(str::AbstractString; pip = "/usr/bin/pip3")
    try
        run(`pip3 install $str`)
    catch ex
        throw(ErrorException("Couldn't install python package `$str`"))
    end
end
function append_to_file(filename::AbstractString, s::AbstractString)
    f = open(filename, "a+")
    write(f, s)
    close(f)
end
function createfile(filename::AbstractString, s::AbstractString)
    f = open(filename, "w+")
    write(f, s)
    close(f)
end
function read_first_line(filename::AbstractString)
    f = open(filename)
    res = readline(f)
    close(f)
    return res
end

function is_missing_dep(dep::Pair{<:AbstractString,<:AbstractString})::Bool
    try
        strip(readchomp(`which $(dep[1])`))
        return false
    catch
        return true
    end
end
function filter_missing_deps(deps::AbstractVector{<:Pair{<:AbstractString,<:AbstractString}})
    filter(x -> !isnothing(x), [is_missing_dep(d) ? d : nothing for d in deps])
end

## CONTORLS

if VERSION != recommended_version
    @warn "Recommended version is $(recommended_version); currently using: $(VERSION)\n" *
        "Consider switching to version $(recommended_version)\n"

    question("Do you want to continue anyway?", 'y' => ()->nothing, 'n' => exit)
end

missing_deps = filter_missing_deps(_system_deps)
if length(missing_deps) > 0
    local msg = "Some system dependencies are missing: "
    local pkgs = ""
    for (i, (b, p)) in enumerate(missing_deps)
        msg *= string("`$(b)`", i != length(missing_deps) ? ", " : ".")
        pkgs = string("`$(p)`", i != length(missing_deps) ? ", " : ".")
    end
    msg *= "\nUsually they come in the packages: $(pkgs)"
    @warn msg

    question("Do you want to continue anyway?", 'y' => ()->nothing, 'n' => exit)
end

if !haskey(ENV, "PYTHON")
    python_path =
        try
            readchomp(`which python3`)
        catch ex
            ""
        end

    if length(python_path) == 0
        throw(ErrorException("Couldn't locate `python` executable"))
    end

    ENV["PYTHON"] = python_path

    @warn "`python` executable detected at $(python_path)"
end

pip_path =
    try
        readchomp(`which pip3`)
    catch ex
        ""
    end

if length(pip_path) == 0
    throw(ErrorException("Couldn't locate `pip3` executable"))
end

## START

println("Running script version: $(_script_version)")

Pkg.activate(dirname(PROGRAM_FILE))

safe_remove.(map(x -> x[1], [_external_deps..., _external_deps_dev...]))

length(_external_deps) == 0 || Pkg.add(to_package_spec.(_external_deps))

for (name, s) in _external_deps_dev
    Pkg.add(to_package_spec((name, s),))
    Pkg.develop(name)
end

Pkg.instantiate()
Pkg.resolve()
Pkg.precompile()

install_python_dep.(_python_deps; pip = pip_path)

Pkg.add(Pkg.PackageSpec(name="PyCall", rev="master"))
Pkg.build("PyCall")

append_to_file(".gitignore", "Project.toml\n")

global _current_version = 0

createfile(".env", string("# .env

# directory where the datasets are stored
DATA_DIR = ../datasets/
"))
