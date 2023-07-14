
using MLJBase

import Base: show

struct ModelPrinter{M<:MDT.SymbolicModel,SM<:SoleModels.AbstractModel}
    m::MLJBase.Model
    model::M
    solemodel::SM
    var_grouping::Union{Nothing,AbstractVector{<:AbstractVector},AbstractVector{<:AbstractDict}}
end
function (c::ModelPrinter)(args...; kwargs...)
    c(stdout, args...; kwargs...)
end
# Do not remove (generates compile-time warnings)
function (c::ModelPrinter)(io::IO; kwargs... )
    c(io, true, c.m.display_depth; kwargs...)
end
function (c::ModelPrinter)(
    io::IO,
    max_depth::Union{Nothing,Integer};
    kwargs...
)
    c(io, true, max_depth = max_depth; kwargs...)
end
function (c::ModelPrinter)(
    io::IO,
    print_solemodel::Bool,
    max_depth::Union{Nothing,Integer} = c.m.display_depth;
    kwargs...
)
    c(io, (print_solemodel ? c.solemodel : c.model); max_depth = max_depth, kwargs...)
end

function (c::ModelPrinter)(
    io::IO,
    model,
    X = nothing,
    y = nothing;
    max_depth = c.m.display_depth,
    hidemodality = (isnothing(c.var_grouping) || length(c.var_grouping) == 1),
    kwargs...
)
    more_kwargs = begin
        if model isa Union{MDT.DForest,MDT.DTree,MDT.DTNode}
            (; variable_names_map = c.var_grouping, max_depth = max_depth)
        elseif model isa SoleModels.AbstractModel
            (; max_depth = max_depth, syntaxstring_kwargs = (variable_names_map = c.var_grouping, hidemodality = hidemodality))
        else
            error("Unexpected model type $(model)")
        end
    end
    if isnothing(X) && isnothing(y)
        MDT.printmodel(io, model; more_kwargs..., kwargs...)
    elseif !isnothing(X) && !isnothing(y)
        (X, y, var_grouping, classes_seen) = MMI.reformat(c.m, X, y)
        MDT.printapply(io, model, X, y; more_kwargs..., kwargs...)
    else
        error("ModelPrinter: Either provide X and y or don't!")
    end
end


Base.show(io::IO, c::ModelPrinter) = print(io, "ModelPrinter object")
