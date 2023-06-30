
import Base: show

struct ModelPrinter{M<:MDT.SymbolicModel,SM<:SoleModels.AbstractModel}
    model::M
    solemodel::SM
    var_grouping::Union{Nothing,AbstractVector{<:AbstractVector},AbstractVector{<:AbstractDict}}
end
(c::ModelPrinter)(max_depth::Union{Nothing,Integer} = nothing; args...) = c(c.model; max_depth, args...)
(c::ModelPrinter)(print_solemodel::Bool = true, max_depth::Union{Nothing,Integer} = nothing; args...) = c(print_solemodel ? c.solemodel : c.model; max_depth, args...)
(c::ModelPrinter)(model; max_depth = nothing) = MDT.printmodel(model; variable_names_map = c.var_grouping, max_depth = max_depth)

Base.show(io::IO, c::ModelPrinter) = print(io, "ModelPrinter object")
