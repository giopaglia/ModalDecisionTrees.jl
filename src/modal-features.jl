
export ModalFeature,
        DimensionalFeature,
        SingleAttributeMin, SingleAttributeMax,
        SingleAttributeSoftMin, SingleAttributeSoftMax,
        SingleAttributeGenericFeature, MultiAttributeFeature,
        ExternalFWDFeature

############################################################################################
############################################################################################
############################################################################################

# A modal feature represents a function that can be computed on a world.
# The simplest example is, min(A1), which computes the minimum for attribute 1
#  for a given world.
# The value of a feature for a given world can be then evaluated in a condition,
#  such as: min(A1) >= 10.
abstract type ModalFeature{U} <: SoleModels.AbstractFeature{U} end

Base.show(io::IO, f::ModalFeature, args...; kwargs...) = print(io, display_feature(f, args...; kwargs...))

################################################################################
################################################################################

# A dimensional feature represents a function that can be computed when the world
#  is an entity that lives in a dimensional context; for example, the world
#  can be a region of the matrix representing a b/w image.
abstract type DimensionalFeature{U} <: ModalFeature{U} end
@inline (f::DimensionalFeature)(args...) = interpret_feature(f, args...)

# Dimensional features functions are computed on dimensional channels, 
#  namely, interpretations of worlds on a dimensional contexts
# const DimensionalFeatureFunction = FunctionWrapper{Number,Tuple{AbstractArray{<:Number}}}

############################################################################################

# A dimensional feature represented by the application of a function to a channel
#  (e.g., how much a region of the image resembles a horse)
struct MultiAttributeFeature{U} <: DimensionalFeature{U}
    f::Function
end
function interpret_feature(f::MultiAttributeFeature{U}, inst::AbstractDimensionalInstance{T})::U where {U,T}
    (f.f(inst))
end
display_feature(f::MultiAttributeFeature,         args...; kwargs...) = "$(f.f)"

############################################################################################

abstract type SingleAttributeFeature{U} <: DimensionalFeature{U} end

i_attribute(f::SingleAttributeFeature) = f.i_attribute
attribute_name(f::SingleAttributeFeature; attribute_names_map::Union{Nothing,AbstractDict,AbstractVector} = nothing) = (isnothing(attribute_names_map) ? "A$(i_attribute(f))" : "$(attribute_names_map[i_attribute(f)])")


# A feature can be just a name
struct SingleAttributeNamedFeature{U} <: SingleAttributeFeature{U}
    i_attribute::Integer
    name::String
end
function interpret_feature(f::SingleAttributeNamedFeature, inst::AbstractDimensionalInstance{T}) where {T}
    @error "Can't intepret SingleAttributeNamedFeature on any structure at all."
end
display_feature(f::SingleAttributeNamedFeature;    attribute_names_map::Union{Nothing,AbstractVector,AbstractDict} = nothing) = (n = attribute_name(f; attribute_names_map = attribute_names_map); (f.name == "" ? "$(n)" : "$(f.name)($n)"))

############################################################################################

############################################################################################

# Notable single-attribute features: minimum and maximum of a given attribute
#  e.g., min(A1), max(A10)
struct SingleAttributeMin{U} <: SingleAttributeFeature{U}
    i_attribute::Integer
    function SingleAttributeMin{U}(i_attribute::Integer)
        return new{U}(i_attribute)
    end
    function SingleAttributeMin(i_attribute::Integer)
        return SingleAttributeMin{Real}(i_attribute)
    end
end
function interpret_feature(f::SingleAttributeMin{U}, inst::AbstractDimensionalInstance{T}) where {U,T}
    (minimum(get_instance_attribute(inst,f.i_attribute)))::U
end
display_feature(f::SingleAttributeMin;             attribute_names_map::Union{Nothing,AbstractVector,AbstractDict} = nothing) = "min($(attribute_name(f; attribute_names_map = attribute_names_map)))"

struct SingleAttributeMax{U} <: SingleAttributeFeature{U}
    i_attribute::Integer
    function SingleAttributeMax{U}(i_attribute::Integer)
        return new{U}(i_attribute)
    end
    function SingleAttributeMax(i_attribute::Integer)
        return SingleAttributeMax{Real}(i_attribute)
    end
end
function interpret_feature(f::SingleAttributeMax{U}, inst::AbstractDimensionalInstance{T}) where {U,T}
    (maximum(get_instance_attribute(inst,f.i_attribute)))::U
end
display_feature(f::SingleAttributeMax;             attribute_names_map::Union{Nothing,AbstractVector,AbstractDict} = nothing) = "max($(attribute_name(f; attribute_names_map = attribute_names_map)))"

############################################################################################

# Softened versions (quantiles) of single-attribute minimum and maximum
#  e.g., min80(A1), max80(A10)
struct SingleAttributeSoftMin{U,T<:AbstractFloat} <: SingleAttributeFeature{U}
    i_attribute::Integer
    alpha::T
    function SingleAttributeSoftMin{U}(i_attribute::Integer, alpha::T) where {T}
        @assert !(alpha > 1.0 || alpha < 0.0) "Can't instantiate SingleAttributeSoftMin with alpha = $(alpha)"
        @assert !isone(alpha) "Can't instantiate SingleAttributeSoftMin with alpha = $(alpha). Use SingleAttributeMin instead!"
        new{U,T}(i_attribute, alpha)
    end
end
alpha(f::SingleAttributeSoftMin) = f.alpha
display_feature(f::SingleAttributeSoftMin;         attribute_names_map::Union{Nothing,AbstractVector,AbstractDict} = nothing) = "min" * util.subscriptnumber(rstrip(rstrip(string(f.alpha*100), '0'), '.')) * "($(attribute_name(f; attribute_names_map = attribute_names_map)))"

function interpret_feature(f::SingleAttributeSoftMin{U}, inst::AbstractDimensionalInstance{T}) where {U,T}
    util.softminimum(get_instance_attribute(inst,f.i_attribute), f.alpha)::U
end
struct SingleAttributeSoftMax{U,T<:AbstractFloat} <: SingleAttributeFeature{U}
    i_attribute::Integer
    alpha::T
    function SingleAttributeSoftMax{U}(i_attribute::Integer, alpha::T) where {T}
        @assert !(alpha > 1.0 || alpha < 0.0) "Can't instantiate SingleAttributeSoftMax with alpha = $(alpha)"
        @assert !isone(alpha) "Can't instantiate SingleAttributeSoftMax with alpha = $(alpha). Use SingleAttributeMax instead!"
        new{U,T}(i_attribute, alpha)
    end
end
function interpret_feature(f::SingleAttributeSoftMax{U}, inst::AbstractDimensionalInstance{T}) where {U,T}
    util.softmaximum(get_instance_attribute(inst,f.i_attribute), f.alpha)::U
end
alpha(f::SingleAttributeSoftMax) = f.alpha
display_feature(f::SingleAttributeSoftMax;         attribute_names_map::Union{Nothing,AbstractVector,AbstractDict} = nothing) = "max" * util.subscriptnumber(rstrip(rstrip(string(f.alpha*100), '0'), '.')) * "($(attribute_name(f; attribute_names_map = attribute_names_map)))"

# TODO simplify OneWorld case:
# function interpret_feature(f::SingleAttributeSoftMin, inst::AbstractDimensionalInstance{T}) where {T}
#     get_instance_attribute(inst,f.i_attribute)::T
# end
# function interpret_feature(f::SingleAttributeSoftMax, inst::AbstractDimensionalInstance{T}) where {T}
#     get_instance_attribute(inst,f.i_attribute)::T
# end
# Note: Maybe features should dispatch on WorldType, (as well or on the type of underlying data?)

############################################################################################

# A dimensional feature represented by the application of a function to a
#  single attribute (e.g., avg(red), that is, how much red is in an image region)
struct SingleAttributeGenericFeature{U} <: SingleAttributeFeature{U}
    i_attribute::Integer
    f::Function
end
function interpret_feature(f::SingleAttributeGenericFeature{U}, inst::AbstractDimensionalInstance{T}) where {U,T}
    (f.f(util.vectorize(get_instance_attribute(inst,f.i_attribute));))::U
end
display_feature(f::SingleAttributeGenericFeature;  attribute_names_map::Union{Nothing,AbstractVector,AbstractDict} = nothing) = "$(f.f)($(attribute_name(f; attribute_names_map = attribute_names_map)))"

############################################################################################

# These features collapse to a single value; this can be useful to know
is_collapsing_single_attribute_feature(f::Union{SingleAttributeMin, SingleAttributeMax, SingleAttributeSoftMin, SingleAttributeSoftMax}) = true
is_collapsing_single_attribute_feature(f::SingleAttributeGenericFeature) = (f.f in [minimum, maximum, mean])

############################################################################################

# A feature can also be just a name
struct NamedFeature{U} <: ModalFeature{U}
    name::String
end
function interpret_feature(f::NamedFeature, inst::AbstractDimensionalInstance{T}) where {T}
    @error "Can't intepret NamedFeature on any structure at all."
end
display_feature(f::NamedFeature,             args...; kwargs...) = "$(f.name)"

############################################################################################


# A feature can be imported from a FWD (FWD) structure (see ModalLogic module)
struct ExternalFWDFeature{U} <: ModalFeature{U}
    name::String
    fwd::Any
end
function interpret_feature(f::ExternalFWDFeature, inst::AbstractDimensionalInstance{T}) where {T}
    @error "Can't intepret ExternalFWDFeature on any structure at all."
end
display_feature(f::ExternalFWDFeature,             args...; kwargs...) = "$(f.name)"

################################################################################
