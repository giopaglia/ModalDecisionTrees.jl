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
abstract type ModalFeature <: Function end

Base.show(io::IO, f::ModalFeature, args...; kwargs...) = print(io, display_feature(f, args...; kwargs...))

################################################################################
################################################################################

# A dimensional feature represents a function that can be computed when the world
#  is an entity that lives in a dimensional context; for example, the world
#  can be a region of the matrix representing a b/w image.
abstract type DimensionalFeature <: ModalFeature end
@inline (f::DimensionalFeature)(args...) = interpret_feature(f, args...)

# Dimensional features functions are computed on dimensional channels, 
#  namely, interpretations of worlds on a dimensional contexts
# const DimensionalFeatureFunction = FunctionWrapper{Number,Tuple{AbstractArray{<:Number}}}

############################################################################################

# A dimensional feature represented by the application of a function to a channel
#  (e.g., how much a region of the image resembles a horse)
struct MultiAttributeFeature <: DimensionalFeature
    f::Function
end
function interpret_feature(f::MultiAttributeFeature, inst::AbstractDimensionalInstance{T}) where {T}
    (f.f(inst))::T
end
display_feature(f::MultiAttributeFeature,         args...; kwargs...) = "$(f.f)"

############################################################################################

abstract type SingleAttributeFeature <: DimensionalFeature end

attribute_name(f::SingleAttributeFeature; attribute_names_map::Union{Nothing,AbstractDict,AbstractVector} = nothing) = (isnothing(attribute_names_map) ? "A$(f.i_attribute)" : "$(attribute_names_map[f.i_attribute])")


# A feature can be just a name
struct SingleAttributeNamedFeature <: SingleAttributeFeature
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
struct SingleAttributeMin <: SingleAttributeFeature
    i_attribute::Integer
end
function interpret_feature(f::SingleAttributeMin, inst::AbstractDimensionalInstance{T}) where {T}
    (minimum(get_instance_attribute(inst,f.i_attribute)))::T
end
display_feature(f::SingleAttributeMin;             attribute_names_map::Union{Nothing,AbstractVector,AbstractDict} = nothing) = "min($(attribute_name(f; attribute_names_map = attribute_names_map)))"

struct SingleAttributeMax <: SingleAttributeFeature
    i_attribute::Integer
end
function interpret_feature(f::SingleAttributeMax, inst::AbstractDimensionalInstance{T}) where {T}
    (maximum(get_instance_attribute(inst,f.i_attribute)))::T
end
display_feature(f::SingleAttributeMax;             attribute_names_map::Union{Nothing,AbstractVector,AbstractDict} = nothing) = "max($(attribute_name(f; attribute_names_map = attribute_names_map)))"

############################################################################################

# Softened versions (quantiles) of single-attribute minimum and maximum
#  e.g., min80(A1), max80(A10)
struct SingleAttributeSoftMin{T<:AbstractFloat} <: SingleAttributeFeature
    i_attribute::Integer
    alpha::T
    function SingleAttributeSoftMin(
        i_attribute::Integer,
        alpha::T,
    ) where {T}
        @assert !(alpha > 1.0 || alpha < 0.0) "Can't instantiate SingleAttributeSoftMin with alpha = $(alpha)"
        @assert !isone(alpha) "Can't instantiate SingleAttributeSoftMin with alpha = $(alpha). Use SingleAttributeMin instead!"
        new{T}(i_attribute, alpha)
    end
end
alpha(f::SingleAttributeSoftMin) = f.alpha
display_feature(f::SingleAttributeSoftMin;         attribute_names_map::Union{Nothing,AbstractVector,AbstractDict} = nothing) = "min" * util.subscriptnumber(rstrip(rstrip(string(f.alpha*100), '0'), '.')) * "($(attribute_name(f; attribute_names_map = attribute_names_map)))"

function interpret_feature(f::SingleAttributeSoftMin, inst::AbstractDimensionalInstance{T}) where {T}
    ((vals = util.vectorize(get_instance_attribute(inst,f.i_attribute)); partialsort!(vals,ceil(Int, f.alpha*length(vals)); rev=true)))::T
end
struct SingleAttributeSoftMax{T<:AbstractFloat} <: SingleAttributeFeature
    i_attribute::Integer
    alpha::T
    function SingleAttributeSoftMax(
        i_attribute::Integer,
        alpha::T,
    ) where {T}
        @assert !(alpha > 1.0 || alpha < 0.0) "Can't instantiate SingleAttributeSoftMax with alpha = $(alpha)"
        @assert !isone(alpha) "Can't instantiate SingleAttributeSoftMax with alpha = $(alpha). Use SingleAttributeMax instead!"
        new{T}(i_attribute, alpha)
    end
end
function interpret_feature(f::SingleAttributeSoftMax, inst::AbstractDimensionalInstance{T}) where {T}
    ((vals = util.vectorize(get_instance_attribute(inst,f.i_attribute)); partialsort!(vals,ceil(Int, f.alpha*length(vals)))))::T
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
struct SingleAttributeGenericFeature <: SingleAttributeFeature
    i_attribute::Integer
    f::Function
end
function interpret_feature(f::SingleAttributeGenericFeature, inst::AbstractDimensionalInstance{T}) where {T}
    (f.f(util.vectorize(get_instance_attribute(inst,f.i_attribute));))::T
end
display_feature(f::SingleAttributeGenericFeature;  attribute_names_map::Union{Nothing,AbstractVector,AbstractDict} = nothing) = "$(f.f)($(attribute_name(f; attribute_names_map = attribute_names_map)))"

############################################################################################

# These features collapse to a single value; this can be useful to know
is_collapsing_single_attribute_feature(f::Union{SingleAttributeMin, SingleAttributeMax, SingleAttributeSoftMin, SingleAttributeSoftMax}) = true
is_collapsing_single_attribute_feature(f::SingleAttributeGenericFeature) = (f.f in [minimum, maximum, mean])

############################################################################################

# A feature can also be just a name
struct NamedFeature <: ModalFeature
    name::String
end
function interpret_feature(f::NamedFeature, inst::AbstractDimensionalInstance{T}) where {T}
    @error "Can't intepret NamedFeature on any structure at all."
end
display_feature(f::NamedFeature,             args...; kwargs...) = "$(f.name)"

############################################################################################


# A feature can be imported from a FWD (FWD) structure (see ModalLogic module)
struct ExternalFWDFeature <: ModalFeature
    name::String
    fwd::Any
end
function interpret_feature(f::ExternalFWDFeature, inst::AbstractDimensionalInstance{T}) where {T}
    @error "Can't intepret ExternalFWDFeature on any structure at all."
end
display_feature(f::ExternalFWDFeature,             args...; kwargs...) = "$(f.name)"

################################################################################
