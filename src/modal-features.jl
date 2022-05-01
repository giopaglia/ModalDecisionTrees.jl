export ModalFeature,
        DimensionalFeature,
        SingleAttributeMin, SingleAttributeMax,
        SingleAttributeSoftMin, SingleAttributeSoftMax,
        SingleAttributeFeature, MultiAttributeFeature,
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

################################################################################
################################################################################

# Dummy modal feature
struct _ModalFeatureNone  <: ModalFeature end; const ModalFeatureNone  = _ModalFeatureNone();
function get_interpretation_function(f::_ModalFeatureNone)
    @error "Can't intepret ModalFeatureNone on any structure at all."
end
Base.show(io::IO, f::_ModalFeatureNone) = print(io, "(Empty ModalFeature)")

############################################################################################

# A dimensional feature represents a function that can be computed when the world
#  is an entity that lives in a dimensional context; for example, the world
#  can be a region of the matrix representing a b/w image.
abstract type DimensionalFeature <: ModalFeature end
(f::DimensionalFeature)(args...) = (get_interpretation_function(f))(args...)

# Dimensional features functions are computed on dimensional channels, 
#  namely, interpretations of worlds on a dimensional contexts
const DimensionalFeatureFunction = FunctionWrapper{Number,Tuple{AbstractArray{<:Number}}}

############################################################################################

# Notable single-attribute features: minimum and maximum of a given attribute
#  e.g., min(A1), max(A10)
struct SingleAttributeMin <: DimensionalFeature
    i_attribute::Integer
end
function get_interpretation_function(f::SingleAttributeMin)
    DimensionalFeatureFunction(minimum ∘ (x)->ModalLogic.get_instance_attribute(x,f.i_attribute))
end
Base.show(io::IO, f::SingleAttributeMin) = print(io, "min(A$(f.i_attribute))")

struct SingleAttributeMax <: DimensionalFeature
    i_attribute::Integer
end
function get_interpretation_function(f::SingleAttributeMax)
    DimensionalFeatureFunction(maximum ∘ (x)->ModalLogic.get_instance_attribute(x,f.i_attribute))
end
Base.show(io::IO, f::SingleAttributeMax) = print(io, "max(A$(f.i_attribute))")

############################################################################################

# Softened versions (quantiles) of single-attribute minimum and maximum
#  e.g., min80(A1), max80(A10)
struct SingleAttributeSoftMin{T<:AbstractFloat} <: DimensionalFeature
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
Base.show(io::IO, f::SingleAttributeSoftMin) = print(io, "min" * util.subscriptnumber(rstrip(rstrip(string(f.alpha*100), '0'), '.')) * "(A$(f.i_attribute))")

function get_interpretation_function(f::SingleAttributeSoftMin)
    DimensionalFeatureFunction((x)->(vals = util.vectorize(ModalLogic.get_instance_attribute(x,f.i_attribute)); partialsort!(vals,ceil(Int, f.alpha*length(vals)); rev=true)))
end
struct SingleAttributeSoftMax{T<:AbstractFloat} <: DimensionalFeature
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
function get_interpretation_function(f::SingleAttributeSoftMax)
    DimensionalFeatureFunction((x)->(vals = util.vectorize(ModalLogic.get_instance_attribute(x,f.i_attribute)); partialsort!(vals,ceil(Int, f.alpha*length(vals)))))
end
alpha(f::SingleAttributeSoftMax) = f.alpha
Base.show(io::IO, f::SingleAttributeSoftMax) = print(io, "max" * util.subscriptnumber(rstrip(rstrip(string(f.alpha*100), '0'), '.')) * "(A$(f.i_attribute))")

# TODO simplify OneWorld case:
# function get_interpretation_function(f::SingleAttributeSoftMin)
#     ModalLogic.get_instance_attribute(x,f.i_attribute)
# end
# function get_interpretation_function(f::SingleAttributeSoftMax)
#     ModalLogic.get_instance_attribute(x,f.i_attribute)
# end
# Note: Maybe features should dispatch on WorldType, (as well or on the type of underlying data?)

############################################################################################

# A dimensional feature represented by the application of a function to a
#  single attribute (e.g., avg(red), that is, how much red is in an image region)
struct SingleAttributeFeature <: DimensionalFeature
    i_attribute::Integer
    f::Function
end
function get_interpretation_function(f::SingleAttributeFeature)
    DimensionalFeatureFunction(f.f ∘ (x)->(vals = util.vectorize(ModalLogic.get_instance_attribute(x,f.i_attribute));))
end
Base.show(io::IO, f::SingleAttributeFeature) = print(io, "$(f.f)(A$(f.i_attribute))")

############################################################################################

# A dimensional feature represented by the application of a function to a channel
#  (e.g., how much a region of the image resembles a horse)
struct MultiAttributeFeature <: DimensionalFeature
    f::Function
end
function get_interpretation_function(f::MultiAttributeFeature)
    DimensionalFeatureFunction(f.f)
end
Base.show(io::IO, f::MultiAttributeFeature) = print(io, "$(f.f)")

############################################################################################

# A feature can be imported from a FeaturedWorldDataset (FWD) structure (see ModalLogic module)
struct ExternalFWDFeature <: ModalFeature
    name::String
    fwd::Any
end
function get_interpretation_function(f::ExternalFWDFeature)
    @error "Can't intepret ModalFeatureNone on any structure at all."
end
Base.show(io::IO, f::ExternalFWDFeature) = print(io, "$(f.name)")

################################################################################
