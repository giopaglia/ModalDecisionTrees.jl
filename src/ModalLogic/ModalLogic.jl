module ModalLogic

using IterTools
import Base: argmax, argmin, size, show, convert
using Logging: @logmsg
using ..DecisionTree

using ComputedFieldTypes

export AbstractWorld, AbstractRelation,
				Ontology, OntologicalDataset,
				WorldSet,
				display_propositional_test,
				display_modal_test,
				RelationAll, RelationNone, RelationId,
				world_type
				# enumAccessibles, enumAccRepr

# Fix
Base.keys(g::Base.Generator) = g.iter

# Abstract classes for world & relations
abstract type AbstractWorld end
abstract type AbstractRelation end

# Concrete class for ontology models (world type + set of relations)
struct Ontology{WorldType<:AbstractWorld}
	relationSet :: AbstractVector{<:AbstractRelation}
	Ontology{WorldType}(relationSet) where {WorldType<:AbstractWorld} = begin
		relationSet = unique(relationSet)
		for relation in relationSet
			if !goesWith(WorldType, relation)
				error("Can't instantiate Ontology with WorldType $(WorldType) and relation $(relation)")
			end
		end
		return new{WorldType}(relationSet)
	end
	# Ontology(worldType, relationSet) = new(worldType, relationSet)
end

world_type(::Ontology{WT}) where {WT} = WT

# strip_ontology(ontology::Ontology) = Ontology{OneWorld}(AbstractRelation[])


# This constant is used to create the default world for each WorldType
#  (e.g. Interval(ModalLogic.emptyWorld) = Interval(-1,0))
struct _firstWorld end;    const firstWorld    = _firstWorld();
struct _emptyWorld end;    const emptyWorld    = _emptyWorld();
struct _centeredWorld end; const centeredWorld = _centeredWorld();

# World generators/enumerators and array/set-like structures
# TODO test the functions for WorldSets with Sets and Arrays, and find the performance optimum
const AbstractWorldSet{W} = Union{AbstractVector{W},AbstractSet{W}} where {W<:AbstractWorld}
# Concrete type for sets: vectors are faster than sets, so we
# const WorldSet = AbstractSet{W} where W<:AbstractWorld
const WorldSet{W} = Vector{W} where {W<:AbstractWorld}
WorldSet{W}(S::WorldSet{W}) where {W<:AbstractWorld} = S


# TODO improve, decouple from relationSets definitions
# Actually, this will not work because relationSet does this collect(set(...)) thing... mh maybe better avoid that thing?
show(io::IO, o::Ontology{WorldType}) where {WorldType} = begin
	print(io, "Ontology(")
	show(io, WorldType)
	print(io, ",")
	if issetequal(o.relationSet, IARelations)
		print(io, "IARelations")
	elseif issetequal(o.relationSet, IARelations_extended)
		print(io, "IARelations_extended")
	elseif issetequal(o.relationSet, IA2DRelations)
		print(io, "IA2DRelations")
	elseif issetequal(o.relationSet, IA2DRelations_U)
		print(io, "IA2DRelations_U")
	elseif issetequal(o.relationSet, IA2DRelations_extended)
		print(io, "IA2DRelations_extended")
	elseif issetequal(o.relationSet, RCC8Relations)
		print(io, "RCC8")
	elseif issetequal(o.relationSet, RCC5Relations)
		print(io, "RCC5")
	else
		show(io, o.relationSet)
	end
	print(io, ")")
end

################################################################################
# BEGIN Helpers
################################################################################

# https://stackoverflow.com/questions/46671965/printing-variable-subscripts-in-julia/46674866
# '₀'
subscriptnumber(i::Int) = begin
	join([
		(if i < 0
			[Char(0x208B)]
		else [] end)...,
		[Char(0x2080+d) for d in reverse(digits(abs(i)))]...
	])
end
# https://www.w3.org/TR/xml-entity-names/020.html
# '․', 'ₑ', '₋'
subscriptnumber(s::AbstractString) = begin
	char_to_subscript(ch) = begin
		if ch == 'e'
			'ₑ'
		elseif ch == '.'
			'․'
		elseif ch == '.'
			'․'
		elseif ch == '-'
			'₋'
		else
			subscriptnumber(parse(Int, ch))
		end
	end

	try
		join(map(char_to_subscript, [string(ch) for ch in s]))
	catch
		s
	end
end

subscriptnumber(i::AbstractFloat) = subscriptnumber(string(i))

################################################################################
# END Helpers
################################################################################

################################################################################
# BEGIN Dataset types
################################################################################

abstract type AbstractModalDataset{T<:Real,WorldType<:AbstractWorld} end



export n_samples, n_attributes, channel_size, max_channel_size,
				MatricialInstance,
				MatricialDataset,
				# MatricialUniDataset,
				MatricialChannel

# A dataset, given by a set of N-dimensional (multi-attribute) matrices/instances,
#  and an Ontology to be interpreted on each of them.
# - The size of the domain array is {X×Y×...} × n_attributes × n_samples
# - N is the dimensionality of the domain itself (e.g. 1 for the temporal case, 2 for the spatial case)
#    and its dimensions are denoted as X,Y,Z,...
# - A uni-attribute dataset is of dimensionality S=N+1
# - A multi-attribute dataset is of dimensionality D=N+1+1
#  https://discourse.julialang.org/t/addition-to-parameter-of-parametric-type/20059/5

# TODO: It'd be nice to define these as a function of N, https://github.com/JuliaLang/julia/issues/8322
#   e.g. const MatricialUniDataset{T,N}       = AbstractArray{T,N+1}

const MatricialDataset{T,D}     = AbstractArray{T,D}
# const MatricialUniDataset{T,UD} = AbstractArray{T,UD}
const MatricialChannel{T,N}     = AbstractArray{T,N}
const MatricialInstance{T,MN}   = AbstractArray{T,MN}

n_samples(d::MatricialDataset{T,D})    where {T,D} = size(d, D)
n_attributes(d::MatricialDataset{T,D}) where {T,D} = size(d, D-1)
channel_size(d::MatricialDataset{T,D}) where {T,D} = size(d)[1:end-2]
max_channel_size = channel_size # TODO rename channel_size into max_channel_size
inst_channel_size(inst::MatricialInstance{T,MN}) where {T,MN} = size(inst)[1:end-1]

@inline getInstance(d::MatricialDataset{T,2},     idx::Integer) where T = @views d[:, idx]         # N=0
@inline getInstance(d::MatricialDataset{T,3},     idx::Integer) where T = @views d[:, :, idx]      # N=1
@inline getInstance(d::MatricialDataset{T,4},     idx::Integer) where T = @views d[:, :, :, idx]   # N=2

@inline getInstances(d::MatricialDataset{T,2}, inds::AbstractVector{<:Integer}; return_view = false) where T = if return_view @views d[:, inds]       else d[:, inds]    end # N=0
@inline getInstances(d::MatricialDataset{T,3}, inds::AbstractVector{<:Integer}; return_view = false) where T = if return_view @views d[:, :, inds]    else d[:, :, inds] end # N=1
@inline getInstances(d::MatricialDataset{T,4}, inds::AbstractVector{<:Integer}; return_view = false) where T = if return_view @views d[:, :, :, inds] else d[:, :, :, inds] end # N=2

@inline getChannel(d::MatricialDataset{T,2},      idx_i::Integer, idx_a::Integer) where T = @views d[      idx_a, idx_i]::T                     # N=0
@inline getChannel(d::MatricialDataset{T,3},      idx_i::Integer, idx_a::Integer) where T = @views d[:,    idx_a, idx_i]::MatricialChannel{T,1} # N=1
@inline getChannel(d::MatricialDataset{T,4},      idx_i::Integer, idx_a::Integer) where T = @views d[:, :, idx_f, idx_i]::MatricialChannel{T,2} # N=2
# @inline getUniChannel(ud::MatricialUniDataset{T,1},  idx::Integer) where T = @views ud[idx]           # N=0
# @inline getUniChannel(ud::MatricialUniDataset{T,2},  idx::Integer) where T = @views ud[:, idx]        # N=1
# @inline getUniChannel(ud::MatricialUniDataset{T,3},  idx::Integer) where T = @views ud[:, :, idx]     # N=2
@inline getInstanceAttribute(inst::MatricialInstance{T,1},      idx::Integer) where T = @views inst[      idx]::T                     # N=0
@inline getInstanceAttribute(inst::MatricialInstance{T,2},      idx::Integer) where T = @views inst[:,    idx]::MatricialChannel{T,1} # N=1
@inline getInstanceAttribute(inst::MatricialInstance{T,3},      idx::Integer) where T = @views inst[:, :, idx]::MatricialChannel{T,2} # N=2

# @inline strip_domain(d::MatricialDataset{T,2}) where T = d  # N=0
# @inline strip_domain(d::MatricialDataset{T,3}) where T = dropdims(d; dims=1)      # N=1
# @inline strip_domain(d::MatricialDataset{T,4}) where T = dropdims(d; dims=(1,2))  # N=2

# Initialize MatricialUniDataset by slicing across the attribute dimension
# MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,2}) where T = Array{T, 1}(undef, n_samples(d))::MatricialUniDataset{T, 1}
# MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,3}) where T = Array{T, 2}(undef, size(d)[1:end-1])::MatricialUniDataset{T, 2}
# MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,4}) where T = Array{T, 3}(undef, size(d)[1:end-1])::MatricialUniDataset{T, 3}

# TODO generalize as init_Xf(X::OntologicalDataset{T, N}) where T = Array{T, N+1}(undef, size(X)[3:end]..., n_samples(X))
@computed struct OntologicalDataset{T, N, WorldType} <: AbstractModalDataset{T,WorldType}
	ontology  :: Ontology{WorldType}
	domain    :: MatricialDataset{T,N+1+1}
	
	OntologicalDataset{T, N}(ontology::Ontology{WorldType}, domain::MatricialDataset{T,D}) where {T, N, D, WorldType<:AbstractWorld} = begin
		OntologicalDataset{T, N, WorldType}(ontology, domain)
	end
	
	OntologicalDataset{T, N, WorldType}(ontology::Ontology{WorldType}, domain::MatricialDataset{T,D}) where {T, N, D, WorldType<:AbstractWorld} = begin
		_check_dims(T, N, D, WorldType)
		
		# Type unstable?
		# if prod(channel_size(domain)) == 1
		# 	ontology = ModalLogic.strip_ontology(ontology)
		# 	WorldType = world_type(strip_ontology)
		# end
		
		new{T, N, WorldType}(ontology, domain)
	end
	
	_check_dims(T, N, D, WorldType) = begin
		if N != ModalLogic.worldTypeDimensionality(WorldType)
			error("ERROR! Dimensionality mismatch: can't interpret worldType $(WorldType) (dimensionality = $(ModalLogic.worldTypeDimensionality(WorldType)) on MatricialDataset of dimensionality = $(N)")
		end
		
		if D != (N+1+1)
			error("ERROR! Dimensionality mismatch: can't instantiate OntologicalDataset{$(T), $(N)} with MatricialDataset{$(T),$(D)}")
		end
	end
end


size(X::OntologicalDataset{T,N})             where {T,N} = size(X.domain)
size(X::OntologicalDataset{T,N}, i::Integer) where {T,N} = size(X.domain, i)
n_samples(X::OntologicalDataset{T,N})        where {T,N} = n_samples(X.domain)
n_attributes(X::OntologicalDataset{T,N})     where {T,N} = n_attributes(X.domain)
channel_size(X::OntologicalDataset{T,N})     where {T,N} = channel_size(X.domain)

@inline getInstance(d::OntologicalDataset{T,N,WT}, args::Vararg) where {T,N,WT}  = getInstance(d.domain, args...)
@inline getInstances(d::OntologicalDataset{T,N,WT}, args::Vararg) where {T,N,WT} = getInstances(d.domain, args...)
@inline getChannel(d::OntologicalDataset{T,N,WT},   args::Vararg) where {T,N,WT} = getChannel(d.domain, args...)

# TODO use Xf[i,[(:) for i in 1:N]...]
# @computed @inline getChannel(X::OntologicalDataset{T,N}, idxs::AbstractVector{Integer}, attribute::Integer) where T = X[idxs, attribute, fill(:, N)...]::AbstractArray{T,N-1}

# TODO maybe using views can improve performances
# attributeview(X::OntologicalDataset{T,0}, idxs::AbstractVector{Integer}, attribute::Integer) = X.domain[idxs, attribute]
# attributeview(X::OntologicalDataset{T,1}, idxs::AbstractVector{Integer}, attribute::Integer) = view(X.domain, idxs, attribute, :)
# attributeview(X::OntologicalDataset{T,2}, idxs::AbstractVector{Integer}, attribute::Integer) = view(X.domain, idxs, attribute, :, :)

################################################################################
# END Dataset types
################################################################################

include("operators.jl")
include("testOperators.jl")
include("featureTypes.jl")

display_propositional_test(test_operator::TestOperator, lhs::String, threshold::Number) =
	"$(lhs) $(test_operator) $(threshold)"

display_modal_test(modality::AbstractRelation, test_operator::TestOperator, feature::FeatureType, threshold::Number) = begin
	test = display_propositional_test(test_operator, display_feature(feature), threshold)
	if modality != RelationId
		"$(display_existential_modality(modality)) ($test)"
	else
		"$test"
	end
end

################################################################################
################################################################################

show(io::IO, r::AbstractRelation) = print(io, display_existential_modality(r))
display_existential_modality(r) = "⟨" * display_rel_short(r) * "⟩"

# Utility type for enhanced computation of thresholds
abstract type _ReprTreatment end
struct _ReprFake{worldType<:AbstractWorld} <: _ReprTreatment w :: worldType end
struct _ReprMax{worldType<:AbstractWorld}  <: _ReprTreatment w :: worldType end
struct _ReprMin{worldType<:AbstractWorld}  <: _ReprTreatment w :: worldType end
struct _ReprVal{worldType<:AbstractWorld}  <: _ReprTreatment w :: worldType end
struct _ReprNone{worldType<:AbstractWorld} <: _ReprTreatment end

## Enumerate accessible worlds

# Fallback: enumAccessibles works with domains AND their dimensions
enumAccessibles(S::AbstractWorldSet{WorldType}, r::AbstractRelation, channel::MatricialChannel{T,N}) where {T,N,WorldType<:AbstractWorld} = enumAccessibles(S, r, size(channel)...)
enumAccRepr(S::Any, r::AbstractRelation, channel::MatricialChannel{T,N}) where {T,N} = enumAccRepr(S, r, size(channel)...)
# Fallback: enumAccessibles for world sets maps to enumAcc-ing their elements
#  (note: one may overload this function to provide improved implementations for special cases (e.g. <L> of a world set in interval algebra))
enumAccessibles(S::AbstractWorldSet{WorldType}, r::AbstractRelation, XYZ::Vararg{Integer,N}) where {T,N,WorldType<:AbstractWorld} = begin
	IterTools.imap(WorldType,
		IterTools.distinct(Iterators.flatten((enumAccBare(w, r, XYZ...) for w in S)))
	)
end
enumAccessibles(w::WorldType, r::AbstractRelation, XYZ::Vararg{Integer,N}) where {T,N,WorldType<:AbstractWorld} = begin
	IterTools.imap(WorldType, enumAccBare(w, r, XYZ...))
end
## Basic, ontology-agnostic relations:

# None relation      (RelationNone)  =  Used as the "nothing" constant
struct _RelationNone  <: AbstractRelation end; const RelationNone = _RelationNone();

# Identity relation  (RelationId)    =  S -> S
struct _RelationId    <: AbstractRelation end; const RelationId   = _RelationId();

enumAccessibles(w::WorldType,           ::_RelationId, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = [w] # IterTools.imap(identity, [w])
enumAccessibles(S::AbstractWorldSet{W}, ::_RelationId, XYZ::Vararg{Integer,N}) where {W<:AbstractWorld,N} = S # TODO try IterTools.imap(identity, S) ?

enumAccRepr(::_TestOpGeq, w::WorldType, ::_RelationId, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = _ReprMin(w)
enumAccRepr(::_TestOpLeq, w::WorldType, ::_RelationId, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = _ReprMax(w)

enumAccReprAggr(f::FeatureTypeFun, a::Aggregator, w::WorldType, r::AbstractRelation, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = enumAccessibles(w, r, XYZ...)
enumAccReprAggr(::FeatureTypeFun, ::Aggregator, w::WorldType, r::_RelationId, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = enumAccessibles(w, r, XYZ...)

# computeModalThresholdDual(test_operator::TestOperator, w::WorldType, relation::_RelationId, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
# 	computePropositionalThresholdDual(test_operator, w, channel)
# computeModalThreshold(test_operator::TestOperator, w::WorldType, relation::_RelationId, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
# 	computePropositionalThreshold(test_operator, w, channel)
# computeModalThresholdMany(test_ops::Vector{<:TestOperator}, w::WorldType, relation::_RelationId, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
# 	computeModalThresholdMany(test_ops, w, channel)

display_rel_short(::_RelationId)  = "Id"

# Global relation    (RelationAll)   =  S -> all-worlds
struct _RelationAll   <: AbstractRelation end; const RelationAll  = _RelationAll();

display_rel_short(::_RelationAll) = ""

# Shortcut for enumerating all worlds
enumAll(::Type{WorldType}, args::Vararg) where {WorldType<:AbstractWorld} = enumAccessibles(WorldType[], RelationAll, args...)


################################################################################
################################################################################

minExtrema(extr::Union{NTuple{N,NTuple{2,T}},AbstractVector{NTuple{2,T}}}) where {T<:Number,N} = reduce(((fst,snd),(f,s))->(min(fst,f),max(snd,s)), extr; init=(typemax(T),typemin(T)))
maxExtrema(extr::Union{NTuple{N,NTuple{2,T}},AbstractVector{NTuple{2,T}}}) where {T<:Number,N} = reduce(((fst,snd),(f,s))->(max(fst,f),min(snd,s)), extr; init=(typemin(T),typemax(T)))
minExtrema(extr::Vararg{NTuple{2,T}}) where {T<:Number} = minExtrema(extr)
maxExtrema(extr::Vararg{NTuple{2,T}}) where {T<:Number} = maxExtrema(extr)

include("OneWorld.jl")
# include("Point.jl")

include("Interval.jl")
include("IARelations.jl")
include("TopoRelations.jl")

include("Interval2D.jl")
include("IA2DRelations.jl")
include("Topo2DRelations.jl")

################################################################################
################################################################################

# TODO
# A relation can be defined as a union of other relations.
# In this case, thresholds can be computed by maximization/minimization of the
#  thresholds referred to the relations involved.
# abstract type AbstractRelation end
struct _UnionOfRelations{T<:NTuple{N,<:AbstractRelation} where N} <: AbstractRelation end;

# computeModalThresholdDual(test_operator::TestOperator, w::WorldType, relation::R where R<:_UnionOfRelations{relsTuple}, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
# 	computePropositionalThresholdDual(test_operator, w, channel)
# 	fieldtypes(relsTuple)
# computeModalThreshold(test_operator::TestOperator, w::WorldType, relation::R where R<:_UnionOfRelations{relsTuple}, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
# 	computePropositionalThreshold(test_operator, w, channel)
# 	fieldtypes(relsTuple)

################################################################################
################################################################################

# Perform the modal step, that is, evaluate a modal formula
#  on a domain, and eventually compute the new world set.
# TODO write better
modalStep(S::WorldSetType,
					relation::R,
					channel::AbstractArray{T,N},
					test_operator::TestOperator,
					threshold::T) where {W<:AbstractWorld, WorldSetType<:Union{AbstractSet{W},AbstractVector{W}}, R<:AbstractRelation, T, N} = begin
	@logmsg DTDetail "modalStep" S relation display_modal_test(relation, test_operator, -1, threshold)
	satisfied = false
	worlds = enumAccessibles(S, relation, channel)
	if length(collect(Iterators.take(worlds, 1))) > 0
		new_worlds = WorldSetType()
		for w in worlds
			if testCondition(test_operator, w, channel, threshold)
				@logmsg DTDetail " Found world " w ch_readWorld(w,channel)
				satisfied = true
				push!(new_worlds, w)
			end
		end
		if satisfied == true
			S = new_worlds
		else 
			# If none of the neighboring worlds satisfies the condition, then 
			#  the new set is left unchanged
		end
	else
		@logmsg DTDetail "   No world found"
		# If there are no neighboring worlds, then the modal condition is not met
	end
	if satisfied
		@logmsg DTDetail "   YES" S
	else
		@logmsg DTDetail "   NO" 
	end
	return (satisfied,S)
end

################################################################################
################################################################################

export genericIntervalOntology,
				IntervalOntology,
				Interval2DOntology,
				getIntervalOntologyOfDim,
				genericIntervalRCC8Ontology,
				IntervalRCC8Ontology,
				Interval2DRCC8Ontology,
				getIntervalRCC8OntologyOfDim,
				getIntervalRCC5OntologyOfDim,
				IntervalRCC5Ontology,
				Interval2DRCC5Ontology

abstract type OntologyType end
struct _genericIntervalOntology  <: OntologyType end; const genericIntervalOntology  = _genericIntervalOntology();  # After
const IntervalOntology   = Ontology{Interval}(IARelations)
const Interval2DOntology = Ontology{Interval2D}(IA2DRelations)

struct _genericIntervalRCC8Ontology  <: OntologyType end; const genericIntervalRCC8Ontology  = _genericIntervalRCC8Ontology();  # After
const IntervalRCC8Ontology   = Ontology{Interval}(RCC8Relations)
const Interval2DRCC8Ontology = Ontology{Interval2D}(RCC8Relations)
const IntervalRCC5Ontology   = Ontology{Interval}(RCC5Relations)
const Interval2DRCC5Ontology = Ontology{Interval2D}(RCC5Relations)
 
getIntervalOntologyOfDim(::MatricialDataset{T,D}) where {T,D} = getIntervalOntologyOfDim(Val(D-2))
getIntervalOntologyOfDim(::Val{1}) = IntervalOntology
getIntervalOntologyOfDim(::Val{2}) = Interval2DOntology

getIntervalRCC8OntologyOfDim(::MatricialDataset{T,D}) where {T,D} = getIntervalRCC8OntologyOfDim(Val(D-2))
getIntervalRCC8OntologyOfDim(::Val{1}) = IntervalRCC8Ontology
getIntervalRCC8OntologyOfDim(::Val{2}) = Interval2DRCC8Ontology

getIntervalRCC5OntologyOfDim(::MatricialDataset{T,D}) where {T,D} = getIntervalRCC5OntologyOfDim(Val(D-2))
getIntervalRCC5OntologyOfDim(::Val{1}) = IntervalRCC5Ontology
getIntervalRCC5OntologyOfDim(::Val{2}) = Interval2DRCC5Ontology

end # module
