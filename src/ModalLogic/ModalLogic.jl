module ModalLogic

using IterTools
import Base: argmax, argmin, size, show, convert
using Logging: @logmsg
using ..DecisionTree

using ComputedFieldTypes

export AbstractWorld, AbstractRelation,
				Ontology, OntologicalDataset,
				n_samples, n_variables, channel_size,
				MatricialInstance,
				MatricialDataset,
				MatricialUniDataset,
				WorldSet,
				display_propositional_test,
				display_modal_test
				# , TestOperator
				# RelationAll, RelationNone, RelationId,
				# enumAcc, enumAccRepr

# Fix
Base.keys(g::Base.Generator) = g.iter

# Abstract classes for world & relations
abstract type AbstractWorld end
abstract type AbstractRelation end

# Concrete class for ontology models (world type + set of relations)
struct Ontology
	worldType   :: Type{<:AbstractWorld}
	relationSet :: AbstractVector{<:AbstractRelation}
	Ontology(worldType, relationSet) = new(worldType, unique(relationSet))
	# Ontology(worldType, relationSet) = new(worldType, relationSet)
end

# TODO improve, decouple from relationSets definitions
# Actually, this will not work because relationSet does this collect(set(...)) thing... mh maybe better avoid that thing?
show(io::IO, o::Ontology) = begin
	print(io, "Ontology(")
	show(io, o.worldType)
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
	elseif issetequal(o.relationSet, TopoRelations)
		print(io, "TopoRelations")
	else
		show(io, o.relationSet)
	end
	print(io, ")")
end

################################################################################
# BEGIN Matricial dataset
################################################################################

# A dataset, given by a set of N-dimensional (multi-variate) matrices/instances,
#  and an Ontology to be interpreted on each of them.
# - The size of the domain array is {X×Y×...} × n_samples × n_variables
# - N is the dimensionality of the domain itself (e.g. 1 for the temporal case, 2 for the spatialCase)
#    and its dimensions are denoted as X,Y,Z,...
# - A uni-variate dataset is of dimensionality S=N+1
# - A multi-variate dataset is of dimensionality D=N+1+1
#  https://discourse.julialang.org/t/addition-to-parameter-of-parametric-type/20059/5

const MatricialChannel{T,N}   = AbstractArray{T,N}
const MatricialInstance{T,MN} = AbstractArray{T,MN}
# TODO: It'd be nice to define these as a function of N, https://github.com/JuliaLang/julia/issues/8322
#   e.g. const MatricialUniDataset{T,N}       = AbstractArray{T,N+1}
const MatricialUniDataset{T,UD} = AbstractArray{T,UD}
const MatricialDataset{T,D}     = AbstractArray{T,D}

n_samples(d::MatricialDataset{T,D})        where {T,D} = size(d, D-1)
n_variables(d::MatricialDataset{T,D})      where {T,D} = size(d, D)
channel_size(d::MatricialDataset{T,D})     where {T,D} = size(d)[1:end-2]

@computed struct OntologicalDataset{T,N}
	ontology  :: Ontology
	domain    :: MatricialDataset{T,N+1+1}
end

size(X::OntologicalDataset{T,N})             where {T,N} = size(X.domain)
size(X::OntologicalDataset{T,N}, i::Integer) where {T,N} = size(X.domain, i)
n_samples(X::OntologicalDataset{T,N})        where {T,N} = n_samples(X.domain)
n_variables(X::OntologicalDataset{T,N})      where {T,N} = n_variables(X.domain)
channel_size(X::OntologicalDataset{T,N})     where {T,N} = channel_size(X.domain)

@inline getChannel(ud::MatricialUniDataset{T,1},  idx::Integer) where T = ud[idx]           # N=0
@inline getChannel(ud::MatricialUniDataset{T,2},  idx::Integer) where T = ud[:, idx]        # N=1
@inline getChannel(ud::MatricialUniDataset{T,3},  idx::Integer) where T = ud[:, :, idx]     # N=2
@inline getInstance(d::MatricialDataset{T,2},     idx::Integer) where T = d[idx, :]         # N=0
@inline getInstance(d::MatricialDataset{T,3},     idx::Integer) where T = d[:, idx, :]      # N=1
@inline getInstance(d::MatricialDataset{T,4},     idx::Integer) where T = d[:, :, idx, :]   # N=2
@inline getInstanceFeature(instance::MatricialInstance{T,1},      idx::Integer) where T = instance[      idx]::T                      # N=0
@inline getInstanceFeature(instance::MatricialInstance{T,2},      idx::Integer) where T = instance[:,    idx]::MatricialChannel{T,1}  # N=1
@inline getInstanceFeature(instance::MatricialInstance{T,3},      idx::Integer) where T = instance[:, :, idx]::MatricialChannel{T,2}  # N=2
@inline getFeature(d::MatricialDataset{T,2},      idx::Integer, feature::Integer) where T = d[      idx, feature]::T                     # N=0
@inline getFeature(d::MatricialDataset{T,3},      idx::Integer, feature::Integer) where T = d[:,    idx, feature]::MatricialChannel{T,1} # N=1
@inline getFeature(d::MatricialDataset{T,4},      idx::Integer, feature::Integer) where T = d[:, :, idx, feature]::MatricialChannel{T,2} # N=2

# TODO generalize as init_Xf(X::OntologicalDataset{T, N}) where T = Array{T, N+1}(undef, size(X)[3:end]..., n_samples(X))
# Initialize MatricialUniDataset by slicing across the features dimension
MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,2}) where T = Array{T, 1}(undef, n_samples(d))::MatricialUniDataset{T, 1}
MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,3}) where T = Array{T, 2}(undef, size(d)[1:end-1])::MatricialUniDataset{T, 2}
MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,4}) where T = Array{T, 3}(undef, size(d)[1:end-1])::MatricialUniDataset{T, 3}

# TODO use Xf[i,[(:) for i in 1:N]...]
# @computed @inline getFeature(X::OntologicalDataset{T,N}, idxs::AbstractVector{Integer}, feature::Integer) where T = X[idxs, feature, fill(:, N)...]::AbstractArray{T,N-1}

# TODO maybe using views can improve performances
# featureview(X::OntologicalDataset{T,0}, idxs::AbstractVector{Integer}, feature::Integer) = X.domain[idxs, feature]
# featureview(X::OntologicalDataset{T,1}, idxs::AbstractVector{Integer}, feature::Integer) = view(X.domain, idxs, feature, :)
# featureview(X::OntologicalDataset{T,2}, idxs::AbstractVector{Integer}, feature::Integer) = view(X.domain, idxs, feature, :, :)

################################################################################
# END Matricial dataset
################################################################################

# World generators/enumerators and array/set-like structures
# TODO test the functions for WorldSets with Sets and Arrays, and find the performance optimum
const AbstractWorldSet{W} = Union{AbstractVector{W},AbstractSet{W}} where {W<:AbstractWorld}
# Concrete type for sets: vectors are faster than sets, so we
# const WorldSet = AbstractSet{W} where W<:AbstractWorld
const WorldSet{W} = Vector{W} where {W<:AbstractWorld}
WorldSet{W}(S::WorldSet{W}) where {W<:AbstractWorld} = S

################################################################################
# BEGIN Test operators
################################################################################

abstract type TestOperator end
struct _TestOpNone  <: TestOperator end; const TestOpNone  = _TestOpNone();
# >=
struct _TestOpGeq  <: TestOperator end; const TestOpGeq  = _TestOpGeq();
# <
struct _TestOpLes  <: TestOperator end; const TestOpLes  = _TestOpLes();

dual_test_operator(::_TestOpGeq) = TestOpLes
dual_test_operator(::_TestOpLes) = TestOpGeq

primary_test_operator(x::_TestOpGeq) = TestOpGeq # x
primary_test_operator(x::_TestOpLes) = TestOpGeq # dual_test_operator(x)

polarity(x::_TestOpGeq) = true
polarity(x::_TestOpLes) = false

# >=_α
struct _TestOpGeqSoft  <: TestOperator
  alpha :: AbstractFloat
end;
const TestOpGeq_95  = _TestOpGeqSoft((Rational(95,100)));
const TestOpGeq_90  = _TestOpGeqSoft((Rational(90,100)));
const TestOpGeq_80  = _TestOpGeqSoft((Rational(80,100)));
const TestOpGeq_75  = _TestOpGeqSoft((Rational(75,100)));

# <_α
struct _TestOpLesSoft  <: TestOperator
  alpha :: AbstractFloat
end;
const TestOpLes_95  = _TestOpLesSoft((Rational(95,100)));
const TestOpLes_90  = _TestOpLesSoft((Rational(90,100)));
const TestOpLes_80  = _TestOpLesSoft((Rational(80,100)));
const TestOpLes_75  = _TestOpLesSoft((Rational(75,100)));

alpha(x::_TestOpGeqSoft) = x.alpha
alpha(x::_TestOpLesSoft) = x.alpha

dual_test_operator(x::_TestOpGeqSoft) = _TestOpLesSoft(1-alpha(x))
dual_test_operator(x::_TestOpLesSoft) = _TestOpGeqSoft(1-alpha(x))

primary_test_operator(x::_TestOpGeqSoft) = x
primary_test_operator(x::_TestOpLesSoft) = dual_test_operator(x)

# TODO use
polarity(x::_TestOpGeqSoft) = true
polarity(x::_TestOpLesSoft) = false

# dual_test_operator(::_TestOpGeqSoft{Val{Rational(95,100)}}) = TestOpLes_95
# dual_test_operator(::_TestOpLesSoft{Val{Rational(95,100)}}) = TestOpGeq_95
# dual_test_operator(::_TestOpGeqSoft{Val{Rational(90,100)}}) = TestOpLes_90
# dual_test_operator(::_TestOpLesSoft{Val{Rational(90,100)}}) = TestOpGeq_90
# dual_test_operator(::_TestOpGeqSoft{Val{Rational(80,100)}}) = TestOpLes_80
# dual_test_operator(::_TestOpLesSoft{Val{Rational(80,100)}}) = TestOpGeq_80
# dual_test_operator(::_TestOpGeqSoft{Val{Rational(75,100)}}) = TestOpLes_75
# dual_test_operator(::_TestOpLesSoft{Val{Rational(75,100)}}) = TestOpGeq_75


# alpha(::_TestOpGeqSoft{Val{Rational(95,100)}}) = Rational(95,100)
# alpha(::_TestOpLesSoft{Val{Rational(95,100)}}) = Rational(95,100)
# alpha(::_TestOpGeqSoft{Val{Rational(90,100)}}) = Rational(90,100)
# alpha(::_TestOpLesSoft{Val{Rational(90,100)}}) = Rational(90,100)
# alpha(::_TestOpGeqSoft{Val{Rational(80,100)}}) = Rational(80,100)
# alpha(::_TestOpLesSoft{Val{Rational(80,100)}}) = Rational(80,100)
# alpha(::_TestOpGeqSoft{Val{Rational(75,100)}}) = Rational(75,100)
# alpha(::_TestOpLesSoft{Val{Rational(75,100)}}) = Rational(75,100)

const all_ordered_test_operators = [
		TestOpGeq, TestOpLes,
		TestOpGeq_95, TestOpLes_95,
		TestOpGeq_90, TestOpLes_90,
		TestOpGeq_80, TestOpLes_80,
		TestOpGeq_75, TestOpLes_75,
	]
const all_test_operators_order = [
		TestOpGeq, TestOpLes,
		TestOpGeq_95, TestOpLes_95,
		TestOpGeq_90, TestOpLes_90,
		TestOpGeq_80, TestOpLes_80,
		TestOpGeq_75, TestOpLes_75,
	]
sort_test_operators!(x::Vector{TO}) where {TO<:TestOperator} = begin
	intersect(all_test_operators_order, x)
end

display_propositional_test(test_operator::_TestOpGeq, lhs::String, featval::Number) = "$(lhs) >= $(featval)"
display_propositional_test(test_operator::_TestOpLes, lhs::String, featval::Number) = "$(lhs) <= $(featval)"
display_propositional_test(test_operator::_TestOpGeqSoft, lhs::String, featval::Number) = "$(alpha(test_operator)*100)% [$(lhs) >= $(featval)]"
display_propositional_test(test_operator::_TestOpLesSoft, lhs::String, featval::Number) = "$(alpha(test_operator)*100)% [$(lhs) <= $(featval)]"

display_modal_test(modality::AbstractRelation, test_operator::ModalLogic.TestOperator, featid::Integer, featval::Number) = begin
	test = display_propositional_test(test_operator, "V$(featid)", featval)
	if modality != ModalLogic.RelationId
		"<$(ModalLogic.display_rel_short(modality))> ($test)"
	else
		"$test"
	end
end


@inline WExtrema(::_TestOpGeq, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = reverse(extrema(readWorld(w,channel)))
@inline WExtreme(::_TestOpGeq, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
	# println(_TestOpGeq)
	# println(w)
	# println(channel)
	# println(maximum(readWorld(w,channel)))
	# readline()
	maximum(readWorld(w,channel))
end
@inline WExtreme(::_TestOpLes, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
	# println(_TestOpLes)
	# println(w)
	# println(channel)
	# readline()
	minimum(readWorld(w,channel))
end


# TODO improved version for Rational numbers
# TODO check
@inline WExtrema(test_op::_TestOpGeqSoft, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
	vals = vec(readWorld(w,channel))
	x = partialsort!(vals,1+floor(Int, (1.0-alpha(test_op))*length(vals)))
	x,x
end
@inline WExtreme(test_op::_TestOpGeqSoft, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
	vals = vec(readWorld(w,channel))
	partialsort!(vals,1+floor(Int, (1.0-alpha(test_op))*length(vals)))
end
@inline WExtreme(test_op::_TestOpLesSoft, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
	vals = vec(readWorld(w,channel))
	partialsort!(vals,1+floor(Int, (alpha(test_op))*length(vals)))
end

# TODO remove
# @inline WMax(w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = maximum(readWorld(w,channel))
# @inline WMin(w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = minimum(readWorld(w,channel))

@inline TestCondition(test_operator::_TestOpGeq, w::AbstractWorld, channel::MatricialChannel{T,N}, featval::Number) where {T,N} = begin # TODO maybe this becomes SIMD, or sum/all(readWorld(w,channel)  .<= featval)
	# Source: https://stackoverflow.com/questions/47564825/check-if-all-the-elements-of-a-julia-array-are-equal
	# @inbounds
	for x in readWorld(w,channel)
		x >= featval || return false
	end
	return true
end
@inline TestCondition(test_operator::_TestOpLes, w::AbstractWorld, channel::MatricialChannel{T,N}, featval::Number) where {T,N} = begin # TODO maybe this becomes SIMD, or sum/all(readWorld(w,channel)  .<= featval)
	# Source: https://stackoverflow.com/questions/47564825/check-if-all-the-elements-of-a-julia-array-are-equal
	# @info "WLes" w featval #n readWorld(w,channel)
	# @inbounds
	for x in readWorld(w,channel)
		x <= featval || return false
	end
	return true
end

# Utility type for enhanced computation of thresholds
abstract type _ReprTreatment end
struct _ReprFake{worldType<:AbstractWorld}  <: _ReprTreatment w :: worldType end
struct _ReprMax{worldType<:AbstractWorld}  <: _ReprTreatment w :: worldType end
struct _ReprMin{worldType<:AbstractWorld}  <: _ReprTreatment w :: worldType end
struct _ReprVal{worldType<:AbstractWorld}  <: _ReprTreatment w :: worldType end
struct _ReprNone{worldType<:AbstractWorld} <: _ReprTreatment end

################################################################################
# END Test operators
################################################################################


# This constant is used to create the default world for each WorldType
#  (e.g. Interval(ModalLogic.emptyWorld) = Interval(-1,0))
struct _firstWorld end;    const firstWorld    = _firstWorld();
struct _emptyWorld end;    const emptyWorld    = _emptyWorld();
struct _centeredWorld end; const centeredWorld = _centeredWorld();

## Enumerate accessible worlds

# Fallback: enumAcc works with domains AND their dimensions
enumAcc(S::Any, r::AbstractRelation, channel::MatricialChannel{T,N}) where {T,N} = enumAcc(S, r, size(channel)...)
enumAccRepr(S::Any, r::AbstractRelation, channel::MatricialChannel{T,N}) where {T,N} = enumAccRepr(S, r, size(channel)...)
# Fallback: enumAcc for world sets maps to enumAcc-ing their elements
#  (note: one may overload this function to provide improved implementations for special cases (e.g. <L> of a world set in interval algebra))
enumAcc(S::AbstractWorldSet{WorldType}, r::AbstractRelation, XYZ::Vararg{Integer,N}) where {T,N,WorldType<:AbstractWorld} = begin
	IterTools.imap(WorldType,
		IterTools.distinct(Iterators.flatten((enumAccBare(w, r, XYZ...) for w in S)))
	)
end

# Ontology-agnostic relations:
# - Identity relation  (RelationId)    =  S -> S
# - None relation      (RelationNone)  =  Used as the "nothing" constant
# - Universal relation (RelationAll)   =  S -> all-worlds
struct _RelationId    <: AbstractRelation end; const RelationId   = _RelationId();
struct _RelationNone  <: AbstractRelation end; const RelationNone = _RelationNone();
struct _RelationAll   <: AbstractRelation end; const RelationAll  = _RelationAll();

enumAcc(w::WorldType,           ::_RelationId, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = [w]
enumAcc(S::AbstractWorldSet{W}, ::_RelationId, XYZ::Vararg{Integer,N}) where {W<:AbstractWorld,N} = S # TODO try IterTools.imap(identity, S) ?
# Maybe this will have a use: enumAccW1(w::AbstractWorld, ::_RelationId,   X::Integer) where T = [w] # IterTools.imap(identity, [w])

# TODO parametrize on test operator (any test operator in this case)
enumAccRepr(w::WorldType, ::_RelationId, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = [w]
WExtremaModal(test_operator::ModalLogic._TestOpGeq, w::WorldType, relation::_RelationId, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
	WExtrema(test_operator, w, channel)
WExtremeModal(test_operator::Union{ModalLogic._TestOpGeq,ModalLogic._TestOpLes}, w::WorldType, relation::_RelationId, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
	WExtreme(test_operator, w, channel)

display_rel_short(::_RelationId)  = "Id"
display_rel_short(::_RelationAll) = ""

# Perform the modal step, that is, evaluate a modal formula
#  on a domain, and eventually compute the new world set.
# TODO perhaps fastMode never needed, figure out
modalStep(S::WorldSetType,
					relation::R,
					channel::AbstractArray{T,N},
					test_operator::TestOperator,
					threshold::T) where {W<:AbstractWorld, WorldSetType<:Union{AbstractSet{W},AbstractVector{W}}, R<:AbstractRelation, T, N} = begin
	@logmsg DTDetail "modalStep" S relation display_modal_test(relation, test_operator, -1, threshold)
	satisfied = false
	worlds = enumAcc(S, relation, channel)
	if length(collect(Iterators.take(worlds, 1))) > 0
		new_worlds = WorldSetType()
		for w in worlds
			if TestCondition(test_operator, w, channel, threshold)
				@logmsg DTDetail " Found world " w readWorld(w,channel)
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

include("IntervalLogic.jl")

end # module
