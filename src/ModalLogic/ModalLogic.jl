module ModalLogic

using IterTools
import Base: argmax, argmin, size, show, convert, getindex, iterate, length
using Logging: @logmsg
using ..DecisionTree

using ComputedFieldTypes

using ResumableFunctions

using DataStructures

using BenchmarkTools # TODO only need this when testing and using @btime

export AbstractWorld, AbstractRelation,
				Ontology,
				AbstractWorldSet, WorldSet,
				display_decision,
				RelationGlob, RelationNone, RelationId,
				world_type, world_types # TODO maybe remove this function?
				# enumAccessibles, enumAccRepr

# Fix
Base.keys(g::Base.Generator) = g.iter

# Abstract classes for world & relations
abstract type AbstractWorld end
abstract type AbstractRelation end

show(io::IO, r::AbstractRelation) = print(io, display_existential_relation(r))
display_existential_relation(r) = "⟨$(display_rel_short(r))⟩"

const initWorldSetFunction = Function
const accFunction = Function
const accReprFunction = Function

# Concrete class for ontology models (world type + set of relations)
struct Ontology{WorldType<:AbstractWorld}
	relationSet :: AbstractVector{<:AbstractRelation}
	Ontology{WorldType}(relationSet) where {WorldType<:AbstractWorld} = begin
		relationSet = unique(relationSet)
		for relation in relationSet
			@assert goesWith(WorldType, relation) "Can't instantiate Ontology with WorldType $(WorldType) and relation $(relation)"
		end
		# if WorldType == OneWorld
		# 	relationSet = []
		# end
		return new{WorldType}(relationSet)
	end
	# Ontology(worldType, relationSet) = new(worldType, relationSet)
end

world_type(::Ontology{WT}) where {WT<:AbstractWorld} = WT

# Actually, this will not work because relationSet does this collect(set(...)) thing... mh maybe better avoid that thing?
show(io::IO, o::Ontology{WorldType}) where {WorldType} = begin
	if o == OneWorldOntology
		print(io, "OneWorldOntology")
	else
		print(io, "Ontology{")
		show(io, WorldType)
		print(io, "}(")
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
end

# strip_ontology(ontology::Ontology) = Ontology{OneWorld}(AbstractRelation[])


# This constant is used to create the default world for each WorldType
#  (e.g. Interval(emptyWorld) = Interval(-1,0))
struct _firstWorld end;    const firstWorld    = _firstWorld();
struct _emptyWorld end;    const emptyWorld    = _emptyWorld();
struct _centeredWorld end; const centeredWorld = _centeredWorld();

# World generators/enumerators and array/set-like structures
const AbstractWorldSet{W} = Union{AbstractVector{W},AbstractSet{W}} where {W<:AbstractWorld}
const WorldSet{W} = Vector{W} where {W<:AbstractWorld}
WorldSet{W}(S::WorldSet{W}) where {W<:AbstractWorld} = S

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

include("operators.jl")
include("featureTypes.jl")

################################################################################
# BEGIN Dataset types
################################################################################

export n_samples, n_attributes, n_features, n_relations,
				channel_size, max_channel_size,
				n_frames, frames, get_frame,
				display_structure,
				get_gamma, test_decision,
				##############################
				relations,
				initws_function,
				acc_function,
				accrepr_functions,
				features,
				grouped_featsaggrsnops,
				featsnaggrs,
				grouped_featsnaggrs,
				##############################
				SingleFrameGenericDataset,
				MultiFrameGenericDataset,
				GenericDataset,
				AbstractModalDataset,
				OntologicalDataset, 
				MultiFrameOntologicalDataset,
				AbstractFeaturedWorldDataset,
				FeatModalDataset,
				MultiFrameFeatModalDataset,
				StumpFeatModalDataset,
				MatricialInstance,
				MatricialDataset,
				# MatricialUniDataset,
				MatricialChannel,
				FeaturedWorldDataset

abstract type AbstractModalDataset{T<:Real,WorldType<:AbstractWorld} end

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

const MatricialDataset{T<:Number,D}     = AbstractArray{T,D}
# const MatricialUniDataset{T<:Number,UD} = AbstractArray{T,UD}
const MatricialChannel{T<:Number,N}     = AbstractArray{T,N}
const MatricialInstance{T<:Number,MN}   = AbstractArray{T,MN}

# TODO use d[i,[(:) for i in 1:N]...] for accessing it, instead of writing blocks of functions

n_samples(d::MatricialDataset{T,D})    where {T,D} = size(d, D)
n_attributes(d::MatricialDataset{T,D}) where {T,D} = size(d, D-1)
channel_size(d::MatricialDataset{T,D}) where {T,D} = size(d)[1:end-2]
# length(d::MatricialDataset{T,N})        where {T,N} = n_samples(d)
# Base.iterate(d::MatricialDataset{T,D}, state=1) where {T, D} = state > length(d) ? nothing : (getInstance(d, state), state+1)
max_channel_size = channel_size
# TODO rename channel_size into max_channel_size and define channel_size for single instance
# channel_size(d::MatricialDataset{T,2}, idx_i::Integer) where T = size(d[      1, idx_i])
# channel_size(d::MatricialDataset{T,3}, idx_i::Integer) where T = size(d[:,    1, idx_i])
# channel_size(d::MatricialDataset{T,4}, idx_i::Integer) where T = size(d[:, :, 1, idx_i])

# channel_size(d::MatricialDataset{T,D}, idx_i::Integer) where {T,D} = size(d[idx_i])[1:end-2]
inst_channel_size(inst::MatricialInstance{T,MN}) where {T,MN} = size(inst)[1:end-1]

getInstance(d::MatricialDataset{T,2},     idx::Integer) where T = @views d[:, idx]         # N=0
getInstance(d::MatricialDataset{T,3},     idx::Integer) where T = @views d[:, :, idx]      # N=1
getInstance(d::MatricialDataset{T,4},     idx::Integer) where T = @views d[:, :, :, idx]   # N=2

slice_dataset(d::MatricialDataset{T,2}, inds::AbstractVector{<:Integer}; return_view = false) where T = if return_view @views d[:, inds]       else d[:, inds]    end # N=0
slice_dataset(d::MatricialDataset{T,3}, inds::AbstractVector{<:Integer}; return_view = false) where T = if return_view @views d[:, :, inds]    else d[:, :, inds] end # N=1
slice_dataset(d::MatricialDataset{T,4}, inds::AbstractVector{<:Integer}; return_view = false) where T = if return_view @views d[:, :, :, inds] else d[:, :, :, inds] end # N=2

getChannel(d::MatricialDataset{T,2},      idx_i::Integer, idx_a::Integer) where T = @views d[      idx_a, idx_i]::T                     # N=0
getChannel(d::MatricialDataset{T,3},      idx_i::Integer, idx_a::Integer) where T = @views d[:,    idx_a, idx_i]::MatricialChannel{T,1} # N=1
getChannel(d::MatricialDataset{T,4},      idx_i::Integer, idx_a::Integer) where T = @views d[:, :, idx_a, idx_i]::MatricialChannel{T,2} # N=2
# getUniChannel(ud::MatricialUniDataset{T,1},  idx::Integer) where T = @views ud[idx]           # N=0
# getUniChannel(ud::MatricialUniDataset{T,2},  idx::Integer) where T = @views ud[:, idx]        # N=1
# getUniChannel(ud::MatricialUniDataset{T,3},  idx::Integer) where T = @views ud[:, :, idx]     # N=2
getInstanceAttribute(inst::MatricialInstance{T,1},      idx::Integer) where T = @views inst[      idx]::T                     # N=0
getInstanceAttribute(inst::MatricialInstance{T,2},      idx::Integer) where T = @views inst[:,    idx]::MatricialChannel{T,1} # N=1
getInstanceAttribute(inst::MatricialInstance{T,3},      idx::Integer) where T = @views inst[:, :, idx]::MatricialChannel{T,2} # N=2


# TODO maybe using views can improve performances
# @computed getChannel(X::OntologicalDataset{T,N}, idxs::AbstractVector{Integer}, attribute::Integer) where T = X[idxs, attribute, fill(:, N)...]::AbstractArray{T,N-1}
# attributeview(X::MatricialDataset{T,2}, idxs::AbstractVector{Integer}, attribute::Integer) = d[idxs, attribute]
# attributeview(X::MatricialDataset{T,3}, idxs::AbstractVector{Integer}, attribute::Integer) = view(d, idxs, attribute, :)
# attributeview(X::MatricialDataset{T,4}, idxs::AbstractVector{Integer}, attribute::Integer) = view(d, idxs, attribute, :, :)


# strip_domain(d::MatricialDataset{T,2}) where T = d  # N=0
# strip_domain(d::MatricialDataset{T,3}) where T = dropdims(d; dims=1)      # N=1
# strip_domain(d::MatricialDataset{T,4}) where T = dropdims(d; dims=(1,2))  # N=2

# Initialize MatricialUniDataset by slicing across the attribute dimension
# MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,2}) where T = Array{T, 1}(undef, n_samples(d))::MatricialUniDataset{T, 1}
# MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,3}) where T = Array{T, 2}(undef, size(d)[1:end-1])::MatricialUniDataset{T, 2}
# MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,4}) where T = Array{T, 3}(undef, size(d)[1:end-1])::MatricialUniDataset{T, 3}

get_gamma(X::MatricialDataset, i_instance::Integer, w::AbstractWorld, feature::FeatureTypeFun) =
	yieldFunction(feature)(inst_readWorld(w, getInstance(X, i_instance)))


@computed struct OntologicalDataset{T, N, WorldType} <: AbstractModalDataset{T, WorldType}
	
	ontology  :: Ontology{WorldType}
	
	# Core data
	domain    :: MatricialDataset{T,N+1+1}

	function OntologicalDataset(ontology::Ontology{WorldType}, domain::MatricialDataset{T,D}) where {T, N, D, WorldType<:AbstractWorld}
		OntologicalDataset{T}(ontology, domain)
	end

	function OntologicalDataset{T}(ontology::Ontology{WorldType}, domain::MatricialDataset{T,D}) where {T, D, WorldType<:AbstractWorld}
		OntologicalDataset{T, D-1-1}(ontology, domain)
	end

	function OntologicalDataset{T, N}(ontology::Ontology{WorldType}, domain::MatricialDataset{T,D}) where {T, N, D, WorldType<:AbstractWorld}
		@assert D == (N+1+1) "ERROR! Dimensionality mismatch: can't instantiate OntologicalDataset{$(T), $(N)} with MatricialDataset{$(T),$(D)}"
		OntologicalDataset{T, D-1-1, WorldType}(ontology, domain)
	end
	
	function OntologicalDataset{T, N, WorldType}(ontology::Ontology{WorldType}, domain::MatricialDataset{T,D}) where {T, N, D, WorldType<:AbstractWorld}

		@assert n_samples(domain) > 0 "Can't instantiate OntologicalDataset{$(T), $(N), $(WorldType)} with no instance. (domain's type $(typeof(domain)))"
		@assert N == worldTypeDimensionality(WorldType) "ERROR! Dimensionality mismatch: can't interpret WorldType $(WorldType) (dimensionality = $(worldTypeDimensionality(WorldType)) on MatricialDataset of dimensionality = $(N)"
		@assert D == (N+1+1) "ERROR! Dimensionality mismatch: can't instantiate OntologicalDataset{$(T), $(N)} with MatricialDataset{$(T),$(D)}"
		
		# Type unstable?
		# if prod(channel_size(domain)) == 1
		# 	ontology = strip_ontology(ontology)
		# 	WorldType = world_type(strip_ontology)
		# end
		
		new{T, N, WorldType}(ontology, domain)
	end
end

# TODO define getindex?

size(X::OntologicalDataset)             = size(X.domain)
size(X::OntologicalDataset, i::Integer) = size(X.domain, i)
n_samples(X::OntologicalDataset)        = n_samples(X.domain)
n_attributes(X::OntologicalDataset)     = n_attributes(X.domain)
n_relations(X::OntologicalDataset)      = length(X.ontology.relationSet)
world_type(d::OntologicalDataset{T,N,WT})    where {T,N,WT<:AbstractWorld} = WT

length(X::OntologicalDataset)                = n_samples(X)
Base.iterate(X::OntologicalDataset, state=1) = state > length(X) ? nothing : (getInstance(X, state), state+1) # Base.iterate(X.domain, state=state)
channel_size(X::OntologicalDataset)          = channel_size(X.domain)

getInstance(d::OntologicalDataset, args::Vararg)     = getInstance(d.domain, args...)
getChannel(d::OntologicalDataset,   args::Vararg)    = getChannel(d.domain, args...)

slice_dataset(d::OntologicalDataset, inds::AbstractVector{<:Integer}; args...)    = OntologicalDataset(d.ontology, slice_dataset(d.domain, inds; args...))

acc_function(X::OntologicalDataset, i_instance) = (w,R)->enumAccessibles(w,R, inst_channel_size(getInstance(X, i_instance))...)
accAll_function(X::OntologicalDataset{T, N, WorldType}, i_instance) where {T, N, WorldType} = enumAll(WorldType, acc_function(X, i_instance))

struct MultiFrameOntologicalDataset
	frames  :: AbstractVector{<:OntologicalDataset}

	function MultiFrameOntologicalDataset(Xs::AbstractVector{<:OntologicalDataset})
		@assert length(Xs) > 0 && length(unique(n_samples.(Xs))) == 1 "Can't create an empty MultiFrameOntologicalDataset or with mismatching number of samples (n_frames: $(length(Xs)), frame_sizes: $(n_samples.(Xs)))."
		new(Xs)
	end

	function MultiFrameOntologicalDataset(X::OntologicalDataset)
		MultiFrameOntologicalDataset([X])
	end

	# MultiFrameOntologicalDataset with same ontology for each frame
	function MultiFrameOntologicalDataset(ontology::Ontology, Xs::AbstractVector{<:MatricialDataset})
		MultiFrameOntologicalDataset([OntologicalDataset(ontology, X) for X in Xs])
	end
end

# TODO: test all these methods
size(X::MultiFrameOntologicalDataset) = map(size, X.frames)
get_frame(X::MultiFrameOntologicalDataset, i) = X.frames[i]
n_frames(X::MultiFrameOntologicalDataset)             = length(X.frames)
n_samples(X::MultiFrameOntologicalDataset)            = n_samples(X.frames[1]) # n_frames(X) > 0 ? n_samples(X.frames[1]) : 0
length(X::MultiFrameOntologicalDataset)               = n_samples(X)
frames(X::MultiFrameOntologicalDataset) = X.frames
Base.iterate(X::MultiFrameOntologicalDataset, state=1) = state > length(X) ? nothing : (getInstance(X, state), state+1)
n_attributes(X::MultiFrameOntologicalDataset) = map(n_attributes, X.frames)
n_attributes(X::MultiFrameOntologicalDataset, i_frame::Integer) = n_attributes(X.frames[i_frame])
# TODO: Note: channel_size doesn't make sense at this point. Only the acc_functions[i] functions.
n_relations(X::MultiFrameOntologicalDataset) = map(n_relations, X.frames)
n_relations(X::MultiFrameOntologicalDataset, i_frame::Integer) = n_relations(X.frames[i_frame])
world_types(X::MultiFrameOntologicalDataset) = world_type.(X.frames) # convert(Vector{<:Type{<:AbstractWorld}}, world_type.(X.frames))
world_type(X::MultiFrameOntologicalDataset, i_frame::Integer) = world_type(X.frames[i_frame])

getInstance(X::MultiFrameOntologicalDataset,  i_frame::Integer, idx_i::Integer, args::Vararg)  = getInstance(X.frames[i], idx_i, args...)
# slice_dataset(X::MultiFrameOntologicalDataset, i_frame::Integer, inds::AbstractVector{Integer}, args...)  = slice_dataset(X.frames[i], inds; args...)
getChannel(X::MultiFrameOntologicalDataset,   i_frame::Integer, idx_i::Integer, idx_a::Integer, args::Vararg)  = getChannel(X.frames[i], idx_i, idx_a, args...)

# getInstance(X::MultiFrameOntologicalDataset, idx_i::Integer, args::Vararg)  = getInstance(X.frames[i], idx_i, args...) # TODO should slice across the frames!
slice_dataset(X::MultiFrameOntologicalDataset, inds::AbstractVector{<:Integer}; args...) =
	MultiFrameOntologicalDataset(map(frame->slice_dataset(frame, inds; args...), X.frames))

display_structure(X::OntologicalDataset, indent::Integer = 0) = begin
	out = repeat(' ', indent) * "OntologicalDataset\t$(Base.summarysize(X) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t\t"
	out *= "(shape $(Base.size(X.domain)), # relations $(length(X.ontology.relationSet))"
	out *= "max_channel_size $(max_channel_size(X))"
	out
end


get_gamma(X::OntologicalDataset, args...) = get_gamma(X.domain, args...)

abstract type AbstractFeaturedWorldDataset{T, WorldType} end

struct FeatModalDataset{T, WorldType} <: AbstractModalDataset{T, WorldType}
	
	# Core data
	fwd                :: AbstractFeaturedWorldDataset{T,WorldType}
	
	## Modal frame:
	# Accessibility relations
	relations          :: AbstractVector{<:AbstractRelation}
	
	# Worldset initialization functions (one per instance)
	#  with signature (initCondition) -> vs::AbstractWorldSet{WorldType}
	initws_functions   :: AbstractVector{<:initWorldSetFunction}
	# Accessibility functions (one per instance)
	#  with signature (w::WorldType/AbstractWorldSet{WorldType}, r::AbstractRelation) -> vs::AbstractVector{WorldType}
	acc_functions      :: AbstractVector{<:accFunction}
	# Representative accessibility functions (one per instance)
	#  with signature (feature::FeatureTypeFun, aggregator::Aggregator, w::WorldType/AbstractWorldSet{WorldType}, r::AbstractRelation) -> vs::AbstractVector{WorldType}
	accrepr_functions  :: AbstractVector{<:accReprFunction}
	
	# Feature
	features           :: AbstractVector{<:FeatureTypeFun}

	# Test operators associated with each feature
	grouped_featsaggrsnops  :: AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}

	FeatModalDataset(
		fwd                :: AbstractFeaturedWorldDataset{T,WorldType},
		relations          :: AbstractVector{<:AbstractRelation},
		initws_functions   :: AbstractVector{<:initWorldSetFunction},
		acc_functions      :: AbstractVector{<:accFunction},
		accrepr_functions  :: AbstractVector{<:accReprFunction},
		features           :: AbstractVector{<:FeatureTypeFun},
		grouped_featsaggrsnops_or_featsnops, # AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
	) where {T,WorldType} = begin FeatModalDataset{T, WorldType}(fwd, relations, initws_functions, acc_functions, accrepr_functions, features, grouped_featsaggrsnops_or_featsnops) end

	function FeatModalDataset{T, WorldType}(
		fwd                :: AbstractFeaturedWorldDataset{T,WorldType},
		relations          :: AbstractVector{<:AbstractRelation},
		initws_functions   :: AbstractVector{<:initWorldSetFunction},
		acc_functions      :: AbstractVector{<:accFunction},
		accrepr_functions  :: AbstractVector{<:accReprFunction},
		features           :: AbstractVector{<:FeatureTypeFun},
		grouped_featsaggrsnops  :: AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}},
	) where {T,WorldType<:AbstractWorld}

		@assert n_samples(fwd) > 0 "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with no instance. (fwd's type $(typeof(fwd)))"
		@assert length(grouped_featsaggrsnops) > 0 && sum(length.(grouped_featsaggrsnops)) > 0 && sum(vcat([[length(test_ops) for test_ops in aggrs] for aggrs in grouped_featsaggrsnops]...)) > 0 "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with no test operator: grouped_featsaggrsnops"
		@assert n_samples(fwd) == length(initws_functions) "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of initws_functions $(length(initws_functions))."
		@assert n_samples(fwd) == length(acc_functions) "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of acc_functions $(length(acc_functions))."
		@assert n_samples(fwd) == length(accrepr_functions) "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of accrepr_functions $(length(accrepr_functions))."
		@assert n_features(fwd) == length(features) "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of features $(length(features))."
		new{T, WorldType}(fwd, relations, initws_functions, acc_functions, accrepr_functions, features, grouped_featsaggrsnops)
	end

	function FeatModalDataset(
		fwd                :: AbstractFeaturedWorldDataset{T,WorldType},
		relations          :: AbstractVector{<:AbstractRelation},
		initws_functions   :: AbstractVector{<:initWorldSetFunction},
		acc_functions      :: AbstractVector{<:accFunction},
		accrepr_functions  :: AbstractVector{<:accReprFunction},
		features           :: AbstractVector{<:FeatureTypeFun},
		grouped_featsnops  :: AbstractVector{<:AbstractVector{<:TestOperatorFun}},
	) where {T,WorldType<:AbstractWorld}

		grouped_featsaggrsnops = grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops)
 
		FeatModalDataset(fwd, relations, initws_functions, acc_functions, accrepr_functions, features, grouped_featsaggrsnops)
	end

	FeatModalDataset(
		X                  :: OntologicalDataset{T, N, WorldType},
		features           :: AbstractVector{<:FeatureTypeFun},
		grouped_featsaggrsnops_or_featsnops, #  :: AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}},
	) where {T, N, WorldType<:AbstractWorld} = begin
		fwd = FeaturedWorldDataset(X, features);

		relations = X.ontology.relationSet

		# TODO optimize this! When the underlying MatricialDataset is an AbstractArray, this is going to be an array of a single function.
		# How to achievi this? Think about it.
		initws_functions = [(iC)->initWorldSet(iC, WorldType, inst_channel_size(instance)) for instance in X]
		acc_functions = [(w,R)->enumAccessibles(w,R,inst_channel_size(instance)...) for instance in X]
		accrepr_functions = [(f,a,w,R)->enumAccReprAggr(f,a,w,R,inst_channel_size(instance)...) for instance in X]

		FeatModalDataset(fwd, relations, initws_functions, acc_functions, accrepr_functions, features, grouped_featsaggrsnops_or_featsnops)
	end

end

relations(X::FeatModalDataset)         = X.relations
initws_function(X::FeatModalDataset,  i_instance::Integer)  = X.initws_functions[i_instance]
acc_function(X::FeatModalDataset,     i_instance::Integer)  = X.acc_functions[i_instance]
accAll_function(X::FeatModalDataset{T, WorldType}, i_instance) where {T, WorldType} = enumAll(WorldType, acc_function(X, i_instance))
accrepr_function(X::FeatModalDataset, i_instance::Integer)  = X.accrepr_functions[i_instance]
features(X::FeatModalDataset)          = X.features
grouped_featsaggrsnops(X::FeatModalDataset) = X.grouped_featsaggrsnops

size(X::FeatModalDataset)             where {T,N} =  size(X.fwd)
n_samples(X::FeatModalDataset{T, WorldType}) where {T, WorldType}   = n_samples(X.fwd)
n_features(X::FeatModalDataset{T, WorldType}) where {T, WorldType}  = length(X.features)
n_relations(X::FeatModalDataset{T, WorldType}) where {T, WorldType} = length(X.relations)
# length(X::FeatModalDataset{T,WorldType})        where {T,WorldType} = n_samples(X)
# Base.iterate(X::FeatModalDataset{T,WorldType}, state=1) where {T, WorldType} = state > length(X) ? nothing : (getInstance(X, state), state+1)
getindex(X::FeatModalDataset{T,WorldType}, args::Vararg) where {T,WorldType} = getindex(X.fwd, args...)
world_type(X::FeatModalDataset{T,WorldType}) where {T,WorldType<:AbstractWorld} = WorldType


slice_dataset(X::FeatModalDataset{T,WorldType}, inds::AbstractVector{<:Integer}; args...) where {T,WorldType} =
	FeatModalDataset{T,WorldType}(
		slice_dataset(X.fwd, inds; args...),
		X.relations,
		X.initws_functions[inds],
		X.acc_functions[inds],
		X.accrepr_functions[inds],
		X.features,
		X.grouped_featsaggrsnops,
	)


function grouped_featsnops2grouped_featsaggrsnops(grouped_featsnops::AbstractVector{<:AbstractVector{<:TestOperatorFun}})::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}}
	grouped_featsaggrsnops = Dict{<:Aggregator,<:AbstractVector{<:TestOperatorFun}}[]
	for (i_feature, test_operators) in enumerate(grouped_featsnops)
		aggrsnops = Dict{Aggregator,AbstractVector{<:TestOperatorFun}}()
		for test_operator in test_operators
			aggregator = ModalLogic.existential_aggregator(test_operator)
			if (!haskey(aggrsnops, aggregator))
				aggrsnops[aggregator] = TestOperatorFun[]
			end
			push!(aggrsnops[aggregator], test_operator)
		end
		push!(grouped_featsaggrsnops, aggrsnops)
	end
	grouped_featsaggrsnops
end


find_feature_id(X::FeatModalDataset{T,WorldType}, feature::FeatureTypeFun) where {T,WorldType} =
	findall(x->x==feature, features(X))[1]
find_relation_id(X::FeatModalDataset{T,WorldType}, relation::AbstractRelation) where {T,WorldType} =
	findall(x->x==relation, relations(X))[1]

get_gamma(
		X::FeatModalDataset{T,WorldType},
		i_instance::Integer,
		w::WorldType,
		feature::FeatureTypeFun) where {WorldType<:AbstractWorld, T} = begin
	i_feature = find_feature_id(X, feature)
	X[i_instance, w, i_feature]
end


@resumable function generate_propositional_feasible_decisions(
		X::FeatModalDataset{T,WorldType},
		instances_inds::AbstractVector{<:Integer},
		Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
		features_inds::AbstractVector{<:Integer},
		) where {T, WorldType<:AbstractWorld}
	relation = RelationId
	n_instances = length(instances_inds)

	# For each feature
	for i_feature in features_inds
		feature = features(X)[i_feature]
		@logmsg DTDebug "Feature $(i_feature): $(feature)"

		# operators for each aggregator
		aggrsnops = grouped_featsaggrsnops(X)[i_feature]
		# Vector of aggregators
		aggregators = keys(aggrsnops) # Note: order-variant, but that's ok here
		
		# dict->vector
		# aggrsnops = [aggrsnops[i_aggr] for i_aggr in aggregators]

		# Initialize thresholds with the bottoms
		thresholds = Array{T,2}(undef, length(aggregators), n_instances)
		for (i_aggr,aggr) in enumerate(aggregators)
			thresholds[i_aggr,:] .= aggregator_bottom(aggr, T)
		end

		# For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
		for (i_instance,instance_id) in enumerate(instances_inds)
			@logmsg DTDetail " Instance $(i_instance)/$(n_instances)"
			worlds = Sf[i_instance]

			# TODO also try this instead
			# values = [X.fmd[instance_id, w, i_feature] for w in worlds]
			# thresholds[:,i_instance] = map(aggr->aggr(values), aggregators)
				
			for w in worlds
				gamma = X.fwd[instance_id, w, i_feature]
				for (i_aggr,aggr) in enumerate(aggregators)
					thresholds[i_aggr,i_instance] = ModalLogic.aggregator_to_binary(aggr)(gamma, thresholds[i_aggr,i_instance])
				end
			end
		end
		# @logmsg DTDebug "thresholds: " thresholds
		# For each aggregator
		for (i_aggr,aggr) in enumerate(aggregators)
			aggr_thresholds = thresholds[i_aggr,:]
			aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(T), typemax(T)]))
			for (i_test_operator,test_operator) in enumerate(aggrsnops[aggr])
				@logmsg DTDetail " Test operator $(test_operator)"
				# Look for the best threshold 'a', as in propositions like "feature >= a"
				for threshold in aggr_domain
					@logmsg DTDebug " Testing decision: $(display_decision(relation, feature, test_operator, threshold))"
					@yield (relation, feature, test_operator, threshold), aggr_thresholds
				end # for threshold
			end # for test_operator
		end # for aggregator
	end # for feature
end

abstract type AbstractFMDStumpSupport{T, WorldType} end
abstract type AbstractFMDStumpGlobalSupport{T} end

#  Stump support provides a structure for thresholds.
#   A threshold is the unique value γ for which w ⊨ <R> f ⋈ γ and:
#   if polarity(⋈) == true:      ∀ a > γ:    w ⊭ <R> f ⋈ a
#   if polarity(⋈) == false:     ∀ a < γ:    w ⊭ <R> f ⋈ a
#   for a given feature f, world w, relation R and feature f and test operator ⋈,

struct StumpFeatModalDataset{T, WorldType} <: AbstractModalDataset{T, WorldType}
	
	# Core data
	fmd                :: FeatModalDataset{T, WorldType}

	# Stump support
	fmd_m              :: AbstractFMDStumpSupport{T, WorldType}
	fmd_g              :: Union{AbstractFMDStumpGlobalSupport{T},Nothing}

	# Features and Aggregators
	featsnaggrs         :: AbstractVector{Tuple{<:FeatureTypeFun,<:Aggregator}}
	grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}}

	function StumpFeatModalDataset{T, WorldType}(
		fmd                :: FeatModalDataset{T, WorldType},
		fmd_m              :: AbstractFMDStumpSupport{T, WorldType},
		fmd_g              :: Union{AbstractFMDStumpGlobalSupport{T},Nothing},
		featsnaggrs         :: AbstractVector{Tuple{<:FeatureTypeFun,<:Aggregator}},
		grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
	) where {T,WorldType<:AbstractWorld}
		@assert n_samples(fmd) == n_samples(fmd_m) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_samples for fmd and fmd_m support: $(n_samples(fmd)) and $(n_samples(fmd_m))"
		@assert n_relations(fmd) == n_relations(fmd_m) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_relations for fmd and fmd_m support: $(n_relations(fmd)) and $(n_relations(fmd_m))"
		@assert world_type(fmd) == world_type(fmd_m) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching world_type for fmd and fmd_m support: $(world_type(fmd)) and $(world_type(fmd_m))"
		@assert sum(length.(grouped_featsnaggrs)) == length(featsnaggrs) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_featsnaggrs (grouped vs flattened structure): $(sum(length.(fmd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
		@assert sum(length.(fmd.grouped_featsaggrsnops)) == length(featsnaggrs) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_featsnaggrs for fmd and provided featsnaggrs: $(sum(length.(fmd.grouped_featsaggrsnops))) and $(length(featsnaggrs))"
		@assert sum(length.(fmd.grouped_featsaggrsnops)) == n_featsnaggrs(fmd_m) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_featsnaggrs for fmd and fmd_m support: $(sum(length.(fmd.grouped_featsaggrsnops))) and $(n_featsnaggrs(fmd_m))"

		if fmd_g != nothing
			@assert n_samples(fmd) == n_samples(fmd_g) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_samples for fmd and fmd_g support: $(n_samples(fmd)) and $(n_samples(fmd_g))"
			# @assert somethinglike(fmd) == n_featsnaggrs(fmd_g) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching somethinglike for fmd and fmd_g support: $(somethinglike(fmd)) and $(n_featsnaggrs(fmd_g))"
			# @assert world_type(fmd) == world_type(fmd_g) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching world_type for fmd and fmd_g support: $(world_type(fmd)) and $(world_type(fmd_g))"
			@assert sum(length.(fmd.grouped_featsaggrsnops)) == n_featsnaggrs(fmd_g) "Can't instantiate StumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_featsnaggrs for fmd and fmd_g support: $(sum(length.(fmd.grouped_featsaggrsnops))) and $(n_featsnaggrs(fmd_g))"
		end

		new{T, WorldType}(fmd, fmd_m, fmd_g, featsnaggrs, grouped_featsnaggrs)
	end

	function StumpFeatModalDataset(
		fmd                 :: FeatModalDataset{T, WorldType};
		computeRelationGlob :: Bool = false,
	) where {T,WorldType<:AbstractWorld}
		StumpFeatModalDataset{T, WorldType}(fmd, computeRelationGlob = computeRelationGlob)
	end

	function StumpFeatModalDataset{T, WorldType}(
		fmd                 :: FeatModalDataset{T, WorldType};
		computeRelationGlob :: Bool = false,
	) where {T,WorldType<:AbstractWorld}
		
		featsnaggrs = Tuple{<:FeatureTypeFun,<:Aggregator}[]
		grouped_featsnaggrs = AbstractVector{Tuple{<:Integer,<:Aggregator}}[]

		i_featsnaggr = 1
		for (feat,aggrsnops) in zip(fmd.features,fmd.grouped_featsaggrsnops)
			aggrs = []
			for aggr in keys(aggrsnops)
				push!(featsnaggrs, (feat,aggr))
				push!(aggrs, (i_featsnaggr,aggr))
				i_featsnaggr += 1
			end
			push!(grouped_featsnaggrs, aggrs)
		end

		# Compute modal dataset propositions and 1-modal decisions
		fmd_m, fmd_g = computeModalDatasetStumpSupport(fmd, grouped_featsnaggrs, computeRelationGlob = computeRelationGlob);

		StumpFeatModalDataset{T, WorldType}(fmd, fmd_m, fmd_g, featsnaggrs, grouped_featsnaggrs)
	end

	function StumpFeatModalDataset(
		X                   :: OntologicalDataset{T, N, WorldType},
		features            :: AbstractVector{<:FeatureTypeFun},
		grouped_featsnops   :: AbstractVector{<:AbstractVector{<:TestOperatorFun}};
		computeRelationGlob :: Bool = false,
	) where {T, N, WorldType<:AbstractWorld}
		StumpFeatModalDataset{T, WorldType}(X, features, grouped_featsnops, computeRelationGlob = computeRelationGlob)
	end

	function StumpFeatModalDataset{T, WorldType}(
		X                   :: OntologicalDataset{T, N, WorldType},
		features            :: AbstractVector{<:FeatureTypeFun},
		grouped_featsnops   :: AbstractVector{<:AbstractVector{<:TestOperatorFun}};
		computeRelationGlob :: Bool = false,
	) where {T, N, WorldType<:AbstractWorld}

		# Compute modal dataset propositions
		fmd = FeatModalDataset(X, features, grouped_featsnops);

		StumpFeatModalDataset{T, WorldType}(fmd, computeRelationGlob = computeRelationGlob)

		# TODO bring back ModalDatasetStumpSupport computation from X. 

		# fmd_m, fmd_g = computeModalDatasetStumpSupport(X, relations, fmd.grouped_featsaggrsnops??, fmd, features, computeRelationGlob = computeRelationGlob);

		# new{T, WorldType}(fmd, fmd_m, fmd_g, featsnaggrs, grouped_featsnaggrs)
	end
end

featsnaggrs(X::StumpFeatModalDataset)         = X.featsnaggrs
grouped_featsnaggrs(X::StumpFeatModalDataset) = X.grouped_featsnaggrs
relations(X::StumpFeatModalDataset)          = relations(X.fmd)
initws_function(X::StumpFeatModalDataset,  args...) = initws_function(X.fmd, args...)
acc_function(X::StumpFeatModalDataset,     args...) = acc_function(X.fmd, args...)
accAll_function(X::StumpFeatModalDataset, args...) = accAll_function(X.fmd, args...)
accrepr_function(X::StumpFeatModalDataset, args...) = accrepr_function(X.fmd, args...)
features(X::StumpFeatModalDataset)           = features(X.fmd)
grouped_featsaggrsnops(X::StumpFeatModalDataset)  = grouped_featsaggrsnops(X.fmd)


size(X::StumpFeatModalDataset)             where {T,N} =  (size(X.fmd), size(X.fmd_m), (isnothing(X.fmd_g) ? nothing : size(X.fmd_g)))
n_samples(X::StumpFeatModalDataset{T, WorldType}) where {T, WorldType}   = n_samples(X.fmd)
n_features(X::StumpFeatModalDataset{T, WorldType}) where {T, WorldType}  = n_features(X.fmd)
n_relations(X::StumpFeatModalDataset{T, WorldType}) where {T, WorldType} = n_relations(X.fmd)
# getindex(X::StumpFeatModalDataset{T,WorldType}, args::Vararg) where {T,WorldType} = getindex(X.fmd, args...)
world_type(X::StumpFeatModalDataset{T,WorldType}) where {T,WorldType<:AbstractWorld} = WorldType

slice_dataset(X::StumpFeatModalDataset{T,WorldType}, inds::AbstractVector{<:Integer}; args...) where {T,WorldType} =
	StumpFeatModalDataset{T,WorldType}(
		slice_dataset(X.fmd, inds; args...),
		slice_dataset(X.fmd_m, inds; args...),
		(isnothing(X.fmd_g) ? nothing : slice_dataset(X.fmd_g, inds; args...)),
		X.featsnaggrs,
		X.grouped_featsnaggrs)

display_structure(X::StumpFeatModalDataset, indent::Integer = 0) = begin
	out = repeat(' ', indent) * "\t\t$((Base.summarysize(X.fmd) + Base.summarysize(X.fmd_m) + Base.summarysize(X.fmd_g)) / 1024 / 1024 |> x->round(x, digits=2)) MBs\n"
	out *= repeat(' ', indent) * "├ fmd\t\t\t$(Base.summarysize(X.fmd) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t(shape $(Base.size(X.fmd.fwd)))\n"
	out *= repeat(' ', indent) * "├ fmd_m\t\t\t$(Base.summarysize(X.fmd_m) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t(shape $(Base.size(X.fmd_m)))\n"
	out *= repeat(' ', indent) * "└ fmd_g\t\t\t$(Base.summarysize(X.fmd_g) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
	if !isnothing(X.fmd_g)
		out *= "\t(shape $(Base.size(X.fmd_g)))"
	end
	out
end

find_feature_id(X::StumpFeatModalDataset{T,WorldType}, feature::FeatureTypeFun) where {T,WorldType} =
	findall(x->x==feature, features(X))[1]
find_relation_id(X::StumpFeatModalDataset{T,WorldType}, relation::AbstractRelation) where {T,WorldType} =
	findall(x->x==relation, relations(X))[1]
find_featsnaggr_id(X::StumpFeatModalDataset{T,WorldType}, feature::FeatureTypeFun, aggregator::Aggregator) where {T,WorldType} =
	findall(x->x==(feature, aggregator), featsnaggrs(X))[1]

get_gamma(
		X::StumpFeatModalDataset{T,WorldType},
		i_instance::Integer,
		w::WorldType,
		feature::FeatureTypeFun) where {WorldType<:AbstractWorld, T} = get_gamma(X.fmd, i_instance, w, feature)

get_global_gamma(
		X::StumpFeatModalDataset{T,WorldType},
		i_instance::Integer,
		feature::FeatureTypeFun,
		test_operator::TestOperatorFun) where {WorldType<:AbstractWorld, T} = begin
			@assert !isnothing(X.fmd_g) "Error. StumpFeatModalDataset must be built with computeRelationGlob = true for it to be ready to test global decisions."
			i_featsnaggr = find_featsnaggr_id(X, feature, existential_aggregator(test_operator))
			X.fmd_g[i_instance, i_featsnaggr]
end

get_modal_gamma(
		X::StumpFeatModalDataset{T,WorldType},
		i_instance::Integer,
		w::WorldType,
		relation::AbstractRelation,
		feature::FeatureTypeFun,
		test_operator::TestOperatorFun) where {WorldType<:AbstractWorld, T} = begin
			i_relation = find_relation_id(X, relation)
			i_featsnaggr = find_featsnaggr_id(X, feature, existential_aggregator(test_operator))
			X.fmd_m[i_instance, w, i_featsnaggr, i_relation]
end

test_decision(
		X::StumpFeatModalDataset{T,WorldType},
		i_instance::Integer,
		w::WorldType,
		relation::AbstractRelation,
		feature::FeatureTypeFun,
		test_operator::TestOperatorFun,
		threshold::T) where {WorldType<:AbstractWorld, T} = begin
	if relation == RelationId
		test_decision(X, i_instance, w, feature, test_operator, threshold)
	else
		gamma = if relation == RelationGlob
				get_global_gamma(X, i_instance, feature, test_operator)
			else
				get_modal_gamma(X, i_instance, w, relation, feature, test_operator)
		end
		evaluate_thresh_decision(test_operator, gamma, threshold)
	end
end

@resumable function generate_propositional_feasible_decisions(
		X::StumpFeatModalDataset{T,WorldType},
		args...
		) where {T, WorldType<:AbstractWorld}
		for decision in generate_propositional_feasible_decisions(X.fmd, args...)
			@yield decision
		end
end

@resumable function generate_global_feasible_decisions(
		X::StumpFeatModalDataset{T,WorldType},
		instances_inds::AbstractVector{<:Integer},
		Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
		features_inds::AbstractVector{<:Integer},
		) where {T, WorldType<:AbstractWorld}
	relation = RelationGlob
	n_instances = length(instances_inds)
	
	@assert !isnothing(X.fmd_g) "Error. StumpFeatModalDataset must be built with computeRelationGlob = true for it to be ready to generate global decisions."

	# For each feature
	for i_feature in features_inds
		feature = features(X)[i_feature]
		@logmsg DTDebug "Feature $(i_feature): $(feature)"

		# operators for each aggregator
		aggrsnops = grouped_featsaggrsnops(X)[i_feature]
		# println(aggrsnops)
		# Vector of aggregators
		aggregators_with_ids = grouped_featsnaggrs(X)[i_feature]
		# println(aggregators_with_ids)

		# dict->vector
		# aggrsnops = [aggrsnops[i_aggr] for i_aggr in aggregators]

		# # TODO use this optimized version:
		# 	thresholds can in fact be directly given by slicing fmd_g and permuting the two dimensions
		# aggregators_ids = fst.(aggregators_with_ids)
		# thresholds = transpose(X.fmd_g[instances_inds, aggregators_ids])

		# Initialize thresholds with the bottoms
		thresholds = Array{T,2}(undef, length(aggregators_with_ids), n_instances)
		for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)
			thresholds[i_aggr,:] .= aggregator_bottom(aggr, T)
		end
		
		# For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
		for (i_instance,instance_id) in enumerate(instances_inds)
			@logmsg DTDetail " Instance $(i_instance)/$(n_instances)"
			for (i_aggr,(i_featsnaggr,aggr)) in enumerate(aggregators_with_ids)
				gamma = X.fmd_g[instance_id, i_featsnaggr]
				thresholds[i_aggr,i_instance] = ModalLogic.aggregator_to_binary(aggr)(gamma, thresholds[i_aggr,i_instance])
				# println(gamma)
				# println(thresholds[i_aggr,i_instance])
			end
		end

		# println(thresholds)
		@logmsg DTDebug "thresholds: " thresholds

		# For each aggregator
		for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)

			# println(aggr)

			aggr_thresholds = thresholds[i_aggr,:]
			aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(T), typemax(T)]))

			for (i_test_operator,test_operator) in enumerate(aggrsnops[aggr])
				@logmsg DTDetail " Test operator $(test_operator)"
				
				# Look for the best threshold 'a', as in propositions like "feature >= a"
				for threshold in aggr_domain
					@logmsg DTDebug " Testing decision: $(display_decision(relation, feature, test_operator, threshold))"

					@yield (relation, feature, test_operator, threshold), aggr_thresholds
					
				end # for threshold
			end # for test_operator
		end # for aggregator
	end # for feature
end


@resumable function generate_modal_feasible_decisions(
		X::StumpFeatModalDataset{T,WorldType},
		instances_inds::AbstractVector{<:Integer},
		Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
		modal_relations_inds::AbstractVector{<:Integer},
		features_inds::AbstractVector{<:Integer},
		) where {T, WorldType<:AbstractWorld}
	n_instances = length(instances_inds)

	# For each relational operator
	for i_relation in modal_relations_inds
		relation = relations(X)[i_relation]
		@logmsg DTDebug "Relation $(relation)..."

		# For each feature
		for i_feature in features_inds
			feature = features(X)[i_feature]
			@logmsg DTDebug "Feature $(i_feature): $(feature)"

			# operators for each aggregator
			aggrsnops = grouped_featsaggrsnops(X)[i_feature]
			# Vector of aggregators
			aggregators_with_ids = grouped_featsnaggrs(X)[i_feature]

			# dict->vector
			# aggrsnops = [aggrsnops[i_aggr] for i_aggr in aggregators]

			# Initialize thresholds with the bottoms
			thresholds = Array{T,2}(undef, length(aggregators_with_ids), n_instances)
			for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)
				thresholds[i_aggr,:] .= aggregator_bottom(aggr, T)
			end

			# For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
				for (i_instance,instance_id) in enumerate(instances_inds)
				@logmsg DTDetail " Instance $(i_instance)/$(n_instances)"
				worlds = Sf[i_instance] # TODO could also use accrepr_functions here?

				# TODO also try this instead (TODO fix first)
				# values = [X.fmd_m[instance_id, w, i_feature] for w in worlds]
				# thresholds[:,i_instance] = map((_,aggr)->aggr(values), aggregators_with_ids)
					
				for (i_aggr,(i_featsnaggr,aggr)) in enumerate(aggregators_with_ids)
					for w in worlds
						gamma = X.fmd_m[instance_id, w, i_featsnaggr, i_relation]
						thresholds[i_aggr,i_instance] = ModalLogic.aggregator_to_binary(aggr)(gamma, thresholds[i_aggr,i_instance])
					end
				end
			end

			@logmsg DTDebug "thresholds: " thresholds

			# For each aggregator
			for (i_aggr,(_,aggr)) in enumerate(aggregators_with_ids)

				aggr_thresholds = thresholds[i_aggr,:]
				aggr_domain = setdiff(Set(aggr_thresholds),Set([typemin(T), typemax(T)]))

				for (i_test_operator,test_operator) in enumerate(aggrsnops[aggr])
					@logmsg DTDetail " Test operator $(test_operator)"
					
					# Look for the best threshold 'a', as in propositions like "feature >= a"
					for threshold in aggr_domain
						@logmsg DTDebug " Testing decision: $(display_decision(relation, feature, test_operator, threshold))"

						@yield (relation, feature, test_operator, threshold), aggr_thresholds
						
					end # for threshold
				end # for test_operator
			end # for aggregator
		end # for feature
	end # for relation
end

struct MultiFrameFeatModalDataset
	frames  :: AbstractVector{<:AbstractModalDataset}
	# function MultiFrameFeatModalDataset(Xs::AbstractVector{<:AbstractModalDataset{<:T, <:AbstractWorld}}) where {T}
	# function MultiFrameFeatModalDataset(Xs::AbstractVector{<:AbstractModalDataset{T, <:AbstractWorld}}) where {T}
	function MultiFrameFeatModalDataset(Xs::AbstractVector{<:AbstractModalDataset{<:Real, <:AbstractWorld}})
		@assert length(Xs) > 0 && length(unique(n_samples.(Xs))) == 1 "Can't create an empty MultiFrameFeatModalDataset or with mismatching number of samples (n_frames: $(length(Xs)), frame_sizes: $(n_samples.(Xs)))."
		new(Xs)
	end
	function MultiFrameFeatModalDataset(X::AbstractModalDataset{<:Real, <:AbstractWorld})
		new([X])
	end
end

# TODO: test all these methods
size(X::MultiFrameFeatModalDataset) = map(size, X.frames)
get_frame(X::MultiFrameFeatModalDataset, i) = X.frames[i]
n_frames(X::MultiFrameFeatModalDataset)             = length(X.frames)
n_samples(X::MultiFrameFeatModalDataset)            = n_samples(X.frames[1]) # n_frames(X) > 0 ? n_samples(X.frames[1]) : 0
length(X::MultiFrameFeatModalDataset)               = n_samples(X)
Base.iterate(X::MultiFrameFeatModalDataset, state=1) = state > length(X) ? nothing : (getInstance(X, state), state+1)
frames(X::MultiFrameFeatModalDataset) = X.frames
# get total number of features (TODO: figure if this is useless or not)
n_features(X::MultiFrameFeatModalDataset) = map(n_features, X.frames)
# get number of features in a single frame
n_features(X::MultiFrameFeatModalDataset, i_frame::Integer) = n_features(X.frames[i_frame])
# TODO: Note: channel_size doesn't make sense at this point. Only the acc_functions[i] functions.
n_relations(X::MultiFrameFeatModalDataset) = map(n_relations, X.frames)
n_relations(X::MultiFrameFeatModalDataset, i_frame::Integer) = n_relations(X.frames[i_frame])
world_type(X::MultiFrameFeatModalDataset, i_frame::Integer) = world_type(X.frames[i_frame])
world_types(X::MultiFrameFeatModalDataset) = world_type.(X.frames) # convert(Vector{<:Type{<:AbstractWorld}}, world_type.(X.frames))

getInstance(X::MultiFrameFeatModalDataset,  i_frame::Integer, idx_i::Integer, args::Vararg)  = getInstance(X.frames[i], idx_i, args...)
# slice_dataset(X::MultiFrameFeatModalDataset, i_frame::Integer, inds::AbstractVector{<:Integer}, args::Vararg)  = slice_dataset(X.frames[i], inds; args...)
getChannel(X::MultiFrameFeatModalDataset,   i_frame::Integer, idx_i::Integer, idx_f::Integer, args::Vararg)  = getChannel(X.frames[i], idx_i, idx_f, args...)

# getInstance(X::MultiFrameFeatModalDataset, idx_i::Integer, args::Vararg)  = getInstance(X.frames[i], idx_i, args...) # TODO should slice across the frames!
slice_dataset(X::MultiFrameFeatModalDataset, inds::AbstractVector{<:Integer}; args...) =
	MultiFrameFeatModalDataset(map(frame->slice_dataset(frame, inds; args...), X.frames))


const SingleFrameGenericDataset{T} = Union{MatricialDataset{T},OntologicalDataset{T},AbstractModalDataset{T}}
const MultiFrameGenericDataset = Union{MultiFrameOntologicalDataset,MultiFrameFeatModalDataset}
const GenericDataset = Union{SingleFrameGenericDataset,MultiFrameGenericDataset}

display_structure(Xs::MultiFrameGenericDataset, indent::Integer = 0) = begin
	out = repeat(' ', indent) * "$(typeof(Xs))\t\t\t$(Base.summarysize(Xs) / 1024 / 1024 |> x->round(x, digits=2)) MBs"
	for (i_frame, X) in enumerate(frames(Xs))
		if i_frame == n_frames(Xs)
			out *= "\n" * repeat(' ', indent) * "└ "
		else
			out *= "\n" * repeat(' ', indent) * "├ "
		end
		out *= "[$(i_frame)]\t\t\t$(Base.summarysize(X) / 1024 / 1024 |> x->round(x, digits=2)) MBs\t(world_type: $(world_type(X)))"
		out *= display_structure(X, indent+1) * "\n"
	end
	out
end


################################################################################
# END Dataset types
################################################################################

include("testOperators.jl")


display_propositional_decision(feature::FeatureTypeFun, test_operator::TestOperatorFun, threshold::Number) =
	"$(feature) $(test_operator) $(threshold)"

display_decision(relation::AbstractRelation, feature::FeatureTypeFun, test_operator::TestOperatorFun, threshold::Number) = begin
	propositional_decision = display_propositional_decision(feature, test_operator, threshold)
	if relation != RelationId
		"$(display_existential_relation(relation)) ($propositional_decision)"
	else
		"$propositional_decision"
	end
end

display_decision(i_frame::Integer, relation::AbstractRelation, feature::FeatureTypeFun, test_operator::TestOperatorFun, threshold::Number) = begin
	"{$i_frame} $(display_decision(relation, feature, test_operator, threshold))"
end

# TODO reason about shortened features

display_propositional_decision(feature::AttributeMinimumFeatureType, test_operator::typeof(>), threshold::Number) =
	"A$(feature.i_attr) ⫺ $(threshold)"

display_propositional_decision(feature::AttributeMaximumFeatureType, test_operator::typeof(<), threshold::Number) =
	"A$(feature.i_attr) ⫹ $(threshold)"


display_propositional_decision(feature::AttributeSoftMinimumFeatureType, test_operator::typeof(>), threshold::Number) =
	"A$(feature.i_attr) $("⫺" * subscriptnumber(rstrip(rstrip(string(alpha(feature)*100), '0'), '.'))) $(threshold)"

display_propositional_decision(feature::AttributeSoftMaximumFeatureType, test_operator::typeof(<), threshold::Number) =
	"A$(feature.i_attr) $("⫹" * subscriptnumber(rstrip(rstrip(string(alpha(feature)*100), '0'), '.'))) $(threshold)"

################################################################################
################################################################################

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

################################################################################
################################################################################

## Basic, ontology-agnostic relations:

# None relation      (RelationNone)  =  Used as the "nothing" constant
struct _RelationNone  <: AbstractRelation end; const RelationNone = _RelationNone();

# Identity relation  (RelationId)    =  S -> S
struct _RelationId    <: AbstractRelation end; const RelationId   = _RelationId();

enumAccessibles(w::WorldType,           ::_RelationId, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = [w] # IterTools.imap(identity, [w])
enumAccessibles(S::AbstractWorldSet{W}, ::_RelationId, XYZ::Vararg{Integer,N}) where {W<:AbstractWorld,N} = S # TODO try IterTools.imap(identity, S) ?

enumAccReprAggr(f::FeatureTypeFun, a::Aggregator, w::WorldType, r::AbstractRelation, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = enumAccessibles(w, r, XYZ...)
enumAccReprAggr(::FeatureTypeFun, ::Aggregator, w::WorldType, r::_RelationId, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = enumAccessibles(w, r, XYZ...)

# computeModalThresholdDual(test_operator::TestOperatorFun, w::WorldType, relation::_RelationId, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
# 	computePropositionalThresholdDual(test_operator, w, channel)
# computeModalThreshold(test_operator::TestOperatorFun, w::WorldType, relation::_RelationId, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
# 	computePropositionalThreshold(test_operator, w, channel)
# computeModalThresholdMany(test_ops::Vector{<:TestOperatorFun}, w::WorldType, relation::_RelationId, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
# 	computeModalThresholdMany(test_ops, w, channel)

display_rel_short(::_RelationId)  = "Id"


################################################################################
################################################################################
# TODO remove (needed for GAMMAS)
# Utility type for enhanced computation of thresholds
abstract type _ReprTreatment end
struct _ReprFake{WorldType<:AbstractWorld} <: _ReprTreatment w :: WorldType end
struct _ReprMax{WorldType<:AbstractWorld}  <: _ReprTreatment w :: WorldType end
struct _ReprMin{WorldType<:AbstractWorld}  <: _ReprTreatment w :: WorldType end
struct _ReprVal{WorldType<:AbstractWorld}  <: _ReprTreatment w :: WorldType end
struct _ReprNone{WorldType<:AbstractWorld} <: _ReprTreatment end
enumAccRepr(::_TestOpGeq, w::WorldType, ::_RelationId, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = _ReprMin(w)
enumAccRepr(::_TestOpLeq, w::WorldType, ::_RelationId, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = _ReprMax(w)

################################################################################
################################################################################
#
# Global relation    (RelationGlob)   =  S -> all-worlds
struct _RelationGlob   <: AbstractRelation end; const RelationGlob  = _RelationGlob();

display_rel_short(::_RelationGlob) = "G"

# Shortcut for enumerating all worlds
enumAll(::Type{WorldType}, args::Vararg) where {WorldType<:AbstractWorld} = enumAccessibles(WorldType[], RelationGlob, args...)
enumAll(::Type{WorldType}, enumAccFun::Function) where {WorldType<:AbstractWorld} = enumAccFun(WorldType[], RelationGlob)
enumReprAll(::Type{WorldType}, enumReprFun::Function, f::FeatureTypeFun, a::Aggregator) where {WorldType<:AbstractWorld} = enumReprFun(f, a, WorldType[], RelationGlob)


################################################################################
################################################################################

export OneWorldOntology,
				# genericIntervalOntology,
				IntervalOntology,
				Interval2DOntology,
				getIntervalOntologyOfDim,
				# genericIntervalRCC8Ontology,
				IntervalRCC8Ontology,
				Interval2DRCC8Ontology,
				getIntervalRCC8OntologyOfDim,
				getIntervalRCC5OntologyOfDim,
				IntervalRCC5Ontology,
				Interval2DRCC5Ontology

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

# abstract type OntologyType end

const OneWorldOntology   = Ontology{OneWorld}(AbstractRelation[])

# struct _genericIntervalOntology  <: OntologyType end; const genericIntervalOntology  = _genericIntervalOntology();
const IntervalOntology   = Ontology{Interval}(IARelations)
const Interval2DOntology = Ontology{Interval2D}(IA2DRelations)

# struct _genericIntervalRCC8Ontology  <: OntologyType end; const genericIntervalRCC8Ontology  = _genericIntervalRCC8Ontology();
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


################################################################################
################################################################################

include("BuildStumpSupport.jl")

computePropositionalThreshold(feature::FeatureTypeFun, w::AbstractWorld, instance::MatricialInstance) = yieldFunction(feature)(inst_readWorld(w, instance))


# TODO add AbstractWorldSet type
computeModalThreshold(fwd_propositional_slice::FeaturedWorldDatasetSlice{T}, worlds::Any, aggregator::Agg) where {T, Agg<:Aggregator} = begin
	
	# TODO try reduce(aggregator, worlds; init=ModalLogic.bottom(aggregator, T))
	# TODO remove this aggregator_to_binary...
	
	if length(worlds |> collect) == 0
		ModalLogic.aggregator_bottom(aggregator, T)
	else
		aggregator((w)->modalDatasetChannelSliceGet(fwd_propositional_slice, w), worlds)
	end

	# opt = aggregator_to_binary(aggregator)
	# threshold = ModalLogic.bottom(aggregator, T)
	# for w in worlds
	# 	e = modalDatasetChannelSliceGet(fwd_propositional_slice, w)
	# 	threshold = opt(threshold,e)
	# end
	# threshold
end

################################################################################
################################################################################

# TODO A relation can be defined as a union of other relations.
# In this case, thresholds can be computed by maximization/minimization of the
#  thresholds referred to the relations involved.
# abstract type AbstractRelation end
# struct _UnionOfRelations{T<:NTuple{N,<:AbstractRelation} where N} <: AbstractRelation end;

# computeModalThresholdDual(test_operator::TestOperatorFun, w::WorldType, relation::R where R<:_UnionOfRelations{relsTuple}, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
# 	computePropositionalThresholdDual(test_operator, w, channel)
# 	fieldtypes(relsTuple)
# computeModalThreshold(test_operator::TestOperatorFun, w::WorldType, relation::R where R<:_UnionOfRelations{relsTuple}, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
# 	computePropositionalThreshold(test_operator, w, channel)
# 	fieldtypes(relsTuple)

################################################################################
################################################################################

# Perform the modal step, that is, evaluate a modal formula
#  on a domain, and eventually compute the new world set.
function modal_step(
		X::Union{AbstractModalDataset{T,WorldType},OntologicalDataset{T,N,WorldType}},
		i_instance::Integer,
		worlds::WorldSetType,
		relation::AbstractRelation,
		feature::FeatureTypeFun,
		test_operator::TestOperatorFun,
		threshold::T) where {T, N, WorldType<:AbstractWorld, WorldSetType<:AbstractWorldSet{WorldType}}
	@logmsg DTDetail "modal_step" worlds display_decision(relation, feature, test_operator, threshold)

	satisfied = false
	
	# TODO space for optimization here: with some relations (e.g. IA_A, IA_L) can be made smaller

	if length(worlds) == 0
		# If there are no neighboring worlds, then the modal decision is not met
		@logmsg DTDetail "   No accessible world"
	else
		# Otherwise, check whether at least one of the accessible worlds witnesses truth of the decision.
		# TODO rewrite with new_worlds = map(...acc_worlds)
		# Initialize new worldset
		new_worlds = WorldSetType()

		# List all accessible worlds
		acc_worlds = acc_function(X, i_instance)(worlds, relation)

		for w in acc_worlds
			if test_decision(X, i_instance, w, feature, test_operator, threshold)
				# @logmsg DTDetail " Found world " w ch_readWorld(w, channel)
				satisfied = true
				push!(new_worlds, w)
			end
		end

		if satisfied == true
			worlds = new_worlds
		else 
			# If none of the neighboring worlds satisfies the decision, then 
			#  the new set is left unchanged
		end
	end
	if satisfied
		@logmsg DTDetail "   YES" worlds
	else
		@logmsg DTDetail "   NO"
	end
	return (satisfied, worlds)
end

test_decision(
		X::SingleFrameGenericDataset{T},
		i_instance::Integer,
		w::AbstractWorld,
		feature::FeatureTypeFun,
		test_operator::TestOperatorFun,
		threshold::T) where {T} = begin
	gamma = get_gamma(X, i_instance, w, feature)
	evaluate_thresh_decision(test_operator, gamma, threshold)
end

test_decision(
		X::SingleFrameGenericDataset{T},
		i_instance::Integer,
		w::AbstractWorld,
		relation::AbstractRelation,
		feature::FeatureTypeFun,
		test_operator::TestOperatorFun,
		threshold::T) where {T} = begin
	instance = getInstance(X, i_instance)

	aggregator = existential_aggregator(test_operator)
	
	worlds = enumAccReprAggr(feature, aggregator, w, relation, inst_channel_size(instance)...)
	gamma = if length(worlds |> collect) == 0
		ModalLogic.aggregator_bottom(aggregator, T)
	else
		aggregator((w)->get_gamma(X, i_instance, w, feature), worlds)
	end

	evaluate_thresh_decision(test_operator, gamma, threshold)
end


export generate_feasible_decisions,
				generate_propositional_feasible_decisions,
				generate_global_feasible_decisions,
				generate_modal_feasible_decisions

@resumable function generate_feasible_decisions(
		X::AbstractModalDataset{T,WorldType},
		instances_inds::AbstractVector{<:Integer},
		Sf::AbstractVector{<:AbstractWorldSet{WorldType}},
		allow_propositional_decisions::Bool,
		allow_modal_decisions::Bool,
		allow_global_decisions::Bool,
		modal_relations_inds::AbstractVector{<:Integer},
		features_inds::AbstractVector{<:Integer},
		) where {T, WorldType<:AbstractWorld}
	# Propositional splits
	if allow_propositional_decisions
		for decision in generate_propositional_feasible_decisions(X, instances_inds, Sf, features_inds)
			@yield decision
		end
	end
	# Global splits
	if allow_global_decisions
		for decision in generate_global_feasible_decisions(X, instances_inds, Sf, features_inds)
			@yield decision
		end
	end
	# Modal splits
	if allow_modal_decisions
		for decision in generate_modal_feasible_decisions(X, instances_inds, Sf, modal_relations_inds, features_inds)
			@yield decision
		end
	end
end

end # module
