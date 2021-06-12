module ModalLogic

using IterTools
import Base: argmax, argmin, size, show, convert, getindex, iterate, length
using Logging: @logmsg
using ..DecisionTree

using ComputedFieldTypes

using DataStructures

using BenchmarkTools # TODO only need this when testing and using @btime

export AbstractWorld, AbstractRelation,
				Ontology,
				WorldSet,
				display_propositional_test,
				display_modal_test,
				RelationAll, RelationNone, RelationId,
				world_type # TODO maybe remove this function?
				# enumAccessibles, enumAccRepr

# Fix
Base.keys(g::Base.Generator) = g.iter

# Abstract classes for world & relations
abstract type AbstractWorld end
abstract type AbstractRelation end

show(io::IO, r::AbstractRelation) = print(io, display_existential_modality(r))
display_existential_modality(r) = "⟨$(display_rel_short(r))⟩"

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
		return new{WorldType}(relationSet)
	end
	# Ontology(worldType, relationSet) = new(worldType, relationSet)
end

world_type(::Ontology{WT}) where {WT} = WT

# Actually, this will not work because relationSet does this collect(set(...)) thing... mh maybe better avoid that thing?
show(io::IO, o::Ontology{WorldType}) where {WorldType} = begin
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

export n_samples, n_attributes, n_features, channel_size, max_channel_size, n_frames,
				GenericDataset,
				AbstractModalDataset,
				OntologicalDataset, 
				MultiFrameOntologicalDataset,
				AbstractFeaturedWorldDataset, FeatModalDataset,
				MultiFrameFeatModalDataset,
				stumpFeatModalDataset,
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

const MatricialDataset{T,D}     = AbstractArray{T,D}
# const MatricialUniDataset{T,UD} = AbstractArray{T,UD}
const MatricialChannel{T,N}     = AbstractArray{T,N}
const MatricialInstance{T,MN}   = AbstractArray{T,MN}

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
getChannel(d::MatricialDataset{T,4},      idx_i::Integer, idx_a::Integer) where T = @views d[:, :, idx_f, idx_i]::MatricialChannel{T,2} # N=2
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

@computed struct OntologicalDataset{T, N, WorldType}
	
	ontology  :: Ontology{WorldType}
	
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

size(X::OntologicalDataset{T,N})             where {T,N} = size(X.domain)
size(X::OntologicalDataset{T,N}, i::Integer) where {T,N} = size(X.domain, i)
n_samples(X::OntologicalDataset{T,N})        where {T,N} = n_samples(X.domain)
n_attributes(X::OntologicalDataset{T,N})     where {T,N} = n_attributes(X.domain)
n_relations(X::OntologicalDataset{T,N})      where {T,N} = length(X.ontology.relationSet)
world_type(d::OntologicalDataset{T,N,WT}) where {T,N,WT} = WT

length(X::OntologicalDataset{T,N})        where {T,N} = n_samples(X)
Base.iterate(X::OntologicalDataset{T,N}, state=1)  where {T,N} = state > length(X) ? nothing : (getInstance(X, state), state+1) # Base.iterate(X.domain, state=state)
channel_size(X::OntologicalDataset{T,N})     where {T,N} = channel_size(X.domain)

getInstance(d::OntologicalDataset{T,N,WT}, args::Vararg) where {T,N,WT}  = getInstance(d.domain, args...)
getChannel(d::OntologicalDataset{T,N,WT},   args::Vararg) where {T,N,WT} = getChannel(d.domain, args...)

slice_dataset(d::OntologicalDataset{T,N,WT}, args::Vararg) where {T,N,WT} = OntologicalDataset{T, N, WT}(d.ontology, slice_dataset(d.domain, args...))

struct MultiFrameOntologicalDataset{T}
	frames  :: AbstractVector{<:OntologicalDataset{T}}
	MultiFrameOntologicalDataset(Xs::AbstractVector{<:OntologicalDataset{T}}) where {T} = begin
		@assert length(Xs) > 0 && length(unique(n_samples.(Xs))) == 1 "Can't create an empty MultiFrameOntologicalDataset or with mismatching number of samples (n_frames: $(length(Xs)), frame_sizes: $(n_samples.(Xs)))."
		new{T}(Xs)
	end
	# MultiFrameOntologicalDataset with same ontology for each frame
	MultiFrameOntologicalDataset(ontology::Ontology, Xs::AbstractVector{<:MatricialDataset{T}}) where {T} = begin
		MultiFrameOntologicalDataset([OntologicalDataset{T}(ontology, X) for X in Xs])
	end
end

# TODO: test all these methods
size(X::MultiFrameOntologicalDataset) = map(size, X.frames)
getindex(X::MultiFrameOntologicalDataset, i::Integer) = X.frames[i]
n_frames(X::MultiFrameOntologicalDataset)             = length(X.frames)
n_samples(X::MultiFrameOntologicalDataset)            = n_samples(X.frames[1]) # n_frames(X) > 0 ? n_samples(X.frames[1]) : 0
length(X::MultiFrameOntologicalDataset)               = n_samples(X)
frames(X::MultiFrameOntologicalDataset) = X.frames
Base.iterate(X::MultiFrameOntologicalDataset, state=1) = state > length(X) ? nothing : (getInstance(X, state), state+1)
n_attributes(X::MultiFrameOntologicalDataset) = sum(n_attributes.(X.frames))
n_attributes(X::MultiFrameOntologicalDataset, i_frame::Integer) = n_attributes(X.frames[i_frame])
# TODO: Note: channel_size doesn't make sense at this point. Only the acc_functions[i] functions.
n_relations(X::MultiFrameOntologicalDataset) = sum(n_relations.(X.frames))
n_relations(X::MultiFrameOntologicalDataset, i_frame::Integer) = n_relations(X.frames[i_frame])
world_types(d::MultiFrameOntologicalDataset) = world_type.(X.frames)
world_type(d::MultiFrameOntologicalDataset, i_frame::Integer) = world_type(X.frames[i_frame])

getInstance(X::MultiFrameOntologicalDataset,  i_frame::Integer, idx_i::Integer, args::Vararg)  = getInstance(X.frames[i], idx_i, args...)
slice_dataset(X::MultiFrameOntologicalDataset, i_frame::Integer, inds::AbstractVector{Integer}, args::Vararg)  = slice_dataset(X.frames[i], inds, args...)
getChannel(X::MultiFrameOntologicalDataset,   i_frame::Integer, idx_i::Integer, idx_f::Integer, args::Vararg)  = getChannel(X.frames[i], idx_i, idx_f, args...)

# getInstance(X::MultiFrameOntologicalDataset, idx_i::Integer, args::Vararg)  = getInstance(X.frames[i], idx_i, args...) # TODO should slice across the frames!
slice_dataset(X::MultiFrameOntologicalDataset{T}, inds::AbstractVector{<:Integer}, args::Vararg) where {T} = MultiFrameOntologicalDataset{T}(map(frame->slice_dataset(frame, inds, args...), X.frames))



abstract type AbstractFeaturedWorldDataset{T, WorldType} end

struct FeatModalDataset{T, WorldType} <: AbstractModalDataset{T, WorldType}
	
	# Core data
	fwd                :: AbstractFeaturedWorldDataset{T,WorldType}
	
	## Modal frame:
	# Accessibility relations
	relations          :: AbstractVector{<:AbstractRelation}
	
	# Worldset initialization functions (one per instance)
	#  with signature (w::WorldType, r::AbstractRelation) -> vs::AbstractVector{WorldType}
	initws_functions      :: AbstractVector{<:initWorldSetFunction}
	# Accessibility functions (one per instance)
	#  with signature (w::WorldType, r::AbstractRelation) -> vs::AbstractVector{WorldType}
	acc_functions      :: AbstractVector{<:accFunction}
	# Representative accessibility functions (one per instance)
	#  with signature (feature::FeatureTypeFun, aggregator::Aggregator, w::WorldType, r::AbstractRelation) -> vs::AbstractVector{WorldType}
	accrepr_functions  :: AbstractVector{<:accReprFunction}
	
	# Feature
	features      :: AbstractVector{<:FeatureTypeFun}

	# Test operators associated with each feature
	grouped_featsnops  :: AbstractVector{<:AbstractVector{<:TestOperatorFun}}
	
	FeatModalDataset(
		fwd                :: AbstractFeaturedWorldDataset{T,WorldType},
		relations          :: AbstractVector{<:AbstractRelation},
		initws_functions   :: AbstractVector{<:initWorldSetFunction},
		acc_functions      :: AbstractVector{<:accFunction},
		accrepr_functions  :: AbstractVector{<:accReprFunction},
		features           :: AbstractVector{<:FeatureTypeFun},
		grouped_featsnops  :: AbstractVector{<:AbstractVector{<:TestOperatorFun}},
	) where {T,WorldType} = begin FeatModalDataset{T, WorldType<:AbstractWorld}(fwd, relations, initws_functions, acc_functions, accrepr_functions, features, grouped_featsnops) end

	function FeatModalDataset{T, WorldType}(
		fwd                :: AbstractFeaturedWorldDataset{T,WorldType},
		relations          :: AbstractVector{<:AbstractRelation},
		initws_functions   :: AbstractVector{<:initWorldSetFunction},
		acc_functions      :: AbstractVector{<:accFunction},
		accrepr_functions  :: AbstractVector{<:accReprFunction},
		features           :: AbstractVector{<:FeatureTypeFun},
		grouped_featsnops  :: AbstractVector{<:AbstractVector{<:TestOperatorFun}},
	) where {T,WorldType<:AbstractWorld}
		@assert n_samples(fwd) > 0 "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with no instance. (fwd's type $(typeof(fwd)))"
		@assert length(grouped_featsnops) > 0 "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with no test operator."
		@assert n_samples(fwd) == length(initws_functions) "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of initws_functions $(length(initws_functions))."
		@assert n_samples(fwd) == length(acc_functions) "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of acc_functions $(length(acc_functions))."
		@assert n_samples(fwd) == length(accrepr_functions) "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of accrepr_functions $(length(accrepr_functions))."
		@assert n_features(fwd) == length(features) "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of features $(length(features))."
		new{T, WorldType}(fwd, relations, initws_functions, acc_functions, accrepr_functions, features, grouped_featsnops)
	end

	FeatModalDataset(
		X                  :: OntologicalDataset{T, N, WorldType},
		features           :: AbstractVector{<:FeatureTypeFun},
		grouped_featsnops  :: AbstractVector{<:AbstractVector{<:TestOperatorFun}}
	) where {T, N, WorldType<:AbstractWorld} = begin
		fwd = FeaturedWorldDataset(X, features);

		relations = X.ontology.relationSet

		# TODO optimize this! When the underlying MatricialDataset is an AbstractArray, this is going to be an array of a single function.
		# How to achievi this? Think about it.
		initws_functions = [(iC)->initWorldSet(iC, WorldType, inst_channel_size(instance)) for instance in X]
		acc_functions = [(w,R)->enumAccessibles(w,R,inst_channel_size(instance)...) for instance in X]
		accrepr_functions = [(f,a,w,R)->enumAccReprAggr(f,a,w,R,inst_channel_size(instance)...) for instance in X]

		FeatModalDataset{T, WorldType}(fwd, relations, initws_functions, acc_functions, accrepr_functions, features, grouped_featsnops)
	end

end

size(X::FeatModalDataset)             where {T,N} =  size(X.fwd)
n_samples(X::FeatModalDataset{T, WorldType}) where {T, WorldType}   = n_samples(X.fwd)
n_features(X::FeatModalDataset{T, WorldType}) where {T, WorldType}  = length(X.features)
n_relations(X::FeatModalDataset{T, WorldType}) where {T, WorldType} = length(X.relations)
# length(X::FeatModalDataset{T,WorldType})        where {T,WorldType} = n_samples(X)
# Base.iterate(X::FeatModalDataset{T,WorldType}, state=1) where {T, WorldType} = state > length(X) ? nothing : (getInstance(X, state), state+1)
getindex(X::FeatModalDataset{T,WorldType}, args::Vararg) where {T,WorldType} = getindex(X.fwd, args...)
world_type(X::FeatModalDataset{T,WorldType}) where {T,WorldType} = WorldType

slice_dataset(X::FeatModalDataset{T,WorldType}, inds::AbstractVector{<:Integer}, args::Vararg) where {T,WorldType} =
	FeatModalDataset{T,WorldType}(
		slice_dataset(X.fwd, inds, args...),
		X.relations,
		X.initws_functions[inds],
		X.acc_functions[inds],
		X.accrepr_functions[inds],
		X.features,
		X.grouped_featsnops,
	)

# TODO fix these
const AbstractFMDStumpSupport = AbstractArray
const AbstractFMDStumpGlobalSupport = AbstractArray

struct stumpFeatModalDataset{T, WorldType} <: AbstractModalDataset{T, WorldType}
	
	# Core data
	fmd                :: FeatModalDataset{T, WorldType}

	# Stump support
	fmd_m              :: AbstractFMDStumpSupport{T, WorldType}
	fmd_g              :: Union{AbstractFMDStumpGlobalSupport{T, WorldType},Nothing}

	# Features and Aggregators
	featnaggrs         :: AbstractVector{<:Tuple{FeatureTypeFun,<:Aggregator}}
	grouped_featnaggrs :: AbstractVector{<:AbstractVector{<:Aggregator}}

	function stumpFeatModalDataset{T, WorldType}(
		fmd                :: FeatModalDataset{T, WorldType},
		fmd_m              :: AbstractFMDStumpSupport{T, WorldType},
		fmd_g              :: Union{AbstractFMDStumpGlobalSupport{T, WorldType},Nothing},
		featnaggrs         :: AbstractVector{<:Tuple{FeatureTypeFun,<:Aggregator}},
		grouped_featnaggrs :: AbstractVector{<:AbstractVector{<:Aggregator}},
	) where {T,WorldType<:AbstractWorld}
		@assert n_samples(fmd) == n_samples(fmd_m) "Can't instantiate stumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_samples for fmd and fmd_m support: $(n_samples(fmd)) and $(n_samples(fmd_m))"
		# @assert somethinglike(fmd) == n_featnaggrs(fmd_m) "Can't instantiate stumpFeatModalDataset{$(T), $(WorldType)} with unmatching somethinglike for fmd and fmd_m support: $(somethinglike(fmd)) and $(n_featnaggrs(fmd_m))"
		@assert n_relations(fmd) == n_relations(fmd_m) "Can't instantiate stumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_relations for fmd and fmd_m support: $(n_relations(fmd)) and $(n_relations(fmd_m))"
		@assert world_type(fmd) == world_type(fmd_m) "Can't instantiate stumpFeatModalDataset{$(T), $(WorldType)} with unmatching world_type for fmd and fmd_m support: $(world_type(fmd)) and $(world_type(fmd_m))"

		if fmd_g != nothing
			@assert n_samples(fmd) == n_samples(fmd_g) "Can't instantiate stumpFeatModalDataset{$(T), $(WorldType)} with unmatching n_samples for fmd and fmd_g support: $(n_samples(fmd)) and $(n_samples(fmd_g))"
			# @assert somethinglike(fmd) == n_featnaggrs(fmd_g) "Can't instantiate stumpFeatModalDataset{$(T), $(WorldType)} with unmatching somethinglike for fmd and fmd_g support: $(somethinglike(fmd)) and $(n_featnaggrs(fmd_g))"
			@assert world_type(fmd) == world_type(fmd_g) "Can't instantiate stumpFeatModalDataset{$(T), $(WorldType)} with unmatching world_type for fmd and fmd_g support: $(world_type(fmd)) and $(world_type(fmd_g))"
		end

		new{T, WorldType}(fmd, fmd_m, fmd_g, featnaggrs, grouped_featnaggrs)
	end

	function stumpFeatModalDataset(
		fmd                :: FeatModalDataset{T, WorldType};
		computeRelationAll :: Bool = false,
	) where {T,WorldType<:AbstractWorld}
		stumpFeatModalDataset{T, WorldType}(fmd, computeRelationAll = computeRelationAll)
	end

	function stumpFeatModalDataset{T, WorldType}(
		fmd                :: FeatModalDataset{T, WorldType};
		computeRelationAll :: Bool = false,
	) where {T,WorldType<:AbstractWorld}
		
		grouped_featnaggrs = ModalLogic.prepare_featnaggrs(fmd.grouped_featsnops)
		
		featnaggrs = Tuple{FeatureTypeFun,<:Aggregator}[]

		for (feat,aggrs) in zip(fmd.features,grouped_featnaggrs)
			for aggr in aggrs
				push!(featnaggrs, (feat,aggr))
			end
		end

		relations = fmd.relations
		
		# Compute modal dataset propositions and 1-modal decisions
		fmd_m, fmd_g = computeModalDatasetStumpSupport(fmd, grouped_featnaggrs, computeRelationAll = computeRelationAll);

		stumpFeatModalDataset{T, WorldType}(fmd, fmd_m, fmd_g, featnaggrs, grouped_featnaggrs)
	end

	function stumpFeatModalDataset(
		X                  :: OntologicalDataset{T, N, WorldType},
		features           :: AbstractVector{<:FeatureTypeFun},
		grouped_featsnops  :: AbstractVector{<:AbstractVector{<:TestOperatorFun}};
		computeRelationAll :: Bool = false,
		timing_mode        :: Symbol = :time,
	) where {T, N, WorldType<:AbstractWorld}
		stumpFeatModalDataset{T, WorldType}(X, features, grouped_featsnops, computeRelationAll = computeRelationAll, timing_mode = timing_mode)
	end

	# function stumpFeatModalDataset{T, WorldType}(
	# 	X                  :: OntologicalDataset{T, N, WorldType},
	# 	features           :: AbstractVector{<:FeatureTypeFun},
	# 	grouped_featsnops  :: AbstractVector{<:AbstractVector{<:TestOperatorFun}};
	# 	computeRelationAll :: Bool = false,
	# 	timing_mode        :: Symbol = :time,
	# ) where {T, N, WorldType<:AbstractWorld}

	# 	# Compute modal dataset propositions
	# 	fmd = 
	# 		if timing_mode == :none
	# 			FeatModalDataset(X, features, grouped_featsnops);
	# 		elseif timing_mode == :time
	# 			@time FeatModalDataset(X, features, grouped_featsnops);
	# 		elseif timing_mode == :btime
	# 			@btime FeatModalDataset($X, $features, $grouped_featsnops);
	# 	end

	# 	relations = X.ontology.relationSet
		
	# 	grouped_featnaggrs = ModalLogic.prepare_featnaggrs(grouped_featsnops)

	# 	featnaggrs = Aggregator[]

	# 	for (feat,aggrs) in zip(fmd.features,featnaggrs)
	# 		for aggr in aggrs
	# 			push!(featnaggrs, (feat,aggr))
	# 		end
	# 	end
		
	# # Compute modal dataset propositions and 1-modal decisions
	# fmd_m, fmd_g = computeModalDatasetStumpSupport(X, relations, grouped_featnaggrs, fmd, features, computeRelationAll = computeRelationAll);


	# 	new{T, WorldType}(fmd, fmd_m, fmd_g, featnaggrs, grouped_featnaggrs)
	# end
end

size(X::stumpFeatModalDataset)             where {T,N} =  (size(X.fmd), size(X.fmd_m), size(X.fmd_g))
n_samples(X::stumpFeatModalDataset{T, WorldType}) where {T, WorldType}   = n_samples(X.fmd)
n_features(X::stumpFeatModalDataset{T, WorldType}) where {T, WorldType}  = length(X.fmd)
n_relations(X::stumpFeatModalDataset{T, WorldType}) where {T, WorldType} = length(X.fmd)
# getindex(X::stumpFeatModalDataset{T,WorldType}, args::Vararg) where {T,WorldType} = getindex(X.fmd, args...)
world_type(X::stumpFeatModalDataset{T,WorldType}) where {T,WorldType} = WorldType

slice_dataset(X::stumpFeatModalDataset{T,WorldType}, inds::AbstractVector{<:Integer}, args::Vararg) where {T,WorldType} =
	stumpFeatModalDataset{T,WorldType}(
		slice_dataset(X.fmd, inds, args...),
		slice_dataset(X.fmd_m, inds, args...),
		slice_dataset(X.fmd_g, inds, args...),
		X.featnaggrs,
		X.grouped_featnaggrs)


struct MultiFrameFeatModalDataset
	frames  :: AbstractVector{<:AbstractModalDataset}
	# function MultiFrameFeatModalDataset(Xs::AbstractVector{<:AbstractModalDataset{<:T, <:AbstractWorld}}) where {T}
	# function MultiFrameFeatModalDataset(Xs::AbstractVector{<:AbstractModalDataset{T, <:AbstractWorld}}) where {T}
	# function MultiFrameFeatModalDataset(Xs::AbstractVector{MD where MD<:AbstractModalDataset})
	function MultiFrameFeatModalDataset(Xs::AbstractVector{<:AbstractModalDataset{<:Real, <:AbstractWorld}})
		@assert length(Xs) > 0 && length(unique(n_samples.(Xs))) == 1 "Can't create an empty MultiFrameFeatModalDataset or with mismatching number of samples (n_frames: $(length(Xs)), frame_sizes: $(n_samples.(Xs)))."
		new(Xs)
	end
	# TODO write MultiFrameFeatModalDataset(Xs::AbstractVector{<:Tuple{Union{FeatModalDataset,MatricialDataset,OntologicalDataset},NamedTuple}}) = begin
end

# TODO: test all these methods
size(X::MultiFrameFeatModalDataset) = map(size, X.frames)
getindex(X::MultiFrameFeatModalDataset, i::Integer) = X.frames[i]
n_frames(X::MultiFrameFeatModalDataset)             = length(X.frames)
n_samples(X::MultiFrameFeatModalDataset)            = n_samples(X.frames[1]) # n_frames(X) > 0 ? n_samples(X.frames[1]) : 0
length(X::MultiFrameFeatModalDataset)               = n_samples(X)
Base.iterate(X::MultiFrameFeatModalDataset, state=1) = state > length(X) ? nothing : (getInstance(X, state), state+1)
frames(X::MultiFrameFeatModalDataset) = X.frames
# get total number of features (TODO: figure if this is useless or not)
n_features(X::MultiFrameFeatModalDataset) = sum(n_features.(X.frames))
# get number of features in a single frame
n_features(X::MultiFrameFeatModalDataset, i_frame::Integer) = n_features(X.frames[i_frame])
# TODO: Note: channel_size doesn't make sense at this point. Only the acc_functions[i] functions.
n_relations(X::MultiFrameFeatModalDataset) = sum(n_relations.(X.frames))
n_relations(X::MultiFrameFeatModalDataset, i_frame::Integer) = n_relations(X.frames[i_frame])
world_type(d::MultiFrameFeatModalDataset, i_frame::Integer) = world_type(X.frames[i_frame])
world_types(d::MultiFrameFeatModalDataset) = world_type.(X.frames)

getInstance(X::MultiFrameFeatModalDataset,  i_frame::Integer, idx_i::Integer, args::Vararg)  = getInstance(X.frames[i], idx_i, args...)
slice_dataset(X::MultiFrameFeatModalDataset, i_frame::Integer, inds::AbstractVector{<:Integer}, args::Vararg)  = slice_dataset(X.frames[i], inds, args...)
getChannel(X::MultiFrameFeatModalDataset,   i_frame::Integer, idx_i::Integer, idx_f::Integer, args::Vararg)  = getChannel(X.frames[i], idx_i, idx_f, args...)

# getInstance(X::MultiFrameFeatModalDataset, idx_i::Integer, args::Vararg)  = getInstance(X.frames[i], idx_i, args...) # TODO should slice across the frames!
slice_dataset(X::MultiFrameFeatModalDataset, inds::AbstractVector{<:Integer}, args::Vararg) =
	MultiFrameFeatModalDataset(map(frame->slice_dataset(frame, inds, args...), X.frames))


const GenericDataset = Union{MatricialDataset,OntologicalDataset,MultiFrameOntologicalDataset,AbstractModalDataset,MultiFrameFeatModalDataset}

################################################################################
# END Dataset types
################################################################################

include("testOperators.jl")

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
# TODO remove or rebrand?

# Utility type for enhanced computation of thresholds
abstract type _ReprTreatment end
struct _ReprFake{WorldType<:AbstractWorld} <: _ReprTreatment w :: WorldType end
struct _ReprMax{WorldType<:AbstractWorld}  <: _ReprTreatment w :: WorldType end
struct _ReprMin{WorldType<:AbstractWorld}  <: _ReprTreatment w :: WorldType end
struct _ReprVal{WorldType<:AbstractWorld}  <: _ReprTreatment w :: WorldType end
struct _ReprNone{WorldType<:AbstractWorld} <: _ReprTreatment end

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

# TODO rename into RelationGlobal (and then GlobalRelation?)
# Global relation    (RelationAll)   =  S -> all-worlds
struct _RelationAll   <: AbstractRelation end; const RelationAll  = _RelationAll();

display_rel_short(::_RelationAll) = "G"

# Shortcut for enumerating all worlds
enumAll(::Type{WorldType}, args::Vararg) where {WorldType<:AbstractWorld} = enumAccessibles(WorldType[], RelationAll, args...)
enumAll(::Type{WorldType}, enumAccFun::Function) where {WorldType<:AbstractWorld} = enumAccFun(WorldType[], RelationAll)
enumReprAll(::Type{WorldType}, enumReprFun::Function, f::FeatureTypeFun, a::Aggregator) where {WorldType<:AbstractWorld} = enumReprFun(f, a, WorldType[], RelationAll)


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


################################################################################
################################################################################

include("FeaturedWorldDataset.jl")

################################################################################
################################################################################

# TODO A relation can be defined as a union of other relations.
# In this case, thresholds can be computed by maximization/minimization of the
#  thresholds referred to the relations involved.
# abstract type AbstractRelation end
# struct _UnionOfRelations{T<:NTuple{N,<:AbstractRelation} where N} <: AbstractRelation end;

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
# TODO improve this function, and write it for different implementations of the dataset (MatricialChannel, or its generalization)
function modalStep(
		S::WorldSetType,
		relation::R,
		channel::AbstractArray{T,N},
		test_operator::TestOperator,
		threshold::T) where {W<:AbstractWorld, WorldSetType<:AbstractWorldSet{W}, R<:AbstractRelation, T, N}
	@logmsg DTDetail "modalStep" S relation display_modal_test(relation, test_operator, -1, threshold)
	satisfied = false
	worlds = enumAccessibles(S, relation, channel)
	if length(collect(Iterators.take(worlds, 1))) > 0
		new_worlds = WorldSetType()
		for w in worlds
			if testCondition(test_operator, w, channel, threshold)
				@logmsg DTDetail " Found world " w ch_readWorld(w, channel)
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

end # module
