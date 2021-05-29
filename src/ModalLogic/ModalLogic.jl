module ModalLogic

using IterTools
import Base: argmax, argmin, size, show, convert, getindex, iterate, length
using Logging: @logmsg
using ..DecisionTree

using ComputedFieldTypes

using DataStructures

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
display_existential_modality(r) = "⟨" * display_rel_short(r) * "⟩"

const accFunction = Function

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
				AbstractFeatModalDataset,
				MultiFrameFeatModalDataset,
				OntologicalDataset, 
				AbstractFeaturedWorldDataset,
				FeatModalDataset,
				FeatModalDatasetWithStumpSupport,
				MatricialInstance,
				MatricialDataset,
				# MatricialUniDataset,
				MatricialChannel,
				FeaturedWorldDataset

abstract type AbstractFeatModalDataset{T<:Real,WorldType<:AbstractWorld} end

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

getInstances(d::MatricialDataset{T,2}, inds::AbstractVector{<:Integer}; return_view = false) where T = if return_view @views d[:, inds]       else d[:, inds]    end # N=0
getInstances(d::MatricialDataset{T,3}, inds::AbstractVector{<:Integer}; return_view = false) where T = if return_view @views d[:, :, inds]    else d[:, :, inds] end # N=1
getInstances(d::MatricialDataset{T,4}, inds::AbstractVector{<:Integer}; return_view = false) where T = if return_view @views d[:, :, :, inds] else d[:, :, :, inds] end # N=2

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
	
	OntologicalDataset{T, N}(ontology::Ontology{WorldType}, domain::MatricialDataset{T,D}) where {T, N, D, WorldType<:AbstractWorld} = begin
		OntologicalDataset{T, N, WorldType}(ontology, domain)
	end
	
	OntologicalDataset{T, N, WorldType}(ontology::Ontology{WorldType}, domain::MatricialDataset{T,D}) where {T, N, D, WorldType<:AbstractWorld} = begin

		@assert n_samples(domain) > 0 "Can't instantiate OntologicalDataset{$(T), $(N), $(WorldType)} with no instance. (domain's type $(typeof(domain)))"
		@assert N == worldTypeDimensionality(WorldType) "ERROR! Dimensionality mismatch: can't interpret worldType $(WorldType) (dimensionality = $(worldTypeDimensionality(WorldType)) on MatricialDataset of dimensionality = $(N)"
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
length(X::OntologicalDataset{T,N})        where {T,N} = n_samples(X)
Base.iterate(X::OntologicalDataset{T,N}, state=1)  where {T,N} = state > length(X) ? nothing : (getInstance(X, state), state+1) # Base.iterate(X.domain, state=state)
channel_size(X::OntologicalDataset{T,N})     where {T,N} = channel_size(X.domain)

getInstance(d::OntologicalDataset{T,N,WT}, args::Vararg) where {T,N,WT}  = getInstance(d.domain, args...)
getInstances(d::OntologicalDataset{T,N,WT}, args::Vararg) where {T,N,WT} = getInstances(d.domain, args...)
getChannel(d::OntologicalDataset{T,N,WT},   args::Vararg) where {T,N,WT} = getChannel(d.domain, args...)

abstract type AbstractFeaturedWorldDataset{T, WorldType} end
abstract type AbstractFeatModalDatasetStumpSupport{T, WorldType} end
abstract type AbstractFeatModalDatasetStumpSupport_global{T, WorldType} end

struct FeatModalDataset{T, WorldType} <: AbstractFeatModalDataset{T, WorldType}
	
	# Core data
	fwd                :: AbstractFeaturedWorldDataset{T,WorldType}
	
	## Modal frame:
	# Accessibility relations
	relations          :: AbstractVector{<:AbstractRelation}
	
	# Accessibility functions (one per instance) with signature (w::WorldType, r::AbstractRelation) -> vs::AbstractVector{WorldType}
	enumAll_functions  :: AbstractVector{<:accFunction}
	acc_functions      :: AbstractVector{<:accFunction}
	accrepr_functions  :: AbstractVector{<:accFunction}
	
	# Test operators associated with each feature
	featsnops          :: AbstractVector{<:AbstractVector{<:TestOperatorFun}}
	
	# Feature names
	features           :: AbstractVector{<:FeatureTypeFun}

	FeatModalDataset(
		fwd                :: AbstractFeaturedWorldDataset{T,WorldType},
		relations          :: AbstractVector{<:AbstractRelation},
		enumAll_functions  :: AbstractVector{<:accFunction},
		acc_functions      :: AbstractVector{<:accFunction},
		accrepr_functions  :: AbstractVector{<:accFunction},
		featsnops          :: AbstractVector{<:AbstractVector{<:TestOperatorFun}},
		features           :: AbstractVector{<:FeatureTypeFun},
	) where {T,WorldType<:AbstractWorld} = begin FeatModalDataset{T, WorldType}(fwd, relations, enumAll_functions, acc_functions, accrepr_functions, featsnops, features) end

	FeatModalDataset(
		X                  :: OntologicalDataset{T, N, WorldType},
		features           :: AbstractVector{<:FeatureTypeFun},
		featsnops          :: AbstractVector{<:AbstractVector{<:TestOperatorFun}};
		timing_mode        :: Symbol = :none
	) where {T, N, WorldType<:AbstractWorld} = begin
		fwd = FeaturedWorldDataset(X, features);

		relations = X.ontology.relationSet

		# TODO optimize this! When the underlying MatricialDataset is an AbstractArray, this is going to be an array of a single function.
		# How to achievi this? Think about it.
		enumAll_functions = [()->enumAll(WorldType,inst_channel_size(instance)...) for instance in X]
		acc_functions = [(w,R)->enumAccessibles(w,R,inst_channel_size(instance)...) for instance in X]
		accrepr_functions = [(f,a,w,R)->enumAccReprAggr(f,a,w,R,inst_channel_size(instance)...) for instance in X]

		FeatModalDataset{T, WorldType}(fwd, relations, enumAll_functions, acc_functions, accrepr_functions, featsnops, features)
	end

	FeatModalDataset{T, WorldType}(
		fwd                :: AbstractFeaturedWorldDataset{T,WorldType},
		relations          :: AbstractVector{<:AbstractRelation},
		enumAll_functions  :: AbstractVector{<:accFunction},
		acc_functions      :: AbstractVector{<:accFunction},
		accrepr_functions  :: AbstractVector{<:accFunction},
		featsnops          :: AbstractVector{<:AbstractVector{<:TestOperatorFun}},
		features           :: AbstractVector{<:FeatureTypeFun},
	) where {T,WorldType<:AbstractWorld} = begin
		@assert n_samples(fwd) > 0 "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with no instance. (fwd's type $(typeof(fwd)))"
		@assert length(featsnops) > 0 "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with no test operator."
		@assert n_samples(fwd) == length(enumAll_functions) "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of enumAll_functions $(length(enumAll_functions))."
		@assert n_samples(fwd) == length(acc_functions) "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of acc_functions $(length(acc_functions))."
		@assert n_samples(fwd) == length(accrepr_functions) "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of accrepr_functions $(length(accrepr_functions))."
		@assert n_features(fwd) == length(features) "Can't instantiate FeatModalDataset{$(T), $(WorldType)} with different numbers of instances $(n_samples(fwd)) and of features $(length(features))."
		new{T, WorldType}(fwd, relations, enumAll_functions, acc_functions, accrepr_functions, featsnops, features)
	end
end

n_samples(X::FeatModalDataset{T, WorldType}) where {T, WorldType}  = n_samples(X.fwd)
n_features(X::FeatModalDataset{T, WorldType}) where {T, WorldType} = length(X.features)
# length(X::FeatModalDataset{T,WorldType})        where {T,WorldType} = n_samples(X)
# Base.iterate(X::FeatModalDataset{T,WorldType}, state=1) where {T, WorldType} = state > length(X) ? nothing : (getInstance(X, state), state+1)
getindex(X::FeatModalDataset{T,WorldType}, args::Vararg) where {T,WorldType} = getindex(X.fwd, args...)

# TODO move accordingly
const ModalDatasetChannelSliceType{T} = Union{AbstractArray{T},AbstractDict{<:AbstractWorld,T}}

struct FeatModalDatasetWithStumpSupport{T, WorldType} <: AbstractFeatModalDataset{T, WorldType}
	
	# FeaturedWorldDataset
	fmd                      :: AbstractFeatModalDataset{T, WorldType}
	
	# Feature names
	stump_support_relations  :: AbstractFeatModalDatasetStumpSupport{T, WorldType}
	stump_support_global     :: Union{AbstractFeatModalDatasetStumpSupport_global{T, WorldType},Nothing}

	# Features and their aggregators
	grouped_featnaggrs       :: AbstractVector{<:AbstractVector{Tuple{Integer,<:Aggregator}}}

	FeatModalDatasetWithStumpSupport(
			fmd                :: FeatModalDataset{T, WorldType},
			computeRelationAll :: Bool = false,
		) where {T, WorldType<:AbstractWorld} =
		FeatModalDatasetWithStumpSupport{T, WorldType}(fmd, computeRelationAll)
 
	FeatModalDatasetWithStumpSupport{T, WorldType}(
			fmd                :: FeatModalDataset{T, WorldType},
			computeRelationAll :: Bool = false,
		) where {T, WorldType<:AbstractWorld} = begin

		featsnops = fmd.featsnops
		relations = fmd.relations
		features  = fmd.features
		
		# TODO actually take care with unique!
		grouped_featnaggrs = Vector{Tuple{Integer,<:Aggregator}}[]

		i_aggregator = 1
		for (i_feature, test_operators) in enumerate(featsnops)
			cur_feataggrs = Tuple{Integer,<:Aggregator}[]
			for aggregator in unique(ModalLogic.existential_aggregator.(test_operators))
				push!(cur_feataggrs, (i_aggregator,aggregator))
				i_aggregator += 1
			end
			push!(grouped_featnaggrs, cur_feataggrs)
		end
		
		computeModalThreshold(
				cur_fwd      :: ModalDatasetChannelSliceType{T}, # IntervalFeaturedChannel{T} # AbstractFeaturedChannel{T}
				relation     :: AbstractRelation,
				w            :: WorldType,
				aggregator   :: Agg,
				accrepr_fun  :: Function,
				feature      :: FeatureTypeFun,
			) where {T, WorldType<:AbstractWorld, Agg<:Aggregator} = begin
			
			# TODO reintroduce the improvements for some operators: e.g. later. Actually, these can be simplified by using a set of representatives, as in some enumAccRepr!
			# worlds = fmd.acc_functions[i_instance](w, relation)

			worlds = accrepr_fun(feature, aggregator, w, relation)	
		
			# TODO try reduce(aggregator, worlds; init=ModalLogic.bottom(aggregator, T))
			# TODO remove this aggregator_to_binary...
			
			if length(worlds |> collect) == 0
				ModalLogic.aggregator_bottom(aggregator, T)
			else
				aggregator((w)->modalDatasetChannelSliceGet(cur_fwd, w), worlds)
			end

			# opt = aggregator_to_binary(aggregator)
			# threshold = ModalLogic.bottom(aggregator, T)
			# for w in worlds
			# 	e = modalDatasetChannelSliceGet(cur_fwd, w)
			# 	threshold = opt(threshold,e)
			# end
			# threshold
		end

		computeModalDatasetG =
			if RelationAll in relations
				relations = filter!(l->l≠RelationAll, relations)
				true
			elseif computeRelationAll
				true
			else
				false
		end

		n_instances = n_samples(fmd)
		n_relations = length(relations)

		# Prepare modalDatasetM
		stump_support_relations = initModalDataset_m(fmd.fwd, relations, grouped_featnaggrs)

		# Prepare modalDatasetG
		stump_support_global =
			if computeModalDatasetG
				initModalDataset_g(fmd.fwd, grouped_featnaggrs)
			else
				nothing
		end

		firstWorld = WorldType(ModalLogic.firstWorld)

		# Compute features

		@inbounds Threads.@threads for i_instance in 1:n_instances
			@logmsg DTDebug "Instance $(i_instance)/$(n_instances)"
			
			if i_instance == 1 || ((i_instance+1) % (floor(Int, ((n_instances)/5))+1)) == 0
				@logmsg DTOverview "Instance $(i_instance)/$(n_instances)"
			end
			
			accrepr_fun = fmd.accrepr_functions[i_instance]

			for w in fmd.enumAll_functions[i_instance]()
				initModalDatasetWorldSlice_m(stump_support_relations, w)
			end

			for (i_feature,aggregators) in enumerate(grouped_featnaggrs)
				
				@logmsg DTDebug "Feature $(i_feature)"
				
				cur_fwd = modalDatasetChannelSlice(fmd.fwd, i_instance, i_feature)

				# @logmsg DTDebug "instance" instance

				# Global relation (independent of the current world)
				if computeModalDatasetG
					@logmsg DTDebug "RelationAll"

					# TODO optimize: all aggregators are likely reading the same raw values.
					for (i_aggregator,aggregator) in aggregators
						
						threshold = computeModalThreshold(cur_fwd, RelationAll, firstWorld, aggregator, accrepr_fun, features[i_feature])
						
						# @logmsg DTDebug "Aggregator" aggregator threshold
						
						modalDatasetSet_g(stump_support_global, i_instance, i_aggregator, threshold)
					end
				end

				# Other relations
				for (i_relation,relation) in enumerate(relations)

					@logmsg DTDebug "Relation $(i_relation)/$(n_relations)"

					for w in fmd.enumAll_functions[i_instance]()
						
						@logmsg DTDebug "World" w
						
						# TODO optimize: all aggregators are likely reading the same raw values.
						for (i_aggregator,aggregator) in aggregators
							
							threshold = computeModalThreshold(cur_fwd, relation, w, aggregator, accrepr_fun, features[i_feature])

							# @logmsg DTDebug "Aggregator" aggregator threshold
							
							modalDatasetSet_m(stump_support_relations, i_instance, w, i_relation, i_aggregator, threshold)

						end
					end

				end
			end
		end
		new{T, WorldType}(fmd, stump_support_relations, stump_support_global, grouped_featnaggrs)
	end
end

n_samples(X::FeatModalDatasetWithStumpSupport{T, WorldType}) where {T, WorldType}  = n_samples(X.fmd)
n_features(X::FeatModalDatasetWithStumpSupport{T, WorldType}) where {T, WorldType} = length(X.fmd.features)
# length(X::FeatModalDatasetWithStumpSupport{T,WorldType})        where {T,WorldType} = n_samples(X)
# Base.iterate(X::FeatModalDatasetWithStumpSupport{T,WorldType}, state=1) where {T, WorldType} = state > length(X) ? nothing : (getInstance(X, state), state+1)
getindex(X::FeatModalDatasetWithStumpSupport{T,WorldType}, args::Vararg) where {T,WorldType} = getindex(X.fmd, args...)



struct MultiFrameFeatModalDataset
	frames  :: AbstractVector{<:FeatModalDataset}
	MultiFrameFeatModalDataset(Xs::AbstractVector{<:FeatModalDataset}) = begin
		@assert length(Xs) > 0 && length(unique(n_samples.(Xs))) == 1 "Can't create an empty Multi-Frame Modal Dataset or with mismatching number of samples (n_frames: $(length(Xs)), frame_sizes: $(n_samples.(Xs)))."
		new(Xs)
	end
	# Singleton
	MultiFrameFeatModalDataset(X::FeatModalDataset) = MultiFrameFeatModalDataset([X])
	# TODO write MultiFrameFeatModalDataset(Xs::AbstractVector{<:Tuple{Union{FeatModalDataset,MatricialDataset,OntologicalDataset},NamedTuple}}) = begin
end

# TODO: test all these methods
getindex(X::MultiFrameFeatModalDataset, i::Integer) = X.frames[i]
n_frames(X::MultiFrameFeatModalDataset)             = length(X.frames)
n_samples(X::MultiFrameFeatModalDataset)            = n_samples(X.frames[1]) # n_frames(X) > 0 ? n_samples(X.frames[1]) : 0
length(X::MultiFrameFeatModalDataset)               = n_samples(X)
Base.iterate(X::MultiFrameFeatModalDataset, state=1) = state > length(X) ? nothing : (getInstance(X, state), state+1)
# get total number of features (TODO: figure if this is useless or not)
n_features(X::MultiFrameFeatModalDataset) = sum(length.(X.frames))
# get number of features in a single frame
n_features(X::MultiFrameFeatModalDataset, i_frame::Integer) = n_features(X.frames[i_frame])
# TODO: Note: channel_size doesn't make sense at this point. Only the acc_functions[i] functions.

getInstance(X::MultiFrameFeatModalDataset,  i_frame::Integer, idx_i::Integer, args::Vararg)  = getInstance(X.frames[i], idx_i, args...)
getInstances(X::MultiFrameFeatModalDataset, i_frame::Integer, inds::AbstractVector{Integer}, args::Vararg)  = getInstances(X.frames[i], inds, args...)
getChannel(X::MultiFrameFeatModalDataset,   i_frame::Integer, idx_i::Integer, idx_f::Integer, args::Vararg)  = getChannel(X.frames[i], idx_i, idx_f, args...)

getInstance(X::MultiFrameFeatModalDataset, idx_i::Integer, args::Vararg)  = getInstance(X.frames[i], idx_i, args...) # TODO should slice across the frames!
getInstances(X::MultiFrameFeatModalDataset, inds::AbstractVector{Integer}, args::Vararg) = getInstances(X.frames[i], inds, args...) # TODO should slice across the frames!

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
struct _ReprFake{worldType<:AbstractWorld} <: _ReprTreatment w :: worldType end
struct _ReprMax{worldType<:AbstractWorld}  <: _ReprTreatment w :: worldType end
struct _ReprMin{worldType<:AbstractWorld}  <: _ReprTreatment w :: worldType end
struct _ReprVal{worldType<:AbstractWorld}  <: _ReprTreatment w :: worldType end
struct _ReprNone{worldType<:AbstractWorld} <: _ReprTreatment end

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
