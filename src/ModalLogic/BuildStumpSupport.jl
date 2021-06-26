
# FeaturedWorldDataset(
# 		X::OntologicalDataset{T, N, WorldType},
# 		features::AbstractVector{<:FeatureTypeFun}
# 	) where {T, N, WorldType<:AbstractWorld} = FeaturedWorldDataset{T, WorldType}(X, features)

FeaturedWorldDataset(
		X::OntologicalDataset,
		features::AbstractVector{<:FeatureTypeFun}
	) = begin

	@logmsg DTOverview "OntologicalDataset -> FeatModalDataset"

	n_instances = n_samples(X)
	n_features = length(features)
	
	# Prepare FeaturedWorldDataset (the actual implementation depends on the OntologicalDataset)
	fwd = initFeaturedWorldDataset(X, n_features)
	
	# Compute features
	@inbounds Threads.@threads for i_instance in 1:n_instances
		@logmsg DTDebug "Instance $(i_instance)/$(n_instances)"
		
		if i_instance == 1 || ((i_instance+1) % (floor(Int, ((n_instances)/5))+1)) == 0
			@logmsg DTOverview "Instance $(i_instance)/$(n_instances)"
		end

		# instance = getInstance(X, i_instance)
		# @logmsg DTDebug "instance" instance

		for w in accAll_function(X, i_instance)
			
			initFeaturedWorldDatasetWorldSlice(fwd, w)

			@logmsg DTDebug "World" w

			for (i_feature,feature) in enumerate(features)

				# threshold = computePropositionalThreshold(feature, w, instance)
				threshold = get_gamma(X, i_instance, w, feature)

				@logmsg DTDebug "Feature $(i_feature)" threshold
			
				modalDatasetSet(fwd, w, i_instance, i_feature, threshold)

			end
		end
	end
	fwd
end

# TODO write  different versions for generic,0-th,2-th dimensional cases...

# struct GenericFeaturedWorldDataset{T, WorldType} <: AbstractFeaturedWorldDataset{T, WorldType}
# 	d :: AbstractVector{<:AbstractDict{WorldType,T},2}
# end

# n_samples(fwd::GenericFeaturedWorldDataset{T, WorldType}) where {T, WorldType}  = size(fwd.d, 1)
# n_features(fwd::GenericFeaturedWorldDataset{T, WorldType}) where {T, WorldType} = size(fwd.d, 2)

# TODO
# function checkModalDatasetConsistency(modalDataset, X::OntologicalDataset{T, N, WorldType}, features::AbstractVector{<:FeatureTypeFun}) where {T, N, WorldType<:AbstractWorld}
# 	if !(modalDatasetIsConsistent(modalDataset, X, length(features)))
# 		error("The provided modalDataset structure is not consistent with the expected dataset, test operators and/or relations!"
# 			* "\n\tmodalDataset:"
# 			* " $(typeof(modalDataset))"
# 			* " $(eltype(modalDataset))"
# 			* " $(size(modalDataset))"
# 			* "\n\tX: $(n_samples(X))"
# 			* " $(n_attributes(X))"
# 			* " $(channel_size(X))"
# 			* "\n\tfeatures: $(size(features))"
# 		)
# 	end
# end


struct OneWorldFeaturedWorldDataset{T} <: AbstractFeaturedWorldDataset{T, OneWorld}
	d :: Array{T, 2}
end

# TODO rename these functions to underscore case
initFeaturedWorldDataset(X::OntologicalDataset{T, 0, OneWorld}, n_features::Integer) where {T} =
	OneWorldFeaturedWorldDataset{T}(Array{T, 2}(undef, n_samples(X), n_features))
modalDatasetIsConsistent(fwd::Any, X::OntologicalDataset{T, 0, OneWorld}, n_features::Integer) where {T} =
	(typeof(fwd)<:OneWorldFeaturedWorldDataset{T} && size(fwd) == (n_samples(X), n_features))
initFeaturedWorldDatasetWorldSlice(fwd::OneWorldFeaturedWorldDataset{T}, w::OneWorld) where {T} =
	nothing
modalDatasetSet(fwd::OneWorldFeaturedWorldDataset{T}, w::OneWorld, i_instance::Integer, i_feature::Integer, threshold::T) where {T} =
	fwd.d[i_instance, i_feature] = threshold
slice_dataset(fwd::OneWorldFeaturedWorldDataset{T}, inds::AbstractVector{<:Integer}; return_view = false) where {T} =
	OneWorldFeaturedWorldDataset{T}(if return_view @view fwd.d[inds,:] else fwd.d[inds,:] end)
modalDatasetChannelSlice(fwd::OneWorldFeaturedWorldDataset{T}, i_instance::Integer, i_feature::Integer) where {T} =
	fwd.d[i_instance, i_feature]
const OneWorldFeaturedChannel{T} = T
# const OneWorldFeaturedChannel{T} = Union{T}
modalDatasetChannelSliceGet(fwc::T #=OneWorldFeaturedChannel{T}=#, w::OneWorld) where {T} = fwc

n_samples(fwd::OneWorldFeaturedWorldDataset{T}) where {T}  = size(fwd, 1)
n_features(fwd::OneWorldFeaturedWorldDataset{T}) where {T} = size(fwd, 2)
getindex(
	fwd         :: OneWorldFeaturedWorldDataset{T},
	i_instance  :: Integer,
	w           :: OneWorld,
	i_feature   :: Integer) where {T} = fwd.d[i_instance, i_feature]
size(fwd::OneWorldFeaturedWorldDataset{T}, args::Vararg) where {T} = size(fwd.d, args...)
world_type(fwd::OneWorldFeaturedWorldDataset{T}) where {T} = OneWorld


struct IntervalFeaturedWorldDataset{T} <: AbstractFeaturedWorldDataset{T, Interval}
	d :: Array{T, 4}
end

# TODO rename these functions to underscore case
initFeaturedWorldDataset(X::OntologicalDataset{T, 1, Interval}, n_features::Integer) where {T} =
	IntervalFeaturedWorldDataset{T}(Array{T, 4}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), n_features))
modalDatasetIsConsistent(fwd::Any, X::OntologicalDataset{T, 1, Interval}, n_features::Integer) where {T} =
	(typeof(fwd)<:IntervalFeaturedWorldDataset{T} && size(fwd) == (max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), n_features))
initFeaturedWorldDatasetWorldSlice(fwd::IntervalFeaturedWorldDataset{T}, w::Interval) where {T} =
	nothing
modalDatasetSet(fwd::IntervalFeaturedWorldDataset{T}, w::Interval, i_instance::Integer, i_feature::Integer, threshold::T) where {T} =
	fwd.d[w.x, w.y, i_instance, i_feature] = threshold
slice_dataset(fwd::IntervalFeaturedWorldDataset{T}, inds::AbstractVector{<:Integer}; return_view = false) where {T} =
	IntervalFeaturedWorldDataset{T}(if return_view @view fwd.d[:,:,inds,:] else fwd.d[:,:,inds,:] end)
modalDatasetChannelSlice(fwd::IntervalFeaturedWorldDataset{T}, i_instance::Integer, i_feature::Integer) where {T} =
	@views fwd.d[:,:,i_instance, i_feature]
const IntervalFeaturedChannel{T} = AbstractArray{T, 2}
modalDatasetChannelSliceGet(fwc::IntervalFeaturedChannel{T}, w::Interval) where {T} =
	fwc[w.x, w.y]

n_samples(fwd::IntervalFeaturedWorldDataset{T}) where {T}  = size(fwd, 3)
n_features(fwd::IntervalFeaturedWorldDataset{T}) where {T} = size(fwd, 4)
getindex(
	fwd         :: IntervalFeaturedWorldDataset{T},
	i_instance  :: Integer,
	w           :: Interval,
	i_feature   :: Integer) where {T} = fwd.d[w.x, w.y, i_instance, i_feature]
size(fwd::IntervalFeaturedWorldDataset{T}, args::Vararg) where {T} = size(fwd.d, args...)
world_type(fwd::IntervalFeaturedWorldDataset{T}) where {T} = Interval


struct Interval2DFeaturedWorldDataset{T} <: AbstractFeaturedWorldDataset{T, Interval2D}
	d :: Array{T, 6}
end

# TODO rename these functions to underscore case
initFeaturedWorldDataset(X::OntologicalDataset{T, 2, Interval2D}, n_features::Integer) where {T} =
	Interval2DFeaturedWorldDataset{T}(Array{T, 6}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, max_channel_size(X)[2], max_channel_size(X)[2]+1, n_samples(X), n_features))
modalDatasetIsConsistent(fwd::Any, X::OntologicalDataset{T, 2, Interval2D}, n_features::Integer) where {T} =
	(typeof(fwd)<:Interval2DFeaturedWorldDataset{T} && size(fwd) == (max_channel_size(X)[1], max_channel_size(X)[1]+1, max_channel_size(X)[2], max_channel_size(X)[2]+1, n_samples(X), n_features))
initFeaturedWorldDatasetWorldSlice(fwd::Interval2DFeaturedWorldDataset{T}, w::Interval2D) where {T} =
	nothing
modalDatasetSet(fwd::Interval2DFeaturedWorldDataset{T}, w::Interval2D, i_instance::Integer, i_feature::Integer, threshold::T) where {T} =
	fwd.d[w.x.x, w.x.y, w.y.x, w.y.y, i_instance, i_feature] = threshold
slice_dataset(fwd::Interval2DFeaturedWorldDataset{T}, inds::AbstractVector{<:Integer}; return_view = false) where {T} =
	Interval2DFeaturedWorldDataset{T}(if return_view @view fwd.d[:,:,:,:,inds,:] else fwd.d[:,:,:,:,inds,:] end)
modalDatasetChannelSlice(fwd::Interval2DFeaturedWorldDataset{T}, i_instance::Integer, i_feature::Integer) where {T} =
	@views fwd.d[:,:,:,:,i_instance, i_feature]
const Interval2DFeaturedChannel{T} = AbstractArray{T, 4}
modalDatasetChannelSliceGet(fwc::Interval2DFeaturedChannel{T}, w::Interval2D) where {T} =
	fwc[w.x.x, w.x.y, w.y.x, w.y.y]

n_samples(fwd::Interval2DFeaturedWorldDataset{T}) where {T}  = size(fwd, 5)
n_features(fwd::Interval2DFeaturedWorldDataset{T}) where {T} = size(fwd, 6)
getindex(
	fwd         :: Interval2DFeaturedWorldDataset{T},
	i_instance  :: Integer,
	w           :: Interval2D,
	i_feature   :: Integer) where {T} = fwd.d[w.x.x, w.x.y, w.y.x, w.y.y, i_instance, i_feature]
size(fwd::Interval2DFeaturedWorldDataset{T}, args::Vararg) where {T} = size(fwd.d, args...)
world_type(fwd::Interval2DFeaturedWorldDataset{T}) where {T} = Interval2D



const FeaturedWorldDatasetSlice{T} = Union{
	# FeaturedWorldDatasetSlice(OntologicalDataset{T where T, 0, ModalLogic.OneWorld})
	T, # OneWorldFeaturedChannel{T},
	IntervalFeaturedChannel{T},
	Interval2DFeaturedChannel{T},
	# FeaturedWorldDatasetSlice(OntologicalDataset{T where T, 2, Interval2D})
}


################################################################################
################################################################################
################################################################################
################################################################################


# function prepare_featsnaggrs(grouped_featsnops::AbstractVector{<:AbstractVector{<:TestOperatorFun}})
	
# 	# Pairs of feature ids + set of aggregators
# 	grouped_featsnaggrs = Vector{<:Aggregator}[
# 		ModalLogic.existential_aggregator.(test_operators) for (i_feature, test_operators) in enumerate(grouped_featsnops)
# 	]

# 	# grouped_featsnaggrs = [grouped_featsnaggrs[i_feature] for i_feature in 1:length(features)]

# 	# # Flatten dictionary, and enhance aggregators in dictionary with their relative indices
# 	# flattened_featsnaggrs = Tuple{<:FeatureTypeFun,<:Aggregator}[]
# 	# i_featsnaggr = 1
# 	# for (i_feature, aggregators) in enumerate(grouped_featsnaggrs)
# 	# 	for aggregator in aggregators
# 	# 		push!(flattened_featsnaggrs, (features[i_feature],aggregator))
# 	# 		i_featsnaggr+=1
# 	# 	end
# 	# end

# 	grouped_featsnaggrs
# end


# worldType-agnostic
struct GenericFMDStumpSupport{T, WorldType} <: AbstractFMDStumpSupport{T, WorldType}
	# d :: AbstractArray{<:AbstractDict{WorldType,T}, 3}
	d :: AbstractArray{Dict{WorldType,T}, 3}
end

n_samples(fmds::GenericFMDStumpSupport)  = size(fmds, 1)
n_featsnaggrs(fmds::GenericFMDStumpSupport) = size(fmds, 2)
n_relations(fmds::GenericFMDStumpSupport) = size(fmds, 3)
getindex(
	fmds         :: GenericFMDStumpSupport{T, WorldType},
	i_instance   :: Integer,
	w            :: WorldType,
	i_featsnaggr :: Integer,
	i_relation   :: Integer) where {T, WorldType<:AbstractWorld} = fmds.d[i_instance, i_featsnaggr, i_relation][w]
size(fmds::GenericFMDStumpSupport, args::Vararg) = size(fmds.d, args...)
world_type(fmds::GenericFMDStumpSupport{T, WorldType}) where {T, WorldType} = WorldType

initFMDStumpSupport(fmd::FeatModalDataset{T, WorldType}, n_featsnaggrs::Integer, n_relations::Integer) where {T, WorldType} =
	GenericFMDStumpSupport{T, WorldType}(Array{Dict{WorldType,T}, 3}(undef, n_samples(fmd), n_featsnaggrs, n_relations))
# modalDatasetIsConsistent_m(modalDataset, fmd::FeatModalDataset{T, AbstractWorld}, n_featsnaggrs::Integer, n_relations::Integer) where {T, WorldType} =
	# (typeof(modalDataset)<:AbstractArray{T, 7} && size(modalDataset) == (max_channel_size(fmd)[1], max_channel_size(fmd)[1]+1, n_samples(fmd), n_featsnaggrs, n_relations))
initFMDStumpSupportWorldSlice(fmds::GenericFMDStumpSupport{T, WorldType}, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer) where {T, WorldType} =
	fmds.d[i_instance, i_featsnaggr, i_relation] = Dict{WorldType,T}()
FMDStumpSupportSet(fmds::GenericFMDStumpSupport{T, WorldType}, w::AbstractWorld, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T, WorldType} =
	fmds.d[i_instance, i_featsnaggr, i_relation][w] = threshold
slice_dataset(fmds::GenericFMDStumpSupport{T, WorldType}, inds::AbstractVector{<:Integer}; return_view = false) where {T, WorldType} =
	GenericFMDStumpSupport{T, WorldType}(if return_view @view fmds.d[inds,:,:] else fmds.d[inds,:,:] end)



struct OneWorldFMDStumpSupport{T} <: AbstractFMDStumpSupport{T, OneWorld}
	d :: AbstractArray{T, 3}
end

n_samples(fmds::OneWorldFMDStumpSupport{T}) where {T}  = size(fmds, 1)
n_featsnaggrs(fmds::OneWorldFMDStumpSupport{T}) where {T} = size(fmds, 2)
n_relations(fmds::OneWorldFMDStumpSupport{T}) where {T} = size(fmds, 3)
getindex(
	fmds         :: OneWorldFMDStumpSupport{T},
	i_instance   :: Integer,
	w            :: OneWorld,
	i_featsnaggr :: Integer,
	i_relation   :: Integer) where {T} = fmds.d[i_instance, i_featsnaggr, i_relation]
size(fmds::OneWorldFMDStumpSupport{T}, args::Vararg) where {T} = size(fmds.d, args...)
world_type(fmds::OneWorldFMDStumpSupport{T}) where {T} = OneWorld

initFMDStumpSupport(fmd::FeatModalDataset{T, OneWorld}, n_featsnaggrs::Integer, n_relations::Integer) where {T} =
	OneWorldFMDStumpSupport{T}(Array{T, 3}(undef, n_samples(fmd), n_featsnaggrs, n_relations))
# modalDatasetIsConsistent_m(modalDataset, fmd::FeatModalDataset{T, OneWorld}, n_featsnaggrs::Integer, n_relations::Integer) where {T} =
	# (typeof(modalDataset)<:AbstractArray{T, 3} && size(modalDataset) == (n_samples(fmd), n_featsnaggrs, n_relations))
initFMDStumpSupportWorldSlice(fmds::OneWorldFMDStumpSupport, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer) =
	nothing
FMDStumpSupportSet(fmds::OneWorldFMDStumpSupport{T}, w::OneWorld, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
	fmds.d[i_instance, i_featsnaggr, i_relation] = threshold
slice_dataset(fmds::OneWorldFMDStumpSupport{T}, inds::AbstractVector{<:Integer}; return_view = false) where {T} =
	OneWorldFMDStumpSupport{T}(if return_view @view fmds.d[inds,:,:] else fmds.d[inds,:,:] end)



struct IntervalFMDStumpSupport{T} <: AbstractFMDStumpSupport{T, Interval}
	d :: AbstractArray{T, 5}
end

n_samples(fmds::IntervalFMDStumpSupport{T}) where {T}  = size(fmds, 3)
n_featsnaggrs(fmds::IntervalFMDStumpSupport{T}) where {T} = size(fmds, 4)
n_relations(fmds::IntervalFMDStumpSupport{T}) where {T} = size(fmds, 5)
getindex(
	fmds         :: IntervalFMDStumpSupport{T},
	i_instance   :: Integer,
	w            :: Interval,
	i_featsnaggr :: Integer,
	i_relation   :: Integer) where {T} = fmds.d[w.x, w.y, i_instance, i_featsnaggr, i_relation]
size(fmds::IntervalFMDStumpSupport{T}, args::Vararg) where {T} = size(fmds.d, args...)
world_type(fmds::IntervalFMDStumpSupport{T}) where {T} = Interval

initFMDStumpSupport(fmd::FeatModalDataset{T, Interval}, n_featsnaggrs::Integer, n_relations::Integer) where {T} =
	IntervalFMDStumpSupport{T}(Array{T, 5}(undef, size(fmd.fwd, 1), size(fmd.fwd, 2), n_samples(fmd), n_featsnaggrs, n_relations))
# modalDatasetIsConsistent_m(modalDataset, fmd::FeatModalDataset{T, Interval}, n_featsnaggrs::Integer, n_relations::Integer) where {T} =
	# (typeof(modalDataset)<:AbstractArray{T, 5} && size(modalDataset) == (max_channel_size(fmd)[1], max_channel_size(fmd)[1]+1, n_samples(fmd), n_featsnaggrs, n_relations))
initFMDStumpSupportWorldSlice(fmds::IntervalFMDStumpSupport, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer) =
	nothing
FMDStumpSupportSet(fmds::IntervalFMDStumpSupport{T}, w::Interval, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
	fmds.d[w.x, w.y, i_instance, i_featsnaggr, i_relation] = threshold
slice_dataset(fmds::IntervalFMDStumpSupport{T}, inds::AbstractVector{<:Integer}; return_view = false) where {T} =
	IntervalFMDStumpSupport{T}(if return_view @view fmds.d[:,:,inds,:,:] else fmds.d[:,:,inds,:,:] end)


# struct Interval2DFMDStumpSupport{T} <: AbstractFMDStumpSupport{T, Interval2D}
# 	d :: AbstractArray{T, 7}
# end

# n_samples(fmds::Interval2DFMDStumpSupport{T}) where {T}  = size(fmds, 5)
# n_featsnaggrs(fmds::Interval2DFMDStumpSupport{T}) where {T} = size(fmds, 6)
# n_relations(fmds::Interval2DFMDStumpSupport{T}) where {T} = size(fmds, 7)
# getindex(
# 	fmds         :: Interval2DFMDStumpSupport{T},
# 	i_instance   :: Integer,
# 	w            :: Interval2D,
# 	i_featsnaggr :: Integer,
# 	i_relation   :: Integer) where {T} = fmds.d[w.x.x, w.x.y, w.y.x, w.y.y, i_instance, i_featsnaggr, i_relation]
# size(fmds::Interval2DFMDStumpSupport{T}, args::Vararg) where {T} = size(fmds.d, args...)
# world_type(fmds::Interval2DFMDStumpSupport{T}) where {T} = Interval2D

# initFMDStumpSupport(fmd::FeatModalDataset{T, Interval2D}, n_featsnaggrs::Integer, n_relations::Integer) where {T} =
# 	Interval2DFMDStumpSupport{T}(Array{T, 7}(undef, size(fmd.fwd, 1), size(fmd.fwd, 2), size(fmd.fwd, 3), size(fmd.fwd, 4), n_samples(fmd), n_featsnaggrs, n_relations))
# # modalDatasetIsConsistent_m(modalDataset, fmd::FeatModalDataset{T, Interval2D}, n_featsnaggrs::Integer, n_relations::Integer) where {T} =
# 	# (typeof(modalDataset)<:AbstractArray{T, 7} && size(modalDataset) == (max_channel_size(fmd)[1], max_channel_size(fmd)[1]+1, n_samples(fmd), n_featsnaggrs, n_relations))
# initFMDStumpSupportWorldSlice(fmds::Interval2DFMDStumpSupport, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer) =
# 	nothing
# FMDStumpSupportSet(fmds::Interval2DFMDStumpSupport{T}, w::Interval2D, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
# 	fmds.d[w.x.x, w.x.y, w.y.x, w.y.y, i_instance, i_featsnaggr, i_relation] = threshold
# slice_dataset(fmds::Interval2DFMDStumpSupport{T}, inds::AbstractVector{<:Integer}; return_view = false) where {T} =
# 	Interval2DFMDStumpSupport{T}(if return_view @view fmds.d[:,:,:,:,inds,:,:] else fmds.d[:,:,:,:,inds,:,:] end)


# Note: global support is world-agnostic
struct FMDStumpGlobalSupportArray{T} <: AbstractFMDStumpGlobalSupport{T}
	d :: AbstractArray{T, 2}
end

n_samples(fmds::FMDStumpGlobalSupportArray{T}) where {T}  = size(fmds, 1)
n_featsnaggrs(fmds::FMDStumpGlobalSupportArray{T}) where {T} = size(fmds, 2)
getindex(
	fmds         :: FMDStumpGlobalSupportArray{T},
	i_instance   :: Integer,
	i_featsnaggr  :: Integer) where {T} = fmds.d[i_instance, i_featsnaggr]
size(fmds::FMDStumpGlobalSupportArray{T}, args::Vararg) where {T} = size(fmds.d, args...)

initFMDStumpGlobalSupport(fmd::FeatModalDataset{T}, n_featsnaggrs::Integer) where {T} =
	FMDStumpGlobalSupportArray{T}(Array{T, 2}(undef, n_samples(fmd), n_featsnaggrs))
# modalDatasetIsConsistent_g(modalDataset, fmd::FeatModalDataset{T, ModalLogic.anyworld...} n_featsnaggrs::Integer) where {T, N, WorldType<:AbstractWorld} =
# 	(typeof(modalDataset)<:AbstractArray{T, 2} && size(modalDataset) == (n_samples(fmd), n_featsnaggrs))
FMDStumpGlobalSupportSet(fmds::FMDStumpGlobalSupportArray{T}, i_instance::Integer, i_featsnaggr::Integer, threshold::T) where {T} =
	fmds.d[i_instance, i_featsnaggr] = threshold
slice_dataset(fmds::FMDStumpGlobalSupportArray{T}, inds::AbstractVector{<:Integer}; return_view = false) where {T} =
	FMDStumpGlobalSupportArray{T}(if return_view @view fmds.d[inds,:] else fmds.d[inds,:] end)


function computeModalDatasetStumpSupport(
		fmd                 :: FeatModalDataset{T, WorldType},
		grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}};
		computeRelationGlob = false,
	) where {T, N, WorldType<:AbstractWorld}
	
	@logmsg DTOverview "FeatModalDataset -> StumpFeatModalDataset"

	fwd = fmd.fwd
	features = fmd.features
	relations = fmd.relations

	computefmd_g =
		if RelationGlob in relations
			error("RelationGlob in relations: $(relations)")
			relations = filter!(l->l≠RelationGlob, relations)
			true
		elseif computeRelationGlob
			true
		else
			false
	end

	n_instances = n_samples(fmd)
	n_relations = length(relations)
	n_featsnaggrs = sum(length.(grouped_featsnaggrs))

	# println(n_instances)
	# println(n_relations)
	# println(n_featsnaggrs)
	# println(grouped_featsnaggrs)

	# Prepare fmd_m
	fmd_m = initFMDStumpSupport(fmd, n_featsnaggrs, n_relations)

	# Prepare fmd_g
	fmd_g =
		if computefmd_g
			initFMDStumpGlobalSupport(fmd, n_featsnaggrs)
		else
			nothing
	end

	# Compute features

	@inbounds Threads.@threads for i_instance in 1:n_instances
		@logmsg DTDebug "Instance $(i_instance)/$(n_instances)"
		
		if i_instance == 1 || ((i_instance+1) % (floor(Int, ((n_instances)/5))+1)) == 0
			@logmsg DTOverview "Instance $(i_instance)/$(n_instances)"
		end

		for (i_feature,aggregators) in enumerate(grouped_featsnaggrs)
			
			@logmsg DTDebug "Feature $(i_feature)"
			
			cur_fwd_slice = modalDatasetChannelSlice(fwd, i_instance, i_feature)

			@logmsg DTDebug cur_fwd_slice

			# Global relation (independent of the current world)
			if computefmd_g
				@logmsg DTDebug "RelationGlob"

				# TODO optimize: all aggregators are likely reading the same raw values.
				for (i_featsnaggr,aggregator) in aggregators
				# Threads.@threads for (i_featsnaggr,aggregator) in aggregators
					
					# accessible_worlds = accAll_function(fmd, i_instance)
					# TODO reintroduce the improvements for some operators: e.g. later. Actually, these can be simplified by using a set of representatives, as in some enumAccRepr!
					accessible_worlds = ModalLogic.enumReprAll(WorldType, accrepr_function(fmd, i_instance), features[i_feature], aggregator)

					threshold = computeModalThreshold(cur_fwd_slice, accessible_worlds, aggregator)

					@logmsg DTDebug "Aggregator[$(i_featsnaggr)]=$(aggregator)  -->  $(threshold)"
					
					# @logmsg DTDebug "Aggregator" aggregator threshold
					
					FMDStumpGlobalSupportSet(fmd_g, i_instance, i_featsnaggr, threshold)
				end
			end
			# readline()

			# Other relations
			for (i_relation,relation) in enumerate(relations)

				@logmsg DTDebug "Relation $(i_relation)/$(n_relations)"

				for (i_featsnaggr,aggregator) in aggregators
					initFMDStumpSupportWorldSlice(fmd_m, i_instance, i_featsnaggr, i_relation)
				end

				for w in accAll_function(fmd, i_instance)

					@logmsg DTDebug "World" w
					
					# TODO optimize: all aggregators are likely reading the same raw values.
					for (i_featsnaggr,aggregator) in aggregators
											
						# accessible_worlds = acc_function(fmd, i_instance)(w, relation)
						# TODO reintroduce the improvements for some operators: e.g. later. Actually, these can be simplified by using a set of representatives, as in some enumAccRepr!
						accessible_worlds = accrepr_function(fmd, i_instance)(features[i_feature], aggregator, w, relation)
					
						threshold = computeModalThreshold(cur_fwd_slice, accessible_worlds, aggregator)

						# @logmsg DTDebug "Aggregator" aggregator threshold
						
						FMDStumpSupportSet(fmd_m, w, i_instance, i_featsnaggr, i_relation, threshold)
					end
				end
			end
		end
	end
	fmd_m, fmd_g
end
