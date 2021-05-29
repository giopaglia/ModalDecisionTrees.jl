# TODO clean all of this.
inst_readWorld(w::Interval, instance::MatricialInstance{T,2}) where {T} = instance[w.x:w.y-1,:]
# computePropositionalThreshold(feature, w, channel) =  feature(readWorld(w,channel))
computePropositionalThreshold(feature::FeatureTypeFun, w::AbstractWorld, instance::MatricialInstance) = yieldFunction(feature)(inst_readWorld(w,instance))




# FeaturedWorldDataset(
# 		X::OntologicalDataset{T, N, WorldType},
# 		features::AbstractVector{<:FeatureTypeFun}
# 	) where {T, N, WorldType<:AbstractWorld} = FeaturedWorldDataset{T, WorldType}(X, features)

FeaturedWorldDataset(
		X::OntologicalDataset{T, N, WorldType},
		features::AbstractVector{<:FeatureTypeFun}
	) where {T, N, WorldType<:AbstractWorld} = begin

	n_instances = n_samples(X)
	
	# Prepare FeaturedWorldDataset (the actual implementation depends on the OntologicalDataset)
	fwd = initModalDataset(X, features)

	# Compute features
	@inbounds Threads.@threads for i_instance in 1:n_instances
		@logmsg DTDebug "Instance $(i_instance)/$(n_instances)"
		
		if i_instance == 1 || ((i_instance+1) % (floor(Int, ((n_instances)/5))+1)) == 0
			@logmsg DTOverview "Instance $(i_instance)/$(n_instances)"
		end
		
		instance = getInstance(X, i_instance)

		for w in enumAll(WorldType, inst_channel_size(instance)...)
			initModalDatasetWorldSlice(fwd, w)
		end

		@logmsg DTDebug "instance" instance

		for w in enumAll(WorldType, inst_channel_size(instance)...)
			
			@logmsg DTDebug "World" w

			for (i_feature,feature) in enumerate(features)

				threshold = computePropositionalThreshold(feature, w, instance)

				@logmsg DTDebug "Feature $(i_feature)" threshold
			
				modalDatasetSet(fwd, i_instance, w, i_feature, threshold)

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

function checkModalDatasetConsistency(
		fmd      :: FeatModalDataset{T, WorldType},
		X        :: OntologicalDataset{T, N, WorldType},
	) where {T, N, WorldType<:AbstractWorld}
	if !(modalDatasetIsConsistent(fmd, X))
		error("The provided FeaturedWorldDataset structure (of type $(typeof(fmd))) is not consistent with the expected dataset!"
			* "\n\tfmd:"
			* " $(typeof(fmd))"
			* " $(eltype(fmd))"
			* " $(size(fmd))"
			* "\n\tX: $(n_samples(X))"
			* " $(n_attributes(X))"
			* " $(channel_size(X))"
		)
	end
end

# TODO use modalDatasetIsConsistent_m and modalDatasetIsConsistent_g

struct IntervalFeaturedWorldDataset{T} <: AbstractFeaturedWorldDataset{T, Interval}
	d :: AbstractArray{T, 4}
end

# TODO rename these functions to underscore case
n_samples(fwd::IntervalFeaturedWorldDataset{T}) where {T} = size(fwd.d, 3)
n_features(fwd::IntervalFeaturedWorldDataset{T}) where {T} = size(fwd.d, 4)
getindex(
	fwd         :: IntervalFeaturedWorldDataset{T},
	i_instance  :: Integer,
	w           :: Interval,
	i_feature   :: Integer) where {T} = fwd.d[w.x, w.y, i_instance, i_feature]

initModalDataset(X::OntologicalDataset{T, 1, Interval}, features::AbstractVector{<:FeatureTypeFun}) where {T} =
	IntervalFeaturedWorldDataset{T}(Array{T, 4}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), length(features)))
modalDatasetIsConsistent(fmd::FeatModalDataset{T, Interval}, X::OntologicalDataset{T, 1, Interval}) where {T} =
	(typeof(fmd.fwd)<:IntervalFeaturedWorldDataset{T} && size(fmd.fwd.d) == (max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), length(fmd.features)))
initModalDatasetWorldSlice(fwd::IntervalFeaturedWorldDataset{T}, worldType::Interval) where {T} =
	nothing
modalDatasetSet(fwd::IntervalFeaturedWorldDataset{T}, i_instance::Integer, w::Interval, i_feature::Integer, threshold::T) where {T} =
	fwd.d[w.x, w.y, i_instance, i_feature] = threshold
sliceModalDatasetByInstances(fwd::IntervalFeaturedWorldDataset{T}, inds::AbstractVector{<:Integer}; return_view = false) where {T} =
	IntervalFeaturedWorldDataset{T}(if return_view @view fwd.d[:,:,inds,:] else fwd.d[:,:,inds,:] end)
modalDatasetChannelSlice(fwd::IntervalFeaturedWorldDataset{T}, i_instance::Integer, i_feature::Integer) where {T} =
	@view fwd.d[:, :, i_instance, i_feature]
const IntervalFeaturedChannel{T} = AbstractArray{T, 2}
modalDatasetChannelSliceGet(fwc::IntervalFeaturedChannel{T}, w::Interval) where {T} =
	fwc[w.x, w.y]
# TODO move ModalDatasetChannelSliceType{T} definition here


struct IntervalFeatModalDatasetStumpSupport{T} <: AbstractFeatModalDatasetStumpSupport{T, Interval}
	d :: AbstractArray{T, 5}
end

n_samples(stump_support_relations::IntervalFeatModalDatasetStumpSupport{T}) where {T} = size(stump_support_relations.d, 3)
n_featsnaggrs(stump_support_relations::IntervalFeatModalDatasetStumpSupport{T}) where {T} = size(stump_support_relations.d, 4)
n_relations(stump_support_relations::IntervalFeatModalDatasetStumpSupport{T}) where {T} = size(stump_support_relations.d, 5)
initModalDataset_m(fwd::IntervalFeaturedWorldDataset{T}, relations::AbstractVector{<:AbstractRelation}, grouped_featnaggrs::AbstractVector{<:AbstractVector{Tuple{Integer,<:Aggregator}}}) where {T} =
	IntervalFeatModalDatasetStumpSupport{T}(Array{T, 5}(undef, size(fwd.d, 1), size(fwd.d, 2), n_samples(fwd), sum(length.(grouped_featnaggrs)), length(relations)))
modalDatasetIsConsistent_m(stump_support_relations::AbstractFeatModalDatasetStumpSupport{T, Interval}, fwd::IntervalFeaturedWorldDataset{T}, relations::AbstractVector{<:AbstractRelation}, grouped_featnaggrs::AbstractVector{<:AbstractVector{<:Aggregator}}) where {T} =
	(typeof(stump_support_relations)<:IntervalFeatModalDatasetStumpSupport{T} && size(fwd.d) == (size(fwd.d, 1), size(fwd.d, 2), n_samples(fwd), sum(length.(grouped_featnaggrs)), length(relations)))
initModalDatasetWorldSlice_m(stump_support_relations::IntervalFeatModalDatasetStumpSupport{T}, worldType::Interval) where {T} =
	nothing
modalDatasetSet_m(stump_support_relations::IntervalFeatModalDatasetStumpSupport{T}, i_instance::Integer, w::Interval, i_relation::Integer, i_featnaggr::Integer, threshold::T) where {T} =
	stump_support_relations.d[w.x, w.y, i_instance, i_featnaggr, i_relation] = threshold
sliceModalDatasetByInstances_m(stump_support_relations::IntervalFeatModalDatasetStumpSupport{T}, inds::AbstractVector{<:Integer}; return_view = false) where {T} =
	IntervalFeatModalDatasetStumpSupport{T}(if return_view @view stump_support_relations.d[:,:,inds,:,:] else stump_support_relations.d[:,:,inds,:,:] end)
getindex(
		stump_support_relations::IntervalFeatModalDatasetStumpSupport{T},
		i_instance      :: Integer,
		w               :: Interval,
		i_relation      :: Integer,
		i_featnaggr     :: Integer,
	) where {T} = stump_support_relations.d[w.x, w.y, i_instance, i_featnaggr, i_relation]



struct IntervalFeatModalDatasetStumpSupport_global{T} <: AbstractFeatModalDatasetStumpSupport_global{T, Interval}
	d :: AbstractArray{T, 2}
end

n_samples(stump_support_global::IntervalFeatModalDatasetStumpSupport_global{T}) where {T} = size(stump_support_global.d, 1)
n_featsnaggrs(stump_support_global::IntervalFeatModalDatasetStumpSupport_global{T}) where {T} = size(stump_support_global.d, 2)
initModalDataset_g(fwd::IntervalFeaturedWorldDataset{T}, grouped_featnaggrs::AbstractVector{<:AbstractVector{Tuple{Integer,<:Aggregator}}}) where {T, N} =
	IntervalFeatModalDatasetStumpSupport_global{T}(Array{T, 2}(undef, n_samples(fwd), sum(length.(grouped_featnaggrs))))
modalDatasetIsConsistent_g(stump_support_global::AbstractFeatModalDatasetStumpSupport_global{T, Interval}, fwd::IntervalFeaturedWorldDataset{T}, grouped_featnaggrs::AbstractVector{<:AbstractVector{<:Aggregator}}) where {T, N} =
	(typeof(stump_support_global)<:IntervalFeatModalDatasetStumpSupport_global{T} && size(stump_support_global.d) == (n_samples(fwd), sum(length.(grouped_featnaggrs)), length(relations)))
modalDatasetSet_g(stump_support_global::IntervalFeatModalDatasetStumpSupport_global{T}, i_instance::Integer, i_featnaggr::Integer, threshold::T) where {T} =
	stump_support_global.d[i_instance, i_featnaggr] = threshold
sliceModalDatasetByInstances_g(stump_support_global::IntervalFeatModalDatasetStumpSupport_global{T}, inds::AbstractVector{<:Integer}; return_view = false) where {T,N,WorldType<:AbstractWorld} =
	IntervalFeatModalDatasetStumpSupport_global{T}(if return_view @view stump_support_global.d[inds,:] else stump_support_global.d[inds,:] end)
getindex(
	stump_support_global::IntervalFeatModalDatasetStumpSupport_global{T},
	i_instance      :: Integer,
	i_featnaggr     :: Integer,
) where {T} = stump_support_global.d[i_instance, i_featnaggr]



################################################################################
################################################################################
################################################################################
################################################################################


# function prepare_featnaggrs(grouped_featsnops::AbstractVector{<:AbstractVector{<:TestOperatorFun}})
		
# 	# Pairs of feature ids + set of aggregators
# 	grouped_featnaggrs = [
# 		ModalLogic.existential_aggregator.(test_operators) for (i_feature, test_operators) in enumerate(grouped_featsnops)
# 	]


# 	# grouped_featnaggrs = [grouped_featnaggrs[i_feature] for i_feature in 1:length(features)]

# 	# # Flatten dictionary, and enhance aggregators in dictionary with their relative indices
# 	# flattened_featnaggrs = Tuple{<:FeatureTypeFun,<:Aggregator}[]
# 	# i_featnaggr = 1
# 	# for (i_feature, aggregators) in enumerate(grouped_featnaggrs)
# 	# 	for aggregator in aggregators
# 	# 		push!(flattened_featnaggrs, (features[i_feature],aggregator))
# 	# 		i_featnaggr+=1
# 	# 	end
# 	# end

# 	grouped_featnaggrs
# end

