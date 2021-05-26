# TODO clean all of this.
@inline inst_readWorld(w::Interval, instance::MatricialInstance{T,2}) where {T} = instance[w.x:w.y-1,:]
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
	n_features = length(features)

	# Prepare FeaturedWorldDataset (the actual implementation depends on the OntologicalDataset)
	fwd = initModalDataset(X, n_features)

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
# @inline function checkModalDatasetConsistency(modalDataset, X::OntologicalDataset{T, N, WorldType}, features::AbstractVector{<:FeatureTypeFun}) where {T, N, WorldType<:AbstractWorld}
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

struct IntervalFeaturedWorldDataset{T} <: AbstractFeaturedWorldDataset{T, Interval}
	d :: Array{T, 4}
end

# TODO rename these functions to underscore case
initModalDataset(X::OntologicalDataset{T, 1, Interval}, n_features::Integer) where {T} =
	IntervalFeaturedWorldDataset{T}(Array{T, 4}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), n_features))
modalDatasetIsConsistent(fwd::Any, X::OntologicalDataset{T, 1, Interval}, n_features::Integer) where {T} =
	(typeof(fwd)<:IntervalFeaturedWorldDataset{T} && size(fwd.d) == (max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), n_features))
initModalDatasetWorldSlice(fwd::IntervalFeaturedWorldDataset{T}, worldType::Interval) where {T} =
	nothing
modalDatasetSet(fwd::IntervalFeaturedWorldDataset{T}, w::Interval, i_instance::Integer, i_feature::Integer, threshold::T) where {T} =
	fwd.d[w.x, w.y, i_instance, i_feature] = threshold
sliceModalDatasetByInstances(fwd::IntervalFeaturedWorldDataset{T}, inds::AbstractVector{<:Integer}; return_view = false) where {T} =
	if return_view @view fwd.d[:,:,inds,:] else fwd.d[:,:,inds,:] end
getindex(
	fwd         :: IntervalFeaturedWorldDataset{T},
	i_instance  :: Integer,
	w           :: Interval,
	i_feature   :: Integer) where {T} = fwd.d[w.x, w.y, i_instance, i_feature]
modalDatasetChannelSlice(fwd::IntervalFeaturedWorldDataset{T}, i_instance::Integer, i_feature::Integer) where {T} =
	@views fwd.d[:, :, i_instance, i_feature]
const IntervalFeaturedChannel{T} = AbstractArray{T, 2}
modalDatasetChannelSliceGet(fwc::IntervalFeaturedChannel{T}, w::Interval) where {T} =
	fwc[w.x, w.y]

n_samples(fwd::IntervalFeaturedWorldDataset{T}) where {T}  = size(fwd.d, 3)
n_features(fwd::IntervalFeaturedWorldDataset{T}) where {T} = size(fwd.d, 4)


const ModalDatasetSliceType{T} = Union{
	# modalDatasetSliceType(OntologicalDataset{T where T, 0, ModalLogic.OneWorld})
	IntervalFeaturedChannel{T}
	# modalDatasetSliceType(OntologicalDataset{T where T, 2, ModalLogic.Interval2D})
}


################################################################################
################################################################################
################################################################################
################################################################################


function prepare_featnaggrs(grouped_featsops::AbstractVector{<:AbstractVector{<:TestOperatorFun}})
		
	# Pairs of feature ids + set of aggregators
	grouped_featnaggrs = [
		ModalLogic.existential_aggregator.(test_operators) for (i_feature, test_operators) in enumerate(grouped_featsops)
	]


	# grouped_featnaggrs = [grouped_featnaggrs[i_feature] for i_feature in 1:length(features)]

	# # Flatten dictionary, and enhance aggregators in dictionary with their relative indices
	# flattened_featnaggrs = Tuple{<:FeatureTypeFun,<:Aggregator}[]
	# i_featnaggr = 1
	# for (i_feature, aggregators) in enumerate(grouped_featnaggrs)
	# 	for aggregator in aggregators
	# 		push!(flattened_featnaggrs, (features[i_feature],aggregator))
	# 		i_featnaggr+=1
	# 	end
	# end

	grouped_featnaggrs
end
