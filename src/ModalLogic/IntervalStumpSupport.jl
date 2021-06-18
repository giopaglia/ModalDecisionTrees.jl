
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
	fwd = initFeaturedWorldDataset(X, n_features)

	# Compute features
	@inbounds Threads.@threads for i_instance in 1:n_instances
		@logmsg DTDebug "Instance $(i_instance)/$(n_instances)"
		
		if i_instance == 1 || ((i_instance+1) % (floor(Int, ((n_instances)/5))+1)) == 0
			@logmsg DTOverview "Instance $(i_instance)/$(n_instances)"
		end
		
		instance = getInstance(X, i_instance)

		for w in enumAll(WorldType, inst_channel_size(instance)...)
			initFeaturedWorldDatasetWorldSlice(fwd, w)
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

struct IntervalFeaturedWorldDataset{T} <: AbstractFeaturedWorldDataset{T, Interval}
	d :: Array{T, 4}
end

# TODO rename these functions to underscore case
initFeaturedWorldDataset(X::OntologicalDataset{T, 1, Interval}, n_features::Integer) where {T} =
	IntervalFeaturedWorldDataset{T}(Array{T, 4}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), n_features))
modalDatasetIsConsistent(fwd::Any, X::OntologicalDataset{T, 1, Interval}, n_features::Integer) where {T} =
	(typeof(fwd)<:IntervalFeaturedWorldDataset{T} && size(fwd) == (max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), n_features))
initFeaturedWorldDatasetWorldSlice(fwd::IntervalFeaturedWorldDataset{T}, worldType::Interval) where {T} =
	nothing
modalDatasetSet(fwd::IntervalFeaturedWorldDataset{T}, w::Interval, i_instance::Integer, i_feature::Integer, threshold::T) where {T} =
	fwd.d[w.x, w.y, i_instance, i_feature] = threshold
slice_dataset(fwd::IntervalFeaturedWorldDataset{T}, inds::AbstractVector{<:Integer}; return_view = false) where {T} =
	IntervalFeaturedWorldDataset{T}(if return_view @view fwd.d[:,:,inds,:] else fwd.d[:,:,inds,:] end)
modalDatasetChannelSlice(fwd::IntervalFeaturedWorldDataset{T}, i_instance::Integer, i_feature::Integer) where {T} =
	@views fwd.d[:, :, i_instance, i_feature]
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


const FeaturedWorldDatasetSlice{T} = Union{
	# FeaturedWorldDatasetSlice(OntologicalDataset{T where T, 0, ModalLogic.OneWorld})
	IntervalFeaturedChannel{T}
	# FeaturedWorldDatasetSlice(OntologicalDataset{T where T, 2, ModalLogic.Interval2D})
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


struct IntervalFMDStumpSupport{T} <: AbstractFMDStumpSupport{T, Interval}
	d :: Array{T, 5}
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

initFMDStumpSupport(fmd::FeatModalDataset{T, ModalLogic.Interval}, n_featsnaggrs::Integer, n_relations::Integer) where {T} =
	IntervalFMDStumpSupport{T}(Array{T, 5}(undef, size(fmd.fwd, 1), size(fmd.fwd, 2), n_samples(fmd), n_featsnaggrs, n_relations))
# modalDatasetIsConsistent_m(modalDataset, fmd::FeatModalDataset{T, ModalLogic.Interval}, n_featsnaggrs::Integer, n_relations::Integer) where {T} =
	# (typeof(modalDataset)<:AbstractArray{T, 5} && size(modalDataset) == (max_channel_size(fmd)[1], max_channel_size(fmd)[1]+1, n_samples(fmd), n_featsnaggrs, n_relations))
initFMDStumpSupportWorldSlice(fmds::IntervalFMDStumpSupport{T}, worldType::ModalLogic.Interval) where {T} =
	nothing
FMDStumpSupportSet(fmds::IntervalFMDStumpSupport{T}, w::ModalLogic.Interval, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
	fmds.d[w.x, w.y, i_instance, i_featsnaggr, i_relation] = threshold
slice_dataset(fmds::IntervalFMDStumpSupport{T}, inds::AbstractVector{<:Integer}; return_view = false) where {T} =
	IntervalFMDStumpSupport{T}(if return_view @view fmds.d[:,:,inds,:,:] else fmds.d[:,:,inds,:,:] end)

struct IntervalFMDStumpGlobalSupport{T} <: AbstractFMDStumpGlobalSupport{T, Interval}
	d :: Array{T, 2}
end

n_samples(fmds::IntervalFMDStumpGlobalSupport{T}) where {T}  = size(fmds, 1)
n_featsnaggrs(fmds::IntervalFMDStumpGlobalSupport{T}) where {T} = size(fmds, 2)
getindex(
	fmds         :: IntervalFMDStumpGlobalSupport{T},
	i_instance   :: Integer,
	i_featsnaggr  :: Integer) where {T} = fmds.d[i_instance, i_featsnaggr]
size(fmds::IntervalFMDStumpGlobalSupport{T}, args::Vararg) where {T} = size(fmds.d, args...)
world_type(fmds::IntervalFMDStumpGlobalSupport{T}) where {T} = Interval

initFMDStumpGlobalSupport(fmd::FeatModalDataset{T, ModalLogic.Interval}, n_featsnaggrs::Integer) where {T} =
	IntervalFMDStumpGlobalSupport{T}(Array{T, 2}(undef, n_samples(fmd), n_featsnaggrs))
# modalDatasetIsConsistent_g(modalDataset, fmd::FeatModalDataset{T, ModalLogic.Interval} n_featsnaggrs::Integer) where {T, N, WorldType<:AbstractWorld} =
# 	(typeof(modalDataset)<:AbstractArray{T, 2} && size(modalDataset) == (n_samples(fmd), n_featsnaggrs))
FMDStumpGlobalSupportSet(fmds::IntervalFMDStumpGlobalSupport{T}, i_instance::Integer, i_featsnaggr::Integer, threshold::T) where {T} =
	fmds.d[i_instance, i_featsnaggr] = threshold
slice_dataset(fmds::IntervalFMDStumpGlobalSupport{T}, inds::AbstractVector{<:Integer}; return_view = false) where {T} =
	IntervalFMDStumpGlobalSupport{T}(if return_view @view fmds.d[inds,:] else fmds.d[inds,:] end)



function computeModalDatasetStumpSupport(
		fmd                 :: FeatModalDataset{T, WorldType},
		grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}};
		computeRelationAll = false,
	) where {T, N, WorldType<:AbstractWorld}
	
	fwd = fmd.fwd
	features = fmd.features
	relations = fmd.relations

	computefmd_g =
		if RelationAll in relations
			error("RelationAll in relations: $(relations)")
			relations = filter!(l->l≠RelationAll, relations)
			true
		elseif computeRelationAll
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

		for w in ModalLogic.enumAll(WorldType, acc_function(fmd, i_instance))
			initFMDStumpSupportWorldSlice(fmd_m, w)
		end

		for (i_feature,aggregators) in enumerate(grouped_featsnaggrs)
			
			@logmsg DTDebug "Feature $(i_feature)"
			
			cur_fwd_slice = modalDatasetChannelSlice(fwd, i_instance, i_feature)

			@logmsg DTDebug cur_fwd_slice

			# Global relation (independent of the current world)
			if computefmd_g
				@logmsg DTDebug "RelationAll"

				# TODO optimize: all aggregators are likely reading the same raw values.
				for (i_featsnaggr,aggregator) in aggregators
					
					# accessible_worlds = ModalLogic.enumAll(WorldType, acc_function(fmd, i_instance))
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

				for w in ModalLogic.enumAll(WorldType, acc_function(fmd, i_instance))

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
