
const Aggregator = Function

# OntologicalDataset with ModalLogic.Interval
modalDatasetType(::Type{<:OntologicalDataset{T, N, ModalLogic.Interval}}) where {T, N} = AbstractArray{T, 4}
@inline initModalDataset(X::OntologicalDataset{T, N, ModalLogic.Interval}, n_features::Integer) where {T, N} =
	Array{T, 4}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), n_features) # fill(0, (max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), n_features))
@inline modalDatasetIsConsistent(modalDataset, X::OntologicalDataset{T, N, ModalLogic.Interval}, n_features::Integer) where {T, N} =
	(typeof(modalDataset)<:AbstractArray{T, 4} && size(modalDataset) == (max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), n_features))
@inline initModalDatasetWorldSlice(::Type{<:OntologicalDataset{T, N, ModalLogic.Interval}}, modalDataset::AbstractArray{T, 4}, worldType::ModalLogic.Interval) where {T,N} =
	nothing
@inline modalDatasetSet(modalDataset::AbstractArray{T, 4}, w::ModalLogic.Interval, i_instance::Integer, i_feature::Integer, threshold::T) where {T} =
	modalDataset[w.x, w.y, i_instance, i_feature] = threshold
modalDatasetSliceType(::Type{OntologicalDataset{T, N, ModalLogic.Interval}}) where {T, N} = AbstractArray{T, 2}
@inline modalDatasetChannelSlice(modalDataset::AbstractArray{T, 4}, i_instance::Integer, i_feature::Integer) where {T} =
	@views modalDataset[:, :, i_instance, i_feature]
@inline modalDatasetChannelSliceGet(modalDataset::AbstractArray{T, 2}, w::ModalLogic.Interval) where {T} =
	@views modalDataset[w.x, w.y]

modalDatasetType_m(::Type{<:OntologicalDataset{T, N, ModalLogic.Interval}}) where {T, N} = AbstractArray{T, 5}
@inline initModalDataset_m(X::OntologicalDataset{T, N, ModalLogic.Interval}, n_featnaggrs::Integer, n_relations::Integer) where {T, N} =
	Array{T, 5}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), n_featnaggrs, n_relations)
@inline modalDatasetIsConsistent_m(modalDataset, X::OntologicalDataset{T, N, ModalLogic.Interval}, n_featnaggrs::Integer, n_relations::Integer) where {T, N} =
	(typeof(modalDataset)<:AbstractArray{T, 5} && size(modalDataset) == (max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), n_featnaggrs, n_relations))
@inline initModalDatasetWorldSlice_m(::Type{<:OntologicalDataset{T, N, ModalLogic.Interval}}, modalDataset::AbstractArray{T, 5}, worldType::ModalLogic.Interval) where {T,N} =
	nothing
@inline modalDatasetSet_m(modalDataset::AbstractArray{T, 5}, w::ModalLogic.Interval, i_instance::Integer, i_featnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
	modalDataset[w.x, w.y, i_instance, i_featnaggr, i_relation] = threshold

const ModalDatasetType{T} = Union{
	# modalDatasetType(OntologicalDataset{T where T, 0, ModalLogic.OneWorld})
	AbstractArray{T, 4} # modalDatasetType(OntologicalDataset{T where T, 1, ModalLogic.Interval})
	# modalDatasetType(OntologicalDataset{T where T, 2, ModalLogic.Interval2D})
}

const ModalDatasetSliceType{T} = Union{
	# modalDatasetSliceType(OntologicalDataset{T where T, 0, ModalLogic.OneWorld})
	AbstractArray{T, 2} # modalDatasetSliceType(OntologicalDataset{T, 1, ModalLogic.Interval})
	# modalDatasetSliceType(OntologicalDataset{T where T, 2, ModalLogic.Interval2D})
}



@inline inst_readWorld(w::ModalLogic.Interval, instance::MatricialInstance{T,2}) where {T} = instance[w.x:w.y-1,:]

# computePropositionalThreshold(feature, w, channel) =  feature(ModalLogic.readWorld(w,channel))
computePropositionalThreshold(feature::FeatureTypeFun, w::AbstractWorld, instance::MatricialInstance) = ModalLogic.yieldFunction(feature)(inst_readWorld(w,instance))
computeModalThreshold(modalDatasetP_slice::ModalDatasetSliceType{T}, relation::AbstractRelation, w::AbstractWorld, aggregator::Agg, instance::MatricialInstance) where {T, Agg<:Aggregator} = begin
	
	worlds = ModalLogic.enumAccessibles(w, relation, ModalLogic.inst_channel_size(instance)...) # TODO
	# worlds = ModalLogic.enumAccessibles([w], relation, ModalLogic.inst_channel_size(instance)...)

	# TODO try reduce(aggregator, worlds; init=ModalLogic.bottom(aggregator, T))
	# TODO remove this aggregator_to_binary...
	
	if length(worlds |> collect) == 0
		ModalLogic.aggregator_bottom(aggregator, T)
	else
		aggregator((w)->modalDatasetChannelSliceGet(modalDatasetP_slice, w), worlds)
	end

	# opt = aggregator_to_binary(aggregator)
	# threshold = ModalLogic.bottom(aggregator, T)
	# for w in worlds
	# 	e = modalDatasetChannelSliceGet(modalDatasetP_slice, w)
	# 	threshold = opt(threshold,e)
	# end
	# threshold
end

function prepare_feats_n_aggrs(features_n_operators::AbstractVector{Tuple{<:FeatureTypeFun,<:TestOperatorFun}})
	# Different features to compute
	features = unique(first.(features_n_operators))

	# Pairs of feature ids + set of aggregators
	grouped_feats_n_aggrs_pre = DefaultOrderedDict{Integer, Vector{<:Aggregator}}(Vector{Aggregator})

	for (i_feature, feature) in enumerate(features)
		for (f,o) in features_n_operators
			if f == feature
				push!(grouped_feats_n_aggrs_pre[i_feature], ModalLogic.existential_aggregator(o))
			end
		end
	end

	# Flatten dictionary, and enhance aggregators in dictionary with their relative indices
	grouped_feats_n_aggrs = DefaultOrderedDict{Integer, Vector{Tuple{Integer,<:Aggregator}}}(Vector{Tuple{Integer,Aggregator}})
	flattened_feats_n_aggrs = []
	i_featnaggr = 1
	for (i_feature, aggregators) in grouped_feats_n_aggrs_pre
		grouped_feats_n_aggrs[i_feature] = []
		for aggregator in aggregators
			push!(flattened_feats_n_aggrs, (features[i_feature],aggregator))
			push!(grouped_feats_n_aggrs[i_feature], (i_featnaggr,aggregator))
			i_featnaggr+=1
		end
	end

	(features, grouped_feats_n_aggrs, flattened_feats_n_aggrs)
end


function computeModalDataset(
		X::OntologicalDataset{T, N, WorldType},
		features::AbstractVector{<:FeatureTypeFun}
	) where {T, N, WorldType<:AbstractWorld}

	n_instances = n_samples(X)
	n_features = length(features)

	# Prepare modalDataset
	modalDataset = initModalDataset(X, n_features)

	# Compute features
	for i_instance in 1:n_instances
		@logmsg DTDebug "Instance $(i_instance)/$(n_instances)"
		
		if i_instance == 1 || ((i_instance+1) % (floor(Int, ((n_instances)/5))+1)) == 0
			@logmsg DTOverview "Instance $(i_instance)/$(n_instances)"
		end
		
		instance = ModalLogic.getInstance(X, i_instance)

		for w in ModalLogic.enumAll(WorldType, ModalLogic.inst_channel_size(instance)...)
			initModalDatasetWorldSlice(typeof(X), modalDataset, w)
		end

		@logmsg DTDebug "instance" instance

		for w in ModalLogic.enumAll(WorldType, ModalLogic.inst_channel_size(instance)...)
			
			@logmsg DTDebug "World" w

			for (i_feature,feature) in enumerate(features)

				threshold = computePropositionalThreshold(feature, w, instance)

				@logmsg DTDebug "Feature $(i_feature)" threshold
			
				modalDatasetSet(modalDataset, w, i_instance, i_feature, threshold)

			end
		end
	end
	modalDataset
end

function computeModalDataset_m(
		X::OntologicalDataset{T, N, WorldType},
		relations::AbstractVector{<:AbstractRelation},
		grouped_grouped_feats_n_aggrs::AbstractDict{Integer, Vector{Tuple{<:Integer,<:Aggregator}}},
		modalDatasetP::modalDatasetType(typeof(X)), # TODO make either this or X an optional argument
		features::AbstractVector{<:FeatureTypeFun},
	) where {T, N, WorldType<:AbstractWorld}

	n_instances = n_samples(X)
	n_relations = length(relations)
	n_featnaggrs = sum(length(aggregators) for aggregators in grouped_grouped_feats_n_aggrs)

	firstWorld = WorldType(ModalLogic.firstWorld)

	# Prepare modalDataset
	modalDataset = initModalDataset_m(X, n_featnaggrs, n_relations)

	# Compute features

	for i_instance in 1:n_instances
		@logmsg DTDebug "Instance $(i_instance)/$(n_instances)"
		
		if i_instance == 1 || ((i_instance+1) % (floor(Int, ((n_instances)/5))+1)) == 0
			@logmsg DTOverview "Instance $(i_instance)/$(n_instances)"
		end
		
		instance = ModalLogic.getInstance(X, i_instance)

		for w in ModalLogic.enumAll(WorldType, ModalLogic.inst_channel_size(instance)...)
			initModalDatasetWorldSlice_m(typeof(X), modalDataset, w)
		end

		for (i_feature,aggregators) in grouped_grouped_feats_n_aggrs
			
			@logmsg DTDebug "Feature $(i_feature)"
			
			cur_modalDatasetP = modalDatasetChannelSlice(modalDatasetP, i_instance, i_feature)

			@logmsg DTDebug "instance" instance

			for (i_relation,relation) in enumerate(relations)

				@logmsg DTDebug "Relation $(i_relation)/$(n_relations)"
				
				# No use in computing RelationAll for all worlds.
				#  For convenience, these propositions are only computed for a single, distinct world.
				#  TODO this causes waste of space. Fix this by making that slice smaller?
				worlds = if relation != RelationAll
						ModalLogic.enumAll(WorldType, ModalLogic.inst_channel_size(instance)...)
					else
						[firstWorld]
				end

				for w in worlds
					
					# TODO optimize: all aggregators are likely reading the same raw values.
					for (i_aggregator,aggregator) in aggregators
						
						threshold = computeModalThreshold(cur_modalDatasetP, relation, w, aggregator, instance)

						@logmsg DTDebug "World" w threshold
						
						modalDatasetSet_m(modalDataset, w, i_instance, i_aggregator, i_relation, threshold)

					end
				end

			end
		end
	end
	modalDataset
end
