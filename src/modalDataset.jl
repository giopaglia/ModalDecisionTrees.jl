using DataStructures
using BenchmarkTools


export computeModalDataset, computeModalDataset_m,
				stumpModalDataset
				# , modalDataset

@inline function checkModalDatasetConsistency(modalDataset, X::OntologicalDataset{T, N, WorldType}, features::AbstractVector{<:FeatureTypeFun}) where {T, N, WorldType<:AbstractWorld}
	if !(modalDatasetIsConsistent(modalDataset, X, length(features)))
		error("The provided modalDataset structure is not consistent with the expected dataset, test operators and/or relations!"
			* "\n\tmodalDataset:"
			* " $(typeof(modalDataset))"
			* " $(eltype(modalDataset))"
			* " $(size(modalDataset))"
			* "\n\tX: $(n_samples(X))"
			* " $(n_attributes(X))"
			* " $(channel_size(X))"
			* "\n\tfeatures: $(size(features))"
		)
	end
end


# OntologicalDataset with ModalLogic.Interval
modalDatasetType(::Type{<:OntologicalDataset{T, 1, ModalLogic.Interval}}) where {T} = AbstractArray{T, 4}
@inline initModalDataset(X::OntologicalDataset{T, 1, ModalLogic.Interval}, n_features::Integer) where {T} =
	Array{T, 4}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), n_features) # fill(0, (max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), n_features))
@inline modalDatasetIsConsistent(modalDataset, X::OntologicalDataset{T, 1, ModalLogic.Interval}, n_features::Integer) where {T} =
	(typeof(modalDataset)<:AbstractArray{T, 4} && size(modalDataset) == (max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), n_features))
@inline initModalDatasetWorldSlice(::Type{<:OntologicalDataset{T, 1, ModalLogic.Interval}}, modalDataset::AbstractArray{T, 4}, worldType::ModalLogic.Interval) where {T} =
	nothing
@inline modalDatasetSet(modalDataset::AbstractArray{T, 4}, w::ModalLogic.Interval, i_instance::Integer, i_feature::Integer, threshold::T) where {T} =
	modalDataset[w.x, w.y, i_instance, i_feature] = threshold
modalDatasetSliceType(::Type{OntologicalDataset{T, 1, ModalLogic.Interval}}) where {T} = AbstractArray{T, 2}
@inline modalDatasetChannelSlice(modalDataset::AbstractArray{T, 4}, i_instance::Integer, i_feature::Integer) where {T} =
	@views modalDataset[:, :, i_instance, i_feature]
@inline modalDatasetChannelSliceGet(modalDataset::AbstractArray{T, 2}, w::ModalLogic.Interval) where {T} =
	@views modalDataset[w.x, w.y]
@inline sliceModalDatasetByInstances(::Type{<:OntologicalDataset{T, 1, ModalLogic.Interval}}, modalDataset::AbstractArray{T, 4}, inds::AbstractVector{<:Integer}; return_view = false) where {T,WorldType<:AbstractWorld} =
	if return_view @view modalDataset[:,:,inds,:] else modalDataset[:,:,inds,:] end
@inline function modalDatasetGet(
	modalDataset    :: AbstractArray{T, 4},
	w               :: ModalLogic.Interval,
	i_instance      :: Integer,
	i_feature       :: Integer) where {T}
	modalDataset[w.x, w.y, i_instance, i_feature]
end

const ModalDatasetPType{T} = Union{
	# modalDatasetType(OntologicalDataset{T where T, 0, ModalLogic.OneWorld})
	AbstractArray{T, 4} # modalDatasetType(OntologicalDataset{T where T, 1, ModalLogic.Interval})
	# modalDatasetType(OntologicalDataset{T where T, 2, ModalLogic.Interval2D})
}

const ModalDatasetSliceType{T} = Union{
	# modalDatasetSliceType(OntologicalDataset{T where T, 0, ModalLogic.OneWorld})
	AbstractArray{T, 2} # modalDatasetSliceType(OntologicalDataset{T, 1, ModalLogic.Interval})
	# modalDatasetSliceType(OntologicalDataset{T where T, 2, ModalLogic.Interval2D})
}

modalDatasetType_m(::Type{<:OntologicalDataset{T, 1, ModalLogic.Interval}}) where {T} = AbstractArray{T, 5}
@inline initModalDataset_m(X::OntologicalDataset{T, 1, ModalLogic.Interval}, n_featnaggrs::Integer, n_relations::Integer) where {T} =
	Array{T, 5}(undef, max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), n_featnaggrs, n_relations)
@inline modalDatasetIsConsistent_m(modalDataset, X::OntologicalDataset{T, 1, ModalLogic.Interval}, n_featnaggrs::Integer, n_relations::Integer) where {T} =
	(typeof(modalDataset)<:AbstractArray{T, 5} && size(modalDataset) == (max_channel_size(X)[1], max_channel_size(X)[1]+1, n_samples(X), n_featnaggrs, n_relations))
@inline initModalDatasetWorldSlice_m(::Type{<:OntologicalDataset{T, 1, ModalLogic.Interval}}, modalDataset::AbstractArray{T, 5}, worldType::ModalLogic.Interval) where {T} =
	nothing
@inline modalDatasetSet_m(modalDataset::AbstractArray{T, 5}, w::ModalLogic.Interval, i_instance::Integer, i_featnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
	modalDataset[w.x, w.y, i_instance, i_featnaggr, i_relation] = threshold
@inline sliceModalDatasetByInstances_m(::Type{<:OntologicalDataset{T, 1, ModalLogic.Interval}}, modalDataset::AbstractArray{T, 5}, inds::AbstractVector{<:Integer}; return_view = false) where {T,WorldType<:AbstractWorld} =
	if return_view @view modalDataset[:,:,inds,:,:] else modalDataset[:,:,inds,:,:] end
@inline function modalDatasetGet_m(
	modalDataset    :: AbstractArray{T, 5},
	w               :: ModalLogic.Interval,
	i_instance      :: Integer,
	i_featnaggr     :: Integer,
	i_relation      :: Integer) where {T}
	modalDataset[w.x, w.y, i_instance, i_featnaggr, i_relation]
end

const ModalDatasetMType{T} = Union{
	AbstractArray{T, 5}
}

modalDatasetType_g(::Type{<:OntologicalDataset{T, N, WorldType}}) where {T, N, WorldType<:AbstractWorld} = AbstractArray{T, 2}
@inline initModalDataset_g(X::OntologicalDataset{T, N, WorldType}, n_featnaggrs::Integer) where {T, N, WorldType<:AbstractWorld} =
	Array{T, 2}(undef, n_samples(X), n_featnaggrs)
@inline modalDatasetIsConsistent_g(modalDataset, X::OntologicalDataset{T, N, WorldType}, n_featnaggrs::Integer) where {T, N, WorldType<:AbstractWorld} =
	(typeof(modalDataset)<:AbstractArray{T, 2} && size(modalDataset) == (n_samples(X), n_featnaggrs))
@inline modalDatasetSet_g(modalDataset::AbstractArray{T, 2}, i_instance::Integer, i_featnaggr::Integer, threshold::T) where {T} =
	modalDataset[i_instance, i_featnaggr] = threshold
@inline sliceModalDatasetByInstances_g(::Type{<:OntologicalDataset{T, N, WorldType}}, modalDataset::AbstractArray{T, 2}, inds::AbstractVector{<:Integer}; return_view = false) where {T,N,WorldType<:AbstractWorld} =
	if return_view @view modalDataset[inds,:] else modalDataset[inds,:] end
@inline function modalDatasetGet_g(
	modalDataset    :: AbstractArray{T, 2},
	i_instance      :: Integer,
	i_featnaggr     :: Integer) where {T}
	modalDataset[i_instance, i_featnaggr]
end

const ModalDatasetGType{T} = Union{
	AbstractArray{T, 2}
}



@inline inst_readWorld(w::ModalLogic.Interval, instance::MatricialInstance{T,2}) where {T} = instance[w.x:w.y-1,:]

# computePropositionalThreshold(feature, w, channel) =  feature(ModalLogic.readWorld(w,channel))
computePropositionalThreshold(feature::FeatureTypeFun, w::AbstractWorld, instance::MatricialInstance) = ModalLogic.yieldFunction(feature)(inst_readWorld(w,instance))
computeModalThreshold(modalDatasetP_slice::ModalDatasetSliceType{T}, relation::AbstractRelation, w::AbstractWorld, aggregator::Agg, instance::MatricialInstance, feature::FeatureTypeFun) where {T, Agg<:Aggregator} = begin
	
	worlds = ModalLogic.enumAccessibles(w, relation, ModalLogic.inst_channel_size(instance)...) # TODO

	# TODO reintroduce the improvements for some operators: e.g. later. Actually, these can be simplified by using a set of representatives, as in some enumAccRepr!
	# worlds = ModalLogic.enumAccReprAggr(feature, aggregator, w, relation, ModalLogic.inst_channel_size(instance)...)

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


function computeModalDataset(
		X::OntologicalDataset{T, N, WorldType},
		features::AbstractVector{<:FeatureTypeFun}
	) where {T, N, WorldType<:AbstractWorld}

	n_instances = n_samples(X)
	n_features = length(features)

	# Prepare modalDataset
	modalDataset = initModalDataset(X, n_features)

	# Compute features
	@inbounds Threads.@threads for i_instance in 1:n_instances
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
		grouped_featnaggrs::AbstractVector{AbstractVector{<:Aggregator}},
		modalDatasetP::ModalDatasetPType{T}, # TODO write stricter type
		# modalDatasetP::modalDatasetType(OntologicalDataset{T, N, WorldType}),
		# modalDatasetP::modalDatasetType(typeof(X)), # TODO make either this or X an optional argument
		features::AbstractVector{<:FeatureTypeFun};
		computeRelationAll = false,
	) where {T, N, WorldType<:AbstractWorld}
	
	computeModalDatasetG =
		if RelationAll in relations
			relations = filter!(l->l≠RelationAll, relations)
			true
		elseif computeRelationAll
			true
		else
			false
	end

	n_instances = n_samples(X)
	n_relations = length(relations)
	n_featnaggrs = sum(length.(grouped_featnaggrs))

	# Prepare modalDatasetM
	modalDatasetM = initModalDataset_m(X, n_featnaggrs, n_relations)

	# Prepare modalDatasetG
	modalDatasetG =
		if computeModalDatasetG
			initModalDataset_g(X, n_featnaggrs)
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
		
		instance = ModalLogic.getInstance(X, i_instance)

		for w in ModalLogic.enumAll(WorldType, ModalLogic.inst_channel_size(instance)...)
			initModalDatasetWorldSlice_m(typeof(X), modalDatasetM, w)
		end

		for (i_feature,aggregators) in enumerate(grouped_featnaggrs)
			
			@logmsg DTDebug "Feature $(i_feature)"
			
			cur_modalDatasetP = modalDatasetChannelSlice(modalDatasetP, i_instance, i_feature)

			@logmsg DTDebug "instance" instance

			# Global relation (independent of the current world)
			if computeModalDatasetG
				@logmsg DTDebug "RelationAll"

				# TODO optimize: all aggregators are likely reading the same raw values.
				for (i_aggregator,aggregator) in aggregators
					
					threshold = computeModalThreshold(cur_modalDatasetP, RelationAll, firstWorld, aggregator, instance, features[i_feature])
					
					# @logmsg DTDebug "Aggregator" aggregator threshold
					
					modalDatasetSet_g(modalDatasetG, i_instance, i_aggregator, threshold)
				end
			end

			# Other relations
			for (i_relation,relation) in enumerate(relations)

				@logmsg DTDebug "Relation $(i_relation)/$(n_relations)"

				for w in ModalLogic.enumAll(WorldType, ModalLogic.inst_channel_size(instance)...)
					
					@logmsg DTDebug "World" w
					
					# TODO optimize: all aggregators are likely reading the same raw values.
					for (i_aggregator,aggregator) in aggregators
						
						threshold = computeModalThreshold(cur_modalDatasetP, relation, w, aggregator, instance, features[i_feature])

						# @logmsg DTDebug "Aggregator" aggregator threshold
						
						modalDatasetSet_m(modalDatasetM, w, i_instance, i_aggregator, i_relation, threshold)

					end
				end

			end
		end
	end
	modalDatasetM, modalDatasetG
end



################################################################################
################################################################################
################################################################################

struct stumpModalDataset{T}
	# core data
	modalDatasetP            :: ModalDatasetPType{T}
	modalDatasetM            :: ModalDatasetMType{T}
	modalDatasetG            :: Union{ModalDatasetGType{T},Nothing}
	# info
	n_instances              :: Integer
	relations                :: AbstractVector{<:AbstractRelation}
	features                 :: AbstractVector{<:FeatureTypeFun}
	grouped_featnaggrs       :: AbstractDict{Integer, Vector{Tuple{Integer,<:Aggregator}}}
	flattened_featnaggrs     :: AbstractVector{Tuple{<:FeatureTypeFun,<:Aggregator}}

	function stumpModalDataset(X::OntologicalDataset{T, N, WorldType}, features, grouped_featnaggrs, flattened_featnaggrs; computeRelationAll = false, timing_mode = :none) where {T, N, WorldType<:AbstractWorld}
		stumpModalDataset{T}(X, features, grouped_featnaggrs, flattened_featnaggrs, computeRelationAll = computeRelationAll, timing_mode = timing_mode)
	end

	function stumpModalDataset{T}(X::OntologicalDataset{T, N, WorldType}, features, grouped_featnaggrs, flattened_featnaggrs; computeRelationAll = false, timing_mode = :none) where {T, N, WorldType<:AbstractWorld}
		
		n_instances = n_samples(X)

		# Compute modal dataset propositions
		modalDatasetP = 
			if timing_mode == :none
				computeModalDataset(X, features);
			elseif timing_mode == :time
				@time computeModalDataset(X, features);
			elseif timing_mode == :btime
				@btime computeModalDataset($X, $features);
		end

		relations = X.ontology.relationSet
		# relations = relations[1:3]

		# Compute modal dataset propositions and 1-modal decisions
		modalDatasetM, modalDatasetG = 
			if timing_mode == :none
				computeModalDataset_m(X, relations, grouped_featnaggrs, modalDatasetP, features, computeRelationAll = computeRelationAll);
			elseif timing_mode == :time
				@time computeModalDataset_m(X, relations, grouped_featnaggrs, modalDatasetP, features, computeRelationAll = computeRelationAll);
			elseif timing_mode == :btime
				@btime computeModalDataset_m($X, $relations, $grouped_featnaggrs, $modalDatasetP, $features, computeRelationAll = $computeRelationAll);
		end

		# println(Base.size(X))
		# println(Base.size(modalDatasetP))
		# if !isnothing(modalDatasetG)
		# 	println(Base.size(modalDatasetM))
		# end
		println("Dataset size:\t\t\t$(Base.summarysize(X) / 1024 / 1024 |> x->round(x, digits=2)) MBs")
		println("modalDataset total size:\t$((Base.summarysize(modalDatasetP) + Base.summarysize(modalDatasetM) + Base.summarysize(modalDatasetG)) / 1024 / 1024 |> x->round(x, digits=2)) MBs")
		println("├ modalDatasetP size:\t\t$(Base.summarysize(modalDatasetP) / 1024 / 1024 |> x->round(x, digits=2)) MBs")
		println("├ modalDatasetM size:\t\t$(Base.summarysize(modalDatasetM) / 1024 / 1024 |> x->round(x, digits=2)) MBs")
		println("└ modalDatasetG size:\t\t$(Base.summarysize(modalDatasetG) / 1024 / 1024 |> x->round(x, digits=2)) MBs")

		new{T}(modalDatasetP, modalDatasetM, modalDatasetG, n_instances, relations, features, grouped_featnaggrs, flattened_featnaggrs)
	end

end

# function modalDataset(X, features)
# 	computeModalDataset(X, features)
# end

################################################################################
################################################################################
################################################################################


################################################################################
################################################################################
################################################################################
################################################################################


# gammas is a structure holding threshold values for which propositional and modal split labels
#  are on the verge of truth 

# For generic worldTypes, gammas is an n-dim array of dictionaries indicized on the world itself.
#  On the other hand, when the structure of a world is known, its attributes are unrolled as
#  array dimensions; gammas then becomes an (n+k)-dim array,
#  where k is the complexity of the worldType.


# const GammaType{NTO, T} =
# Union{
# 	# worldType-agnostic
# 	AbstractArray{Dict{WorldType,NTuple{NTO, T}}, 3},
# 	# worldType=ModalLogic.OneWorld
# 	AbstractArray{T, 4},
# 	# worldType=ModalLogic.Interval
# 	AbstractArray{T, 6},
# 	# worldType=ModalLogic.Interval2D
# 	AbstractArray{T, 8}
# } where {WorldType<:AbstractWorld}

# const GammaSliceType{NTO, T} =
# Union{
# 	# worldType-agnostic
# 	AbstractArray{Dict{WorldType,NTuple{NTO, T}}, 3},
# 	# worldType=ModalLogic.OneWorld
# 	AbstractArray{T, 1},
# 	# worldType=ModalLogic.Interval
# 	AbstractArray{T, 3},
# 	# worldType=ModalLogic.Interval2D
# 	AbstractArray{T, 5}
# } where {WorldType<:AbstractWorld}

# TODO test with array-only gammas = Array{T, 4}(undef, 2, n_worlds(world_type(X.ontology), channel_size(X)), n_instances, n_attributes(X))
# TODO try something like gammas = fill(No: Dict{world_type(X.ontology),NTuple{NTO,T}}(), n_instances, n_attributes(X))
# gammas = Vector{Dict{AbstractRelation,Vector{Dict{world_type(X.ontology),NTuple{NTO,T}}}}}(undef, n_attributes(X))		
# TODO maybe use offset-arrays? https://docs.julialang.org/en/v1/devdocs/offset-arrays/


# worldType-agnostic gammas
# @inline initGammas(worldType::Type{WorldType}, T::Type, channel_size::Tuple, n_test_operators::Integer, n_instances::Integer, n_relations::Integer, n_features::Integer) where {WorldType<:AbstractWorld} =
# 	Array{Dict{worldType,NTuple{n_test_operators,T}}, 3}(undef, n_instances, n_relations, n_features)
# @inline gammasIsConsistent(gammas, X::OntologicalDataset{T, N, WorldType}, n_test_operators::Integer, n_relations::Integer) where {T, N, WorldType<:AbstractWorld} =
# 	(typeof(gammas)<:AbstractArray{Dict{WorldType,NTuple{n_test_operators,T}}, 3} && size(gammas) == (n_samples(X), n_relations, n_attributes(X)))
# @inline setGamma(gammas::AbstractArray{Dict{WorldType,NTuple{NTO,T}}, 3}, w::WorldType, i_instance::Integer, i_relation::Integer, i_feature::Integer, i_test_operator::Integer, threshold::T) where {WorldType<:AbstractWorld,NTO,T} =
# 	gammas[i_instance, i_relation, i_feature][w][i_test_operator] = threshold
# @inline initGammaSlice(worldType::Type{WorldType}, gammas::AbstractArray{Dict{WorldType,NTuple{NTO,T}}, 3}, i_instance::Integer, i_relation::Integer, i_feature::Integer) where {WorldType<:AbstractWorld,NTO,T} =
# 	gammas[i_instance, i_relation, i_feature] = Dict{WorldType,NTuple{NTO,T}}()
# @inline sliceGammas(worldType::Type{WorldType}, gammas::AbstractArray{Dict{WorldType,NTuple{NTO,T}}, 3}, i_instance::Integer, i_relation::Integer, i_feature::Integer) where {WorldType<:AbstractWorld,NTO,T} =
# 	gammas[i_instance, i_relation, i_feature]
# @inline setGammaSlice(gammaSlice::Dict{WorldType,NTuple{NTO,T}}, w::WorldType, i_test_operator::Integer, threshold::T) where {WorldType<:AbstractWorld,NTO,T} =
# 	gammaSlice[w][i_test_operator] = threshold
# @inline readGammaSlice(gammaSlice::Dict{WorldType,NTuple{NTO,T}}, w::WorldType, i_test_operator::Integer) where {WorldType<:AbstractWorld,NTO,T} =
# 	gammaSlice[w][i_test_operator]
# @inline sliceGammasByInstances(worldType::Type{WorldType}, gammas::AbstractArray{Dict{WorldType,NTuple{NTO,T}}, 3}, inds::AbstractVector{<:Integer}; return_view = false) where {WorldType<:AbstractWorld,NTO, T} =
# 	if return_view @view gammas[inds,:,:] else gammas[inds,:,:] end
# @inline function readGamma(
# 	gammas          :: AbstractArray{<:AbstractDict{WorldType,NTuple{NTO,T}},3},
# 	i_test_operator :: Integer,
# 	w               :: WorldType,
# 	i_instance      :: Integer,
# 	i_relation      :: Integer,
# 	i_feature       :: Integer) where {NTO,T,WorldType<:AbstractWorld}
# 	gammas[i_instance, i_relation, i_feature][w]
# end


# # Adimensional case (worldType = ModalLogic.OneWorld)
# @inline initGammas(worldType::Type{ModalLogic.OneWorld}, T::Type, channel_size::Tuple, n_test_operators::Integer, n_instances::Integer, n_relations::Integer, n_features::Integer) =
# 	Array{T, 4}(undef, n_test_operators, n_instances, n_relations, n_features)
# @inline gammasIsConsistent(gammas, X::OntologicalDataset{T, N, ModalLogic.OneWorld}, n_test_operators::Integer, n_relations::Integer) where {T, N}  =
# 	(typeof(gammas)<:AbstractArray{T, 4} && size(gammas) == (n_test_operators, n_samples(X), n_relations, n_attributes(X)))
# @inline setGamma(gammas::AbstractArray{T, 4}, w::ModalLogic.OneWorld, i_instance::Integer, i_relation::Integer, i_feature::Integer, i_test_operator::Integer, threshold::T) where {T} =
# 	gammas[i_test_operator, i_instance, i_relation, i_feature] = threshold
# @inline initGammaSlice(worldType::Type{ModalLogic.OneWorld}, gammas::AbstractArray{T, 4}, i_instance::Integer, i_relation::Integer, i_feature::Integer) where {T} =
# 	nothing
# @inline sliceGammas(worldType::Type{ModalLogic.OneWorld}, gammas::AbstractArray{T, 4}, i_instance::Integer, i_relation::Integer, i_feature::Integer) where {T} =
# 	@view gammas[:,i_instance, i_relation, i_feature]
# @inline setGammaSlice(gammaSlice::AbstractArray{T,1}, w::ModalLogic.OneWorld, i_test_operator::Integer, threshold::T) where {T} =
# 	gammaSlice[i_test_operator] = threshold
# @inline readGammaSlice(gammaSlice::AbstractArray{T,1}, w::ModalLogic.OneWorld, i_test_operator::Integer) where {T} =
# 	gammaSlice[i_test_operator]
# @inline sliceGammasByInstances(worldType::Type{ModalLogic.OneWorld}, gammas::AbstractArray{T, 4}, inds::AbstractVector{<:Integer}; return_view = false) where {T} =
# 	if return_view @view gammas[:,inds,:,:] else gammas[:,inds,:,:] end
# @inline function readGamma(
# 	gammas          :: AbstractArray{T, 4},
# 	i_test_operator :: Integer,
# 	w               :: ModalLogic.OneWorld,
# 	i_instance      :: Integer,
# 	i_relation      :: Integer,
# 	i_feature       :: Integer) where {T}
# 	gammas[i_test_operator, i_instance, i_relation, i_feature]
# end


# # 2D Interval case (worldType = ModalLogic.Interval2D)
# @inline initGammas(worldType::Type{ModalLogic.Interval2D}, T::Type, (X,Y)::NTuple{2,Integer}, n_test_operators::Integer, n_instances::Integer, n_relations::Integer, n_features::Integer) =
# 	Array{T, 8}(undef, n_test_operators, X, X+1, Y, Y+1, n_instances, n_relations, n_features)
# @inline gammasIsConsistent(gammas, X::OntologicalDataset{T, N, ModalLogic.Interval2D}, n_test_operators::Integer, n_relations::Integer) where {T, N, WorldType<:AbstractWorld} =
# 	(typeof(gammas)<:AbstractArray{T, 8} && size(gammas) == (n_test_operators, channel_size(X)[1], channel_size(X)[1]+1, channel_size(X)[2], channel_size(X)[2]+1, n_samples(X), n_relations, n_attributes(X)))
# @inline setGamma(gammas::AbstractArray{T, 8}, w::ModalLogic.Interval2D, i_instance::Integer, i_relation::Integer, i_feature::Integer, i_test_operators::Integer, threshold::T) where {T} =
# 	gammas[i_test_operators, w.x.x, w.x.y, w.y.x, w.y.y, i_instance, i_relation, i_feature] = threshold
# @inline initGammaSlice(worldType::Type{ModalLogic.Interval2D}, gammas::AbstractArray{T, 8}, n_instances::Integer, n_relations::Integer, n_features::Integer) where {T} =
# 	nothing
# @inline sliceGammas(worldType::Type{ModalLogic.Interval2D}, gammas::AbstractArray{T, 8}, i_instance::Integer, i_relation::Integer, i_feature::Integer) where {T} =
# 	@view gammas[:, :,:,:,:, i_instance, i_relation, i_feature]
# @inline setGammaSlice(gammaSlice::AbstractArray{T, 6}, w::ModalLogic.Interval2D, i_test_operators::Integer, threshold::T) where {T} =
# 	gammaSlice[i_test_operators, w.x.x, w.x.y, w.y.x, w.y.y] = threshold
# @inline readGammaSlice(gammaSlice::AbstractArray{T, 6}, w::ModalLogic.Interval2D, i_test_operators::Integer) where {T} =
# 	gammaSlice[i_test_operators, w.x.x, w.x.y, w.y.x, w.y.y]
# @inline sliceGammasByInstances(worldType::Type{ModalLogic.Interval2D}, gammas::AbstractArray{T, 8}, inds::AbstractVector{<:Integer}; return_view = false) where {T} =
# 	if return_view @view gammas[:, :,:,:,:, inds,:,:] else gammas[:, :,:,:,:, inds,:,:] end
# @inline function readGamma(
# 	gammas          :: AbstractArray{T, 8},
# 	i_test_operator :: Integer,
# 	w               :: ModalLogic.Interval2D,
# 	i_instance      :: Integer,
# 	i_relation      :: Integer,
# 	i_feature       :: Integer) where {T}
# 	gammas[i_test_operator, w.x.x, w.x.y, w.y.x, w.y.y, i_instance, i_relation, i_feature] # TODO try without view
# end









# TODO test which implementation is the best for the 2D case with different memory layout for gammas

# 3x3 spatial window, 12 instances:
# 	Array7 3x4: 90.547 s (1285579691 allocations: 65.67 GiB)
# 	Array5: 105.759 s (1285408103 allocations: 65.70 GiB)
# 	Array7 3x3 con [idx-1]:  113.278 s (1285408102 allocations: 65.69 GiB)
# 	Generic Dict:  100.272 s (1284316309 allocations: 65.64 GiB)
# 	Array8:   100.517 s (1281158366 allocations: 65.49 GiB)
# ---
# using array(undef, ...):	 101.921 s (1285848739 allocations: 65.70 GiB)
# using T[]	100.443 s (1282663890 allocations: 65.69 GiB)

# @inline initGammas(worldType::Type{ModalLogic.Interval2D}, T::Type, (X,Y)::NTuple{2,Integer}, n_test_operators::Integer, n_instances::Integer, n_relations::Integer, n_features::Integer) =
# 	Array{NTuple{n_test_operators,T}, 5}(undef, div((X*(X+1)),2), div((Y*(Y+1)),2), n_instances, n_relations, n_features)
# @inline setGamma(gammas::AbstractArray{NTuple{NTO,T}, 5}, w::ModalLogic.Interval2D, i_instance::Integer, i_relation::Integer, i_feature::Integer, thresholds::NTuple{NTO,T}) where {NTO,T} =
# 	gammas[w.x.x+div((w.x.y-2)*(w.x.y-1),2), w.y.x+div((w.y.y-2)*(w.y.y-1),2), i_instance, i_relation, i_feature] = thresholds
# @inline initGammaSlice(worldType::Type{ModalLogic.Interval2D}, gammas::AbstractArray{NTuple{NTO,T}, 5}, n_instances::Integer, n_relations::Integer, n_features::Integer) where {NTO,T} =
# 	nothing
# @inline sliceGammas(worldType::Type{ModalLogic.Interval2D}, gammas::AbstractArray{NTuple{NTO,T}, 5}, i_instance::Integer, i_relation::Integer, i_feature::Integer) where {NTO,T} =
# 	@view gammas[:,:, i_instance, i_relation, i_feature]
# @inline setGammaSlice(gammaSlice::AbstractArray{NTuple{NTO,T}, 2}, w::ModalLogic.Interval2D, thresholds::NTuple{NTO,T}) where {NTO,T} =
# 	gammaSlice[w.x.x+div((w.x.y-2)*(w.x.y-1),2), w.y.x+div((w.y.y-2)*(w.y.y-1),2)] = thresholds
# @inline function readGamma(
# 	gammas     :: AbstractArray{NTuple{NTO,T},N},
# 	w          :: ModalLogic.Interval2D,
# 	i, i_relation, feature) where {N,NTO,T}
# 	gammas[w.x.x+div((w.x.y-2)*(w.x.y-1),2), w.y.x+div((w.y.y-2)*(w.y.y-1),2), i, i_relation, feature]
# end

# @inline initGammas(worldType::Type{ModalLogic.Interval2D}, T::Type, (X,Y)::NTuple{2,Integer}, n_test_operators::Integer, n_instances::Integer, n_relations::Integer, n_features::Integer) =
# 	Array{NTuple{n_test_operators,T}, 7}(undef, X, X, Y, Y, n_instances, n_relations, n_features)
# @inline setGamma(gammas::AbstractArray{NTuple{NTO,T}, 7}, w::ModalLogic.Interval2D, i_instance::Integer, i_relation::Integer, i_feature::Integer, thresholds::NTuple{NTO,T}) where {NTO,T} =
# 	gammas[w.x.x, w.x.y-1, w.y.x, w.y.y-1, i_instance, i_relation, i_feature] = thresholds
# @inline initGammaSlice(worldType::Type{ModalLogic.Interval2D}, gammas::AbstractArray{NTuple{NTO,T}, 7}, n_instances::Integer, n_relations::Integer, n_features::Integer) where {NTO,T} =
# 	nothing
# @inline sliceGammas(worldType::Type{ModalLogic.Interval2D}, gammas::AbstractArray{NTuple{NTO,T}, 7}, i_instance::Integer, i_relation::Integer, i_feature::Integer) where {NTO,T} =
# 	@view gammas[:,:,:,:, i_instance, i_relation, i_feature]
# @inline setGammaSlice(gammaSlice::AbstractArray{NTuple{NTO,T}, 4}, w::ModalLogic.Interval2D, thresholds::NTuple{NTO,T}) where {NTO,T} =
# 	gammaSlice[w.x.x, w.x.y-1, w.y.x, w.y.y-1] = thresholds
# @inline function readGamma(
# 	gammas     :: AbstractArray{NTuple{NTO,T},N},
# 	w          :: ModalLogic.Interval2D,
# 	i, i_relation, feature) where {N,NTO,T}
# 	gammas[w.x.x, w.x.y-1, w.y.x, w.y.y-1, i, i_relation, feature]
# end

# @inline initGammas(worldType::Type{ModalLogic.Interval2D}, T::Type, (X,Y)::NTuple{2,Integer}, n_test_operators::Integer, n_instances::Integer, n_relations::Integer, n_features::Integer) =
# 	Array{T, 8}(undef, n_test_operators, X, X+1, Y, Y+1, n_instances, n_relations, n_features)
# @inline setGamma(gammas::AbstractArray{T, 8}, w::ModalLogic.Interval2D, i_instance::Integer, i_relation::Integer, i_feature::Integer, i_test_operator::Integer, threshold::T) where {NTO,T} =
# 	gammas[i_test_operator, w.x.x, w.x.y, w.y.x, w.y.y, i_instance, i_relation, i_feature] = threshold
# @inline initGammaSlice(worldType::Type{ModalLogic.Interval2D}, gammas::AbstractArray{T, 8}, n_instances::Integer, n_relations::Integer, n_features::Integer) where {NTO,T} =
# 	nothing
# @inline sliceGammas(worldType::Type{ModalLogic.Interval2D}, gammas::AbstractArray{T, 8}, i_instance::Integer, i_relation::Integer, i_feature::Integer) where {NTO,T} =
# 	@view gammas[:,:,:,:,:, i_instance, i_relation, i_feature]
# @inline setGammaSlice(gammaSlice::AbstractArray{T, 5}, w::ModalLogic.Interval2D, i_test_operator::Integer, threshold::T) where {NTO,T} =
# 	gammaSlice[i_test_operator, w.x.x, w.x.y, w.y.x, w.y.y] = threshold
# @inline function readGamma(
# 	gammas     :: AbstractArray{T,N},
# 	w          :: ModalLogic.Interval2D,
# 	i, i_relation, feature) where {N,T}
# 	@view gammas[:,w.x.x, w.x.y, w.y.x, w.y.y, i, i_relation, feature]
# end

################################################################################
################################################################################
################################################################################
################################################################################
