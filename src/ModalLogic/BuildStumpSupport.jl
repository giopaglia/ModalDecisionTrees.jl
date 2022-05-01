
# FeaturedWorldDataset(
# 		X::OntologicalDataset{T, N, WorldType},
# 		features::AbstractVector{<:ModalFeature}
# 	) where {T, N, WorldType<:AbstractWorld} = FeaturedWorldDataset{T, WorldType}(X, features)

Base.@propagate_inbounds function FeaturedWorldDataset(
		X::OntologicalDataset
	)

	@logmsg DTOverview "OntologicalDataset -> FeatModalDataset"

	features = X.features
	
	n_instances = n_samples(X)
	n_features = length(features)
	
    # Prepare FeaturedWorldDataset (the actual implementation depends on the OntologicalDataset)
	fwd = initFeaturedWorldDataset(X, n_features)

	# Load any (possible) external features
	if any(isa.(features, ExternalFWDFeature))
		i_external_features = first.(filter(((i_feature,is_external_fwd),)->(is_external_fwd), collect(enumerate(isa.(features, ExternalFWDFeature)))))
		for i_feature in i_external_features
			feature = features[i_feature]
			modalDatasetSetFeature(fwd, i_feature, feature.fwd)
		end
	end

	# Load any internal features
	i_features = first.(filter(((i_feature,is_external_fwd),)->!(is_external_fwd), collect(enumerate(isa.(features, ExternalFWDFeature)))))
	enum_features = zip(i_features, features[i_features])

	# Compute features
	@inbounds Threads.@threads for i_instance in 1:n_instances
		@logmsg DTDebug "Instance $(i_instance)/$(n_instances)"
		
		if i_instance == 1 || ((i_instance+1) % (floor(Int, ((n_instances)/4))+1)) == 0
			@logmsg DTOverview "Instance $(i_instance)/$(n_instances)"
		end

		# instance = get_instance(X, i_instance)
		# @logmsg DTDebug "instance" instance

		for w in accAll_function(X, i_instance)
			
			initFeaturedWorldDatasetWorldSlice(fwd, w)

			@logmsg DTDebug "World" w

			for (i_feature,feature) in enum_features

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
# function checkModalDatasetConsistency(modalDataset, X::OntologicalDataset{T, N, WorldType}, features::AbstractVector{<:ModalFeature}) where {T, N, WorldType<:AbstractWorld}
# 	if !(modalDatasetIsConsistent(modalDataset, X, length(features)))
# 		throw_n_log("The provided modalDataset structure is not consistent with the expected dataset, test operators and/or relations!"
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
modalDatasetSetFeature(fwd::OneWorldFeaturedWorldDataset{T}, i_feature::Integer, feature_fwd::Array{T, 1}) where {T} =
	fwd.d[:, i_feature] = feature_fwd
function slice_dataset(fwd::OneWorldFeaturedWorldDataset{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
	OneWorldFeaturedWorldDataset{T}(if return_view @view fwd.d[inds,:] else fwd.d[inds,:] end)
end
modalDatasetChannelSlice(fwd::OneWorldFeaturedWorldDataset{T}, i_instance::Integer, i_feature::Integer) where {T} =
	fwd.d[i_instance, i_feature]
const OneWorldFeaturedChannel{T} = T
# const OneWorldFeaturedChannel{T} = Union{T}
modalDatasetChannelSliceGet(fwc::T #=OneWorldFeaturedChannel{T}=#, w::OneWorld) where {T} = fwc

n_samples(fwd::OneWorldFeaturedWorldDataset{T}) where {T}  = size(fwd, 1)
n_features(fwd::OneWorldFeaturedWorldDataset{T}) where {T} = size(fwd, 2)
Base.getindex(
	fwd         :: OneWorldFeaturedWorldDataset{T},
	i_instance  :: Integer,
	w           :: OneWorld,
	i_feature   :: Integer) where {T} = fwd.d[i_instance, i_feature]
Base.size(fwd::OneWorldFeaturedWorldDataset{T}, args::Vararg) where {T} = size(fwd.d, args...)
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
modalDatasetSetFeature(fwd::IntervalFeaturedWorldDataset{T}, i_feature::Integer, feature_fwd::Array{T, 3}) where {T} =
	fwd.d[:, :, :, i_feature] = feature_fwd
function slice_dataset(fwd::IntervalFeaturedWorldDataset{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
	IntervalFeaturedWorldDataset{T}(if return_view @view fwd.d[:,:,inds,:] else fwd.d[:,:,inds,:] end)
end
modalDatasetChannelSlice(fwd::IntervalFeaturedWorldDataset{T}, i_instance::Integer, i_feature::Integer) where {T} =
	@views fwd.d[:,:,i_instance, i_feature]
const IntervalFeaturedChannel{T} = AbstractArray{T, 2}
modalDatasetChannelSliceGet(fwc::IntervalFeaturedChannel{T}, w::Interval) where {T} =
	fwc[w.x, w.y]

n_samples(fwd::IntervalFeaturedWorldDataset{T}) where {T}  = size(fwd, 3)
n_features(fwd::IntervalFeaturedWorldDataset{T}) where {T} = size(fwd, 4)
Base.getindex(
	fwd         :: IntervalFeaturedWorldDataset{T},
	i_instance  :: Integer,
	w           :: Interval,
	i_feature   :: Integer) where {T} = fwd.d[w.x, w.y, i_instance, i_feature]
Base.size(fwd::IntervalFeaturedWorldDataset{T}, args::Vararg) where {T} = size(fwd.d, args...)
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
modalDatasetSetFeature(fwd::Interval2DFeaturedWorldDataset{T}, i_feature::Integer, feature_fwd::Array{T, 5}) where {T} =
	fwd.d[:, :, :, :, :, i_feature] = feature_fwd
function slice_dataset(fwd::Interval2DFeaturedWorldDataset{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
	Interval2DFeaturedWorldDataset{T}(if return_view @view fwd.d[:,:,:,:,inds,:] else fwd.d[:,:,:,:,inds,:] end)
end
modalDatasetChannelSlice(fwd::Interval2DFeaturedWorldDataset{T}, i_instance::Integer, i_feature::Integer) where {T} =
	@views fwd.d[:,:,:,:,i_instance, i_feature]
const Interval2DFeaturedChannel{T} = AbstractArray{T, 4}
modalDatasetChannelSliceGet(fwc::Interval2DFeaturedChannel{T}, w::Interval2D) where {T} =
	fwc[w.x.x, w.x.y, w.y.x, w.y.y]

n_samples(fwd::Interval2DFeaturedWorldDataset{T}) where {T}  = size(fwd, 5)
n_features(fwd::Interval2DFeaturedWorldDataset{T}) where {T} = size(fwd, 6)
Base.getindex(
	fwd         :: Interval2DFeaturedWorldDataset{T},
	i_instance  :: Integer,
	w           :: Interval2D,
	i_feature   :: Integer) where {T} = fwd.d[w.x.x, w.x.y, w.y.x, w.y.y, i_instance, i_feature]
Base.size(fwd::Interval2DFeaturedWorldDataset{T}, args::Vararg) where {T} = size(fwd.d, args...)
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
# 	# flattened_featsnaggrs = Tuple{<:ModalFeature,<:Aggregator}[]
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
Base.getindex(
	fmds         :: GenericFMDStumpSupport{T, WorldType},
	i_instance   :: Integer,
	w            :: WorldType,
	i_featsnaggr :: Integer,
	i_relation   :: Integer) where {T, WorldType<:AbstractWorld} = fmds.d[i_instance, i_featsnaggr, i_relation][w]
Base.size(fmds::GenericFMDStumpSupport, args::Vararg) = size(fmds.d, args...)
world_type(fmds::GenericFMDStumpSupport{T, WorldType}) where {T, WorldType} = WorldType

initFMDStumpSupport(fmd::FeatModalDataset{T, WorldType}, n_featsnaggrs::Integer, n_relations::Integer; perform_initialization = false) where {T, WorldType} = begin
	if perform_initialization
		_fmd_m = fill!(Array{Dict{WorldType,Union{T,Nothing}}, 3}(undef, n_samples(fmd), n_featsnaggrs, n_relations), nothing)
		GenericFMDStumpSupport{Union{T,Nothing}, WorldType}(_fmd_m)
	else
		_fmd_m = Array{Dict{WorldType,T}, 3}(undef, n_samples(fmd), n_featsnaggrs, n_relations)
		GenericFMDStumpSupport{T, WorldType}(_fmd_m)
	end
end
# modalDatasetIsConsistent_m(modalDataset, fmd::FeatModalDataset{T, AbstractWorld}, n_featsnaggrs::Integer, n_relations::Integer) where {T, WorldType} =
	# (typeof(modalDataset)<:AbstractArray{T, 7} && size(modalDataset) == (max_channel_size(fmd)[1], max_channel_size(fmd)[1]+1, n_samples(fmd), n_featsnaggrs, n_relations))
initFMDStumpSupportWorldSlice(fmds::GenericFMDStumpSupport{T, WorldType}, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer) where {T, WorldType} =
	fmds.d[i_instance, i_featsnaggr, i_relation] = Dict{WorldType,T}()
FMDStumpSupportSet(fmds::GenericFMDStumpSupport{T, WorldType}, w::AbstractWorld, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T, WorldType} =
	fmds.d[i_instance, i_featsnaggr, i_relation][w] = threshold
function slice_dataset(fmds::GenericFMDStumpSupport{T, WorldType}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T, WorldType}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
	GenericFMDStumpSupport{T, WorldType}(if return_view @view fmds.d[inds,:,:] else fmds.d[inds,:,:] end)
end



struct OneWorldFMDStumpSupport{T} <: AbstractFMDStumpSupport{T, OneWorld}
	d :: AbstractArray{T, 3}
end

n_samples(fmds::OneWorldFMDStumpSupport{T}) where {T}  = size(fmds, 1)
n_featsnaggrs(fmds::OneWorldFMDStumpSupport{T}) where {T} = size(fmds, 2)
n_relations(fmds::OneWorldFMDStumpSupport{T}) where {T} = size(fmds, 3)
Base.getindex(
	fmds         :: OneWorldFMDStumpSupport{T},
	i_instance   :: Integer,
	w            :: OneWorld,
	i_featsnaggr :: Integer,
	i_relation   :: Integer) where {T} = fmds.d[i_instance, i_featsnaggr, i_relation]
Base.size(fmds::OneWorldFMDStumpSupport{T}, args::Vararg) where {T} = size(fmds.d, args...)
world_type(fmds::OneWorldFMDStumpSupport{T}) where {T} = OneWorld

initFMDStumpSupport(fmd::FeatModalDataset{T, OneWorld}, n_featsnaggrs::Integer, n_relations::Integer; perform_initialization = false) where {T} = begin
	if perform_initialization
		_fmd_m = fill!(Array{Union{T,Nothing}, 3}(undef, n_samples(fmd), n_featsnaggrs, n_relations), nothing)
		OneWorldFMDStumpSupport{Union{T,Nothing}}(_fmd_m)
	else
		_fmd_m = Array{T, 3}(undef, n_samples(fmd), n_featsnaggrs, n_relations)
		OneWorldFMDStumpSupport{T}(_fmd_m)
	end
end
# modalDatasetIsConsistent_m(modalDataset, fmd::FeatModalDataset{T, OneWorld}, n_featsnaggrs::Integer, n_relations::Integer) where {T} =
	# (typeof(modalDataset)<:AbstractArray{T, 3} && size(modalDataset) == (n_samples(fmd), n_featsnaggrs, n_relations))
initFMDStumpSupportWorldSlice(fmds::OneWorldFMDStumpSupport, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer) =
	nothing
FMDStumpSupportSet(fmds::OneWorldFMDStumpSupport{T}, w::OneWorld, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
	fmds.d[i_instance, i_featsnaggr, i_relation] = threshold
function slice_dataset(fmds::OneWorldFMDStumpSupport{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
	OneWorldFMDStumpSupport{T}(if return_view @view fmds.d[inds,:,:] else fmds.d[inds,:,:] end)
end



struct IntervalFMDStumpSupport{T} <: AbstractFMDStumpSupport{T, Interval}
	d :: AbstractArray{T, 5}
end

n_samples(fmds::IntervalFMDStumpSupport{T}) where {T}  = size(fmds, 3)
n_featsnaggrs(fmds::IntervalFMDStumpSupport{T}) where {T} = size(fmds, 4)
n_relations(fmds::IntervalFMDStumpSupport{T}) where {T} = size(fmds, 5)
Base.getindex(
	fmds         :: IntervalFMDStumpSupport{T},
	i_instance   :: Integer,
	w            :: Interval,
	i_featsnaggr :: Integer,
	i_relation   :: Integer) where {T} = fmds.d[w.x, w.y, i_instance, i_featsnaggr, i_relation]
Base.size(fmds::IntervalFMDStumpSupport{T}, args::Vararg) where {T} = size(fmds.d, args...)
world_type(fmds::IntervalFMDStumpSupport{T}) where {T} = Interval

initFMDStumpSupport(fmd::FeatModalDataset{T, Interval}, n_featsnaggrs::Integer, n_relations::Integer; perform_initialization = false) where {T} = begin
	if perform_initialization
		_fmd_m = fill!(Array{Union{T,Nothing}, 5}(undef, size(fmd.fwd, 1), size(fmd.fwd, 2), n_samples(fmd), n_featsnaggrs, n_relations), nothing)
		IntervalFMDStumpSupport{Union{T,Nothing}}(_fmd_m)
	else
		_fmd_m = Array{T, 5}(undef, size(fmd.fwd, 1), size(fmd.fwd, 2), n_samples(fmd), n_featsnaggrs, n_relations)
		IntervalFMDStumpSupport{T}(_fmd_m)
	end
end
# modalDatasetIsConsistent_m(modalDataset, fmd::FeatModalDataset{T, Interval}, n_featsnaggrs::Integer, n_relations::Integer) where {T} =
	# (typeof(modalDataset)<:AbstractArray{T, 5} && size(modalDataset) == (max_channel_size(fmd)[1], max_channel_size(fmd)[1]+1, n_samples(fmd), n_featsnaggrs, n_relations))
initFMDStumpSupportWorldSlice(fmds::IntervalFMDStumpSupport, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer) =
	nothing
FMDStumpSupportSet(fmds::IntervalFMDStumpSupport{T}, w::Interval, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
	fmds.d[w.x, w.y, i_instance, i_featsnaggr, i_relation] = threshold
function slice_dataset(fmds::IntervalFMDStumpSupport{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
	IntervalFMDStumpSupport{T}(if return_view @view fmds.d[:,:,inds,:,:] else fmds.d[:,:,inds,:,:] end)
end


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

# initFMDStumpSupport(fmd::FeatModalDataset{T, Interval2D}, n_featsnaggrs::Integer, n_relations::Integer; perform_initialization = false) where {T} = begin
# 	if perform_initialization
# 		_fmd_m = fill!(Array{Union{T,Nothing}, 7}(undef, size(fmd.fwd, 1), size(fmd.fwd, 2), size(fmd.fwd, 3), size(fmd.fwd, 4), n_samples(fmd), n_featsnaggrs, n_relations), nothing)
# 		Interval2DFMDStumpSupport{Union{T,Nothing}}(_fmd_m)
# 	else
# 		_fmd_m = Array{T, 7}(undef, size(fmd.fwd, 1), size(fmd.fwd, 2), size(fmd.fwd, 3), size(fmd.fwd, 4), n_samples(fmd), n_featsnaggrs, n_relations)
# 		Interval2DFMDStumpSupport{T}(_fmd_m)
# 	end
# end
# # modalDatasetIsConsistent_m(modalDataset, fmd::FeatModalDataset{T, Interval2D}, n_featsnaggrs::Integer, n_relations::Integer) where {T} =
# 	# (typeof(modalDataset)<:AbstractArray{T, 7} && size(modalDataset) == (max_channel_size(fmd)[1], max_channel_size(fmd)[1]+1, n_samples(fmd), n_featsnaggrs, n_relations))
# initFMDStumpSupportWorldSlice(fmds::Interval2DFMDStumpSupport, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer) =
# 	nothing
# FMDStumpSupportSet(fmds::Interval2DFMDStumpSupport{T}, w::Interval2D, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
# 	fmds.d[w.x.x, w.x.y, w.y.x, w.y.y, i_instance, i_featsnaggr, i_relation] = threshold
# function slice_dataset(fmds::Interval2DFMDStumpSupport{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
# @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
# 	Interval2DFMDStumpSupport{T}(if return_view @view fmds.d[:,:,:,:,inds,:,:] else fmds.d[:,:,:,:,inds,:,:] end)
# end

struct Interval2DFMDStumpSupport{T} <: AbstractFMDStumpSupport{T, Interval2D}
	d :: AbstractArray{T, 5}
end

n_samples(fmds::Interval2DFMDStumpSupport{T}) where {T}  = size(fmds, 3)
n_featsnaggrs(fmds::Interval2DFMDStumpSupport{T}) where {T} = size(fmds, 4)
n_relations(fmds::Interval2DFMDStumpSupport{T}) where {T} = size(fmds, 5)
Base.getindex(
	fmds         :: Interval2DFMDStumpSupport{T},
	i_instance   :: Integer,
	w            :: Interval2D,
	i_featsnaggr :: Integer,
	i_relation   :: Integer) where {T} = fmds.d[w.x.x+div((w.x.y-2)*(w.x.y-1),2), w.y.x+div((w.y.y-2)*(w.y.y-1),2), i_instance, i_featsnaggr, i_relation]
Base.size(fmds::Interval2DFMDStumpSupport{T}, args::Vararg) where {T} = size(fmds.d, args...)
world_type(fmds::Interval2DFMDStumpSupport{T}) where {T} = Interval2D

initFMDStumpSupport(fmd::FeatModalDataset{T, Interval2D}, n_featsnaggrs::Integer, n_relations::Integer; perform_initialization = false) where {T} = begin
	if perform_initialization
		_fmd_m = fill!(Array{Union{T,Nothing}, 5}(undef, div(size(fmd.fwd, 1)*size(fmd.fwd, 2),2), div(size(fmd.fwd, 3)*size(fmd.fwd, 4),2), n_samples(fmd), n_featsnaggrs, n_relations), nothing)
		Interval2DFMDStumpSupport{Union{T,Nothing}}(_fmd_m)
	else
		_fmd_m = Array{T, 5}(undef, div(size(fmd.fwd, 1)*size(fmd.fwd, 2),2), div(size(fmd.fwd, 3)*size(fmd.fwd, 4),2), n_samples(fmd), n_featsnaggrs, n_relations)
		Interval2DFMDStumpSupport{T}(_fmd_m)
	end
end
# modalDatasetIsConsistent_m(modalDataset, fmd::FeatModalDataset{T, Interval2D}, n_featsnaggrs::Integer, n_relations::Integer) where {T} =
	# (typeof(modalDataset)<:AbstractArray{T, 5} && size(modalDataset) == (max_channel_size(fmd)[1], max_channel_size(fmd)[1]+1, n_samples(fmd), n_featsnaggrs, n_relations))
initFMDStumpSupportWorldSlice(fmds::Interval2DFMDStumpSupport, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer) =
	nothing
FMDStumpSupportSet(fmds::Interval2DFMDStumpSupport{T}, w::Interval2D, i_instance::Integer, i_featsnaggr::Integer, i_relation::Integer, threshold::T) where {T} =
	fmds.d[w.x.x+div((w.x.y-2)*(w.x.y-1),2), w.y.x+div((w.y.y-2)*(w.y.y-1),2), i_instance, i_featsnaggr, i_relation] = threshold
function slice_dataset(fmds::Interval2DFMDStumpSupport{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
	Interval2DFMDStumpSupport{T}(if return_view @view fmds.d[:,:,inds,:,:] else fmds.d[:,:,inds,:,:] end)
end


# Note: global support is world-agnostic
struct FMDStumpGlobalSupportArray{T} <: AbstractFMDStumpGlobalSupport{T}
	d :: AbstractArray{T, 2}
end

n_samples(fmds::FMDStumpGlobalSupportArray{T}) where {T}  = size(fmds, 1)
n_featsnaggrs(fmds::FMDStumpGlobalSupportArray{T}) where {T} = size(fmds, 2)
Base.getindex(
	fmds         :: FMDStumpGlobalSupportArray{T},
	i_instance   :: Integer,
	i_featsnaggr  :: Integer) where {T} = fmds.d[i_instance, i_featsnaggr]
Base.size(fmds::FMDStumpGlobalSupportArray{T}, args::Vararg) where {T} = size(fmds.d, args...)

initFMDStumpGlobalSupport(fmd::FeatModalDataset{T}, n_featsnaggrs::Integer) where {T} =
	FMDStumpGlobalSupportArray{T}(Array{T, 2}(undef, n_samples(fmd), n_featsnaggrs))
# modalDatasetIsConsistent_g(modalDataset, fmd::FeatModalDataset{T, ModalLogic.anyworld...} n_featsnaggrs::Integer) where {T, N, WorldType<:AbstractWorld} =
# 	(typeof(modalDataset)<:AbstractArray{T, 2} && size(modalDataset) == (n_samples(fmd), n_featsnaggrs))
FMDStumpGlobalSupportSet(fmds::FMDStumpGlobalSupportArray{T}, i_instance::Integer, i_featsnaggr::Integer, threshold::T) where {T} =
	fmds.d[i_instance, i_featsnaggr] = threshold
function slice_dataset(fmds::FMDStumpGlobalSupportArray{T}, inds::AbstractVector{<:Integer}; allow_no_instances = false, return_view = false) where {T}
    @assert (allow_no_instances || length(inds) > 0) "Can't apply empty slice to dataset."
	FMDStumpGlobalSupportArray{T}(if return_view @view fmds.d[inds,:] else fmds.d[inds,:] end)
end


Base.@propagate_inbounds function computeModalDatasetStumpSupport(
		fmd                 :: FeatModalDataset{T, WorldType},
		grouped_featsnaggrs :: AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}};
		computeRelationGlob = false,
		simply_init_modal = false,
	) where {T, N, WorldType<:AbstractWorld}
	
	@logmsg DTOverview "FeatModalDataset -> StumpFeatModalDataset"

	fwd = fmd.fwd
	features = fmd.features
	relations = fmd.relations

	computefmd_g =
		if RelationGlob in relations
			throw_n_log("RelationGlob in relations: $(relations)")
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
	fmd_m = initFMDStumpSupport(fmd, n_featsnaggrs, n_relations; perform_initialization = simply_init_modal)

	# Prepare fmd_g
	fmd_g =
		if computefmd_g
			initFMDStumpGlobalSupport(fmd, n_featsnaggrs)
		else
			nothing
	end

	@inbounds Threads.@threads for i_instance in 1:n_instances
		@logmsg DTDebug "Instance $(i_instance)/$(n_instances)"
		
		if i_instance == 1 || ((i_instance+1) % (floor(Int, ((n_instances)/4))+1)) == 0
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

			if !simply_init_modal
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
	end
	fmd_m, fmd_g
end
