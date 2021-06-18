# TODO note that these splitting functions simply cut the dataset in two,
#  and they don't necessarily produce balanced cuts. To produce balanced cuts,
#  one must manually stratify the dataset beforehand
function traintestsplit((Xs,Y)::Tuple{AbstractVector{<:GenericDataset},AbstractVector{String}}, args::Vararg)
	Xs_train = Vector(undef, length(Xs))
	Xs_test  = Vector(undef, length(Xs))
	the_Y_train = nothing
	the_Y_test = nothing
	for (i,X) in enumerate(Xs)

		(X_train,Y_train), (X_test,Y_test) = traintestsplit((X,Y), args...)
		Xs_train[i]      = X_train
		Xs_test[i]       = X_test
		the_Y_train      = Y_train
		the_Y_test       = Y_test
	end

	(Xs_train, the_Y_train), (Xs_test, the_Y_test)
end

function traintestsplit((X,Y)::Tuple{GenericDataset,AbstractVector{String}}, split_threshold::AbstractFloat; is_balanced = true)
	if split_threshold == 1.0
		# Full train
		return (X,Y), (X,Y)
	end

	num_instances = n_samples(X)
	spl = ceil(Int, num_instances*split_threshold)
	# In the binary case, make it even
	if length(unique(Y)) == 2 && is_balanced
		spl = isodd(spl) ? (spl-1) : spl
	end
	X_train = ModalLogic.slice_dataset(X, 1:spl)
	Y_train = Y[1:spl]
	X_test  = ModalLogic.slice_dataset(X, spl+1:num_instances)
	Y_test  = Y[spl+1:end]
	(X_train,Y_train), (X_test,Y_test)
end

function slice_mf_dataset((Xs,Y)::Tuple{AbstractVector{<:GenericDataset},AbstractVector}, dataset_slice::AbstractVector)
	(map(X->ModalLogic.slice_dataset(X, dataset_slice), Xs), Y[dataset_slice])
end

# function slice_dataset((X,Y)::Tuple{MatricialDataset,AbstractVector}, dataset_slice::AbstractVector)
# 	(ModalLogic.slice_dataset(X, dataset_slice), Y[dataset_slice])
# end

# Scale and round dataset to fit into a certain datatype's range:
# For integers: find minimum and maximum (ignoring Infs), and rescale the dataset
# For floating-point, numbers, round
roundDataset((X,Y)::Tuple{Any,Vector}, type::Type = UInt8) = (mapArrayToDataType(type, X),Y)
roundDataset(((X_train,Y_train),(X_test,Y_test))::Tuple{Tuple,Tuple}, type::Type = UInt8) = begin
	X_train, X_test = mapArrayToDataType(type, (X_train, X_test))
	(X_train,Y_train),(X_test,Y_test)
end

# Multiframe datasets: scale each frame independently
mapArrayToDataType(type::Type, mf_array::AbstractArray{<:AbstractArray}) = begin
	map(a->mapArrayToDataType(type, a), mf_array)
end
mapArrayToDataType(type::Type, mf_arrays::NTuple{N,AbstractArray{<:AbstractArray}}) where {N} = begin
	zip(map((arrays)->mapArrayToDataType(type, arrays), zip(mf_arrays...))...)
end

# Integer: rescale datasets in the available discrete range.
#  for tuples of datasets, min and max are calculated across all datasets
mapArrayToDataType(type::Type{<:Integer}, arrays::NTuple{N,AbstractArray{<:Real}}) where {N} = begin
	minVal, maxVal = minimum(minimum.(array)), maximum(maximum.(array))
	map((array)->mapArrayToDataType(type,array), arrays; minVal = minVal, maxVal = maxVal)
end
mapArrayToDataType(type::Type{<:Integer}, array::AbstractArray{<:Real}; minVal = minimum(array), maxVal = maximum(array)) = begin
	normalized_array = (array.-minVal)./(maxVal-minVal)
	typemin(type) .+ round.(type, (big(typemax(type))-big(typemin(type)))*normalized_array)
end

# Floats: just round to the nearest float
mapArrayToDataType(type::Type{<:AbstractFloat}, arrays::NTuple{N,AbstractArray{<:Real}}) where {N} = begin
	map((array)->mapArrayToDataType(type,array), arrays)
end

mapArrayToDataType(type::Type{<:AbstractFloat}, array::AbstractArray{<:Real}) = begin
	# TODO worry about eps of the target type and the magnitude of values in array
	#  (and eventually scale up or down the array). Also change mapArrayToDataType(type, Xs::Tuple) then
	type.(array)
end
