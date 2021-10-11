import DecisionTree.ModalLogic: concat_datasets, slice_dataset, dataset_has_nonevalues

# TODO note that these splitting functions simply cut the dataset in two,
#  and they don't necessarily produce balanced cuts. To produce balanced cuts,
#  one must manually stratify the dataset beforehand
function traintestsplit((Xs,Y)::Tuple{AbstractVector{<:GenericDataset},AbstractVector}, split_threshold::AbstractFloat; is_balanced = true, return_view = false)
	Xs_train = Vector(undef, length(Xs))
	Xs_test  = Vector(undef, length(Xs))
	the_Y_train = nothing
	the_Y_test = nothing
	for (i,X) in enumerate(Xs)
		(X_train,Y_train), (X_test,Y_test) = traintestsplit((X,Y), split_threshold; is_balanced = is_balanced, return_view = return_view)
		Xs_train[i]      = X_train
		Xs_test[i]       = X_test
		the_Y_train      = Y_train
		the_Y_test       = Y_test
	end

	(Xs_train, the_Y_train), (Xs_test, the_Y_test)
end

function traintestsplit((X,Y)::Tuple{GenericDataset,AbstractVector}, split_threshold::AbstractFloat; is_balanced = true, return_view = false)
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
	X_train = slice_dataset(X, 1:spl; return_view = return_view)
	Y_train = Y[1:spl]
	X_test  = slice_dataset(X, spl+1:num_instances; return_view = return_view)
	Y_test  = Y[spl+1:end]
	(X_train,Y_train), (X_test,Y_test)
end

# function slice_dataset((X,Y)::Tuple{MatricialDataset,AbstractVector}, dataset_slice::AbstractVector; return_view = false)
# 	(slice_dataset(X, dataset_slice; return_view = return_view), Y[dataset_slice])
# end

# Scale and round dataset to fit into a certain datatype's range:
# For integers: find minimum and maximum (ignoring Infs), and rescale the dataset
# For floating-point, numbers, round
round_dataset((X,Y)::Tuple{Any,Vector}, type::Type = UInt8) = (mapArrayToDataType(type, X),Y)
round_dataset(((X_train,Y_train),(X_test,Y_test))::Tuple{Tuple,Tuple}, type::Type = UInt8) = begin
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

balanced_dataset_slice(dataset::Tuple{Tuple{AbstractVector{<:GenericDataset},AbstractVector},NTuple{N,Integer}}, dataseeds::Vector{<:Integer}; samples_per_class_share::Union{Nothing,AbstractFloat} = nothing) where N = begin
	d, n_label_samples = dataset
	d, [balanced_dataset_slice(n_label_samples, dataseed; samples_per_class_share = samples_per_class_share) for dataseed in dataseeds]
end

balanced_dataset_slice(n_label_samples::NTuple{N,Integer}, dataseed::Union{Nothing,Integer}; samples_per_class_share::Union{Nothing,AbstractFloat} = nothing) where N = begin
	n_instances = sum(n_label_samples)
	if !isnothing(dataseed)
		rng = Random.MersenneTwister(dataseed)
		
		n_classes = length(n_label_samples)

		n_per_class = if isnothing(samples_per_class_share)
			minimum(n_label_samples)
		else
			floor(Int, (samples_per_class_share*n_instances)/n_classes)
		end

		dataset_slice = Array{Int64,2}(undef, n_classes, n_per_class)
		c = 0
		for i in 1:n_classes
			# dataset_slice[i,:] .= sort(c .+ Random.randperm(rng, n_label_samples[i])[1:n_per_class])
			dataset_slice[i,:] .= c .+ Random.randperm(rng, n_label_samples[i])[1:n_per_class]
			c += n_label_samples[i]
		end
		dataset_slice = dataset_slice[:]
	else
		1:n_instances
	end
end

using JuMP
using Clp
# using GLPK
# using CPLEX, MathOptInterfaceCPLEX
# using MathOptInterface

function solve_binary_sampling_prob(split_threshold, an, ap, tn, tp; discourage_only_training = true)

	function gen_binary_sampling_prob()

		"""
		# https://online-optimizer.appspot.com/?model=ms:fATuYfQ2gKPO8uPEf1v5UTq62WibExAI
		var alpha_n >= 0;
		var alpha_p >= 0;
		var beta_n >= 0;
		var beta_p >= 0;
		var gamma_n >= 0;
		var gamma_p >= 0;

		#an = 54;
		#ap = 32;
		#tn = 0;
		#tp = 119;
		#split_threshold = 0.8;

		maximize z: 54*alpha_n + 32*alpha_p + 54*beta_n + 32*beta_p + 0*gamma_n + 119*gamma_p;

		subject to c10:  alpha_n >= 0;
		subject to c20:  alpha_p >= 0;
		subject to c30:  beta_n  >= 0;
		subject to c40:  beta_p  >= 0;
		subject to c50:  gamma_n >= 0;
		subject to c60:  gamma_p >= 0;

		subject to c1:   alpha_n <= 1;
		subject to c2:   alpha_p <= 1;
		subject to c3:   beta_n  <= 1;
		subject to c4:   beta_p  <= 1;
		subject to c5:   gamma_n <= 1;
		subject to c6:   gamma_p <= 1;

		subject to calphanbetan:   alpha_n + beta_n <= 1;
		subject to calphapbetap:   alpha_p + beta_p <= 1;

		subject to cN2P2:   54*alpha_n = 32*alpha_p;
		subject to cN1P1:   54*beta_n + 0*gamma_n = 32*beta_p + 119*gamma_p;

		subject to traintest_ratio_n:   54*beta_n + 0*gamma_n   = 0.8 * ((54*beta_n + 0*gamma_n) + (54*alpha_n));
		subject to traintest_ratio_p:   32*beta_p + 119*gamma_p = 0.8 * ((32*beta_p + 119*gamma_p) + (32*alpha_p));

		end;
		"""

		model = Model(Clp.Optimizer)
		set_optimizer_attribute(model, "LogLevel", 1)
		set_optimizer_attribute(model, "Algorithm", 4)

		# model = Model(CPLEX.Optimizer)
		# model = Model(GLPK.Optimizer)
		# model = Model(GLPK.Optimizer, bridge_constraints = false)

		@variable(model, 0 <= alpha_n  <= 1)
		@variable(model, 0 <= alpha_p  <= 1)
		@variable(model, 0 <= beta_n   <= 1)
		@variable(model, 0 <= beta_p   <= 1)
		@variable(model, 0 <= gamma_n  <= 1)
		@variable(model, 0 <= gamma_p  <= 1)

		@objective(model, Max, an*alpha_n + ap*alpha_p + an*beta_n + ap*beta_p + tn*gamma_n + tp*gamma_p)

		# Non-intersecting sets
		@constraint(model, calphanbetan,   alpha_n + beta_n <= 1)
		@constraint(model, calphapbetap,   alpha_p + beta_p <= 1)

		# Data balance
		@constraint(model, cN2P2,   an*alpha_n             == ap*alpha_p)
		@constraint(model, cN1P1,   an*beta_n + tn*gamma_n == ap*beta_p + tp*gamma_p)

		# Honor split_threshold as train/test ration
		@constraint(model, traintest_ratio_n,   an*beta_n + tn*gamma_n == split_threshold * ((an*beta_n + tn*gamma_n) + (an*alpha_n)))
		@constraint(model, traintest_ratio_p,   ap*beta_p + tp*gamma_p == split_threshold * ((ap*beta_p + tp*gamma_p) + (ap*alpha_p)))

		# # Only use only_training instances if needed
		# if discourage_only_training
		# 	@constraint(model, discourage_only_training_n,   ...)
		# 	@constraint(model, discourage_only_training_p,   ...)
		# end

		model, alpha_n, alpha_p, beta_n, beta_p, gamma_n, gamma_p
	end

	local model, alpha_n, alpha_p, beta_n, beta_p, gamma_n, gamma_p = gen_binary_sampling_prob()

	# print(model)
	optimize!(model)
	# @show termination_status(model)
	# @show primal_status(model)
	# @show dual_status(model)
	# @show objective_value(model)
	# @show value(alpha_n)
	# @show value(alpha_p)
	# @show value(beta_n)
	# @show value(beta_p)
	# @show value(gamma_n)
	# @show value(gamma_p)
	# @show shadow_price(calphanbetan)
	# @show shadow_price(calphapbetap)
	# @show shadow_price(cN2P2)
	# @show shadow_price(cN1P1)
	# @show shadow_price(traintest_ratio_n)
	# @show shadow_price(traintest_ratio_p)

	# @assert termination_status(model) == MathOptInterface.OPTIMAL "TODO Expand code!"
	# @assert primal_status(model)      == MathOptInterface.FEASIBLE_POINT "TODO Expand code!"

	# println()
	# println(objective_value(model))
	# println(round(Int, objective_value(model)))
	# @assert isinteger(objective_value(model)) "TODO Expand code!"

	N2_from_a = value(alpha_n) * an
	P2_from_a = value(alpha_p) * ap
	N1_from_a = value(beta_n)  * an
	N1_from_t = value(gamma_n) * tn
	P1_from_a = value(beta_p)  * ap
	P1_from_t = value(gamma_p) * tp
	tot = round(Int, objective_value(model))

	# N2_from_a, P2_from_a, N1_from_a, N1_from_t, P1_from_a, P1_from_t, tot = 1, 1, 1, 1, 1, 1, 6

	N2_from_a, P2_from_a, N1_from_a, N1_from_t, P1_from_a, P1_from_t, tot

end

balanced_dataset_slice(dataset::NamedTuple{(:train_n_test,:only_training)}, dataseeds::Vector{<:Integer}, split_threshold::AbstractFloat; samples_per_class_share::Union{Nothing,AbstractFloat} = nothing, discourage_only_training = true) where N = begin
	
	@assert length(dataset.train_n_test[2])  == 2 "TODO Expand code. Currently, can't perform balanced_dataset_slice(NamedTuple{(:train_n_test,:only_training)}, ...) on non-binary dataset"
	@assert length(dataset.only_training[2]) == 2 "TODO Expand code. Currently, can't perform balanced_dataset_slice(NamedTuple{(:train_n_test,:only_training)}, ...) on non-binary dataset"

	@assert isnothing(samples_per_class_share) "TODO Expand code"

	@assert split_threshold != 1.0 "TODO expand code. Don't quite know how to perform full training here?"
	
	# ap, an = 54, 32;
	# tp, tn = 0, 119;
	# split_threshold = 0.8;

	ap, an = dataset.train_n_test[2]
	tp, tn = dataset.only_training[2]

	N2_from_a, P2_from_a, N1_from_a, N1_from_t, P1_from_a, P1_from_t, tot = solve_binary_sampling_prob(split_threshold, an, ap, tn, tp; discourage_only_training = discourage_only_training)

	N2_from_a = round(Int, N2_from_a)
	P2_from_a = round(Int, P2_from_a)
	N1_from_a = round(Int, N1_from_a)
	N1_from_t = round(Int, N1_from_t)
	P1_from_a = round(Int, P1_from_a)
	P1_from_t = round(Int, P1_from_t)

	@assert N2_from_a + N1_from_a <= an
	@assert P2_from_a + P1_from_a <= ap
	@assert N1_from_t             <= tn
	@assert P1_from_t             <= tp

	# println()
	# println(N2_from_a)
	# println(P2_from_a)
	# println(N1_from_a)
	# println(N1_from_t)
	# println(P1_from_a)
	# println(P1_from_t)

	if discourage_only_training

		# Result:
		# train neg = 103 from only_training + 0 from all = 103 
		# train pos = 0 from only_training + 103 from all = 103 
		# test neg = (0 from only_training +) 26 from all = 26 
		# test pos = (0 from only_training +) 26 from all = 26 

		# This ensures that, during training, we use at least all of the non-only_training instances
		function __relax(from_a, from_t, avail_a)
			min(from_a+from_t, avail_a), from_a+from_t-min(from_a+from_t, avail_a)
		end

		N1_from_a, N1_from_t = __relax(N1_from_a, N1_from_t, an-N2_from_a)
		P1_from_a, P1_from_t = __relax(P1_from_a, P1_from_t, ap-P2_from_a)
	end

	N1 = N1_from_a + N1_from_t
	P1 = P1_from_a + P1_from_t
	N2 = N2_from_a
	P2 = P2_from_a

	@assert N2 == P2 "Splitting costraint failed: N2 != P2, $(N2) != $(P2)"
	@assert N1 == P1 "Splitting costraint failed: N1 != P1, $(N1) != $(P1)"
	@assert (N1/(N1+N2))-split_threshold <= 0.02 "TODO expand code (constraint on split_threshold can be softened). Splitting costraint failed: N1/(N1+N2) != split_threshold, $(N1)/($(N1)+$(N2))=$(N1/(N1+N2)) != $(split_threshold)"
	@assert (P1/(P1+P2))-split_threshold <= 0.02 "TODO expand code (constraint on split_threshold can be softened). Splitting costraint failed: P1/(P1+P2) != split_threshold, $(P1)/($(P1)+$(P2))=$(P1/(P1+P2)) != $(split_threshold)"
	
	println()
	println("Result:")
	println("- train neg = $(N1_from_t) from only_training + $(N1_from_a) from all = $(N1) ")
	println("- train pos = $(P1_from_t) from only_training + $(P1_from_a) from all = $(P1) ")
	println("- test neg = (0 from only_training +) $(N2_from_a) from all = $(N2) ")
	println("- test pos = (0 from only_training +) $(P2_from_a) from all = $(P2) ")
	println()

	# println()
	# println(N2)
	# println(P2)
	# println(N1)
	# println(P1)

	# println()
	# println(N2+P2+N1+P1)
	# println(tot)

	# Linearizza dataset
	# n_label_samples = (dataset.train_n_test[2] .+ dataset.only_training[2])
	# d = concat_labeled_datasets(dataset.train_n_test[1], dataset.only_training[1])
	# linearized_dataset = (d, n_label_samples)

	linearized_dataset = concat_labeled_datasets(dataset.train_n_test[1], dataset.only_training[1])

	# println(map(size, linearized_dataset[1][1]))

	dataset_slices = AbstractVector{<:Integer}[]

	# Derive balanced slices
	println()
	println("Generating balanced slices...")
	for dataseed in dataseeds
		rng = Random.MersenneTwister(dataseed)

		c = 0
		
		P1P2_from_a_perm = c .+ Random.randperm(rng, ap)[1:(P2_from_a+P1_from_a)]
		@assert length(P1P2_from_a_perm) == (P2_from_a+P1_from_a)
		c += ap
		P2_from_a_perm = P1P2_from_a_perm[1:P2_from_a]
		P1_from_a_perm = P1P2_from_a_perm[(P2_from_a+1):(P2_from_a+P1_from_a)]

		N1N2_from_a_perm = c .+ Random.randperm(rng, an)[1:(N2_from_a+N1_from_a)]
		@assert length(N1N2_from_a_perm) == (N2_from_a+N1_from_a)
		c += an
		N2_from_a_perm = N1N2_from_a_perm[1:N2_from_a]
		N1_from_a_perm = N1N2_from_a_perm[(N2_from_a+1):(N2_from_a+N1_from_a)]
		
		P1_from_t_perm = c .+ Random.randperm(rng, tp)[1:P1_from_t]
		@assert length(P1_from_t_perm) == P1_from_t
		c += tp

		N1_from_t_perm = c .+ Random.randperm(rng, tn)[1:N1_from_t]
		@assert length(N1_from_t_perm) == N1_from_t
		c += tn
		
		N1_idx = vcat(N1_from_t_perm, N1_from_a_perm)
		P1_idx = vcat(P1_from_t_perm, P1_from_a_perm)
		N2_idx = N2_from_a_perm
		P2_idx = P2_from_a_perm

		println()
		println("dataseed: [$(dataseed)]")
		println("train neg = $(length(N1_from_t_perm)) + $(length(N1_from_a_perm)) = $(length(N1_idx)) ", N1_idx)
		println("train pos = $(length(P1_from_t_perm)) + $(length(P1_from_a_perm)) = $(length(P1_idx)) ", P1_idx)
		println("test neg = ", length(N2_idx), " ", N2_idx)
		println("test pos = ", length(P2_idx), " ", P2_idx)
		
		# dataset_slice = vcat(N1_idx, P1_idx, N2_idx, P2_idx)
		# stratify
		dataset_slice = vcat(vec(hcat(N1_idx, P1_idx)'), vec(hcat(N2_idx, P2_idx)'))

		@assert length(dataset_slice) == N2+P2+N1+P1 "length(dataset_slice) == N2+P2+N1+P1: $(length(dataset_slice)) == $(N2)+$(P2)+$(N1)+$(P1) = $(N2+P2+N1+P1)"
		@assert length(dataset_slice) == length(unique(dataset_slice)) "length(dataset_slice) == length(unique(dataset_slice)): $length(dataset_slice)) == $(length(unique(dataset_slice)))"
		n_classes = 2
		
		@assert isinteger(length(dataset_slice)/n_classes) "$(length(dataset_slice))/$(n_classes) = $(length(dataset_slice)/n_classes)"
		# TODO change and make this check smarter. Here, need to check that traintestsplit will split exactly at the specified point @assert isinteger(length(dataset_slice)*split_threshold) "$(length(dataset_slice))*$(split_threshold) = $(length(dataset_slice)*split_threshold)"

		println(length(dataset_slice), " ", dataset_slice)
		push!(dataset_slices, dataset_slice)

	end
	linearized_dataset, dataset_slices
end

# Multi-frame dataset with labels and instance-ids
function concat_labeled_datasets((X1, Y1, f1)::Tuple{AbstractVector{<:GenericDataset},AbstractVector,AbstractVector{<:AbstractVector}}, (X2, Y2, f2)::Tuple{AbstractVector{<:GenericDataset},AbstractVector,AbstractVector{<:AbstractVector}})
	X = concat_datasets(X1, X2)
	Y = vcat(Y1, Y2)
	f = map(x->Iterators.flatten(x)|>collect, zip(f1,f2))
	# println(size.(X1))
	# println(size(Y1))
	# println(size.(f1))
	# println(size.(X2))
	# println(size(Y2))
	# println(size.(f2))
	(X, Y, f)
end

# Single-frame dataset with labels
concat_labeled_datasets((X1, Y1)::Tuple{GenericDataset,AbstractVector}, (X2, Y2)::Tuple{GenericDataset,AbstractVector}) = 
	concat_labeled_datasets(([X1], Y1), ([X2], Y2))

# Multi-frame dataset with labels
function concat_labeled_datasets((X1, Y1)::Tuple{AbstractVector{<:GenericDataset},AbstractVector}, (X2, Y2)::Tuple{AbstractVector{<:GenericDataset},AbstractVector})
	X = concat_datasets(X1, X2)
	Y = vcat(Y1, Y2)
	(X, Y)
end

# Multi-frame dataset with labels
function concat_labeled_datasets((X1, Y1)::Tuple{AbstractVector{U},AbstractVector{T}}, (X2, Y2)::Tuple{AbstractVector{U},AbstractVector{T}}, (class_counts1, class_counts2)::Tuple{NTuple{N,Integer}, NTuple{N,Integer}}) where {T, U<:GenericDataset, N}
	X = U[slice_dataset(X1_frame, Integer[]) for X1_frame in X1]
	Y = T[]
	count1, count2 = 1, 1
	for (cur_class_count1, cur_class_count2) in zip(class_counts1, class_counts2)
		cur_X1  = slice_multiframe_dataset(X1, count1:count1+cur_class_count1-1)
		cur_X2  = slice_multiframe_dataset(X2, count2:count2+cur_class_count2-1)
		cur_X12 = concat_datasets(cur_X1, cur_X2)
		X = concat_datasets(X, cur_X12)
		Y = vcat(Y, Y1[count1:count1+cur_class_count1-1], Y2[count2:count2+cur_class_count2-1])

		count1 += cur_class_count1
		count2 += cur_class_count2
	end
	(X, Y)
end

# function slice_labeled_dataset((Xs,Y)::Tuple{AbstractVector{<:GenericDataset},AbstractVector}, dataset_slice::AbstractVector)
# 	(slice_multiframe_dataset(Xs, dataset_slice), Y[dataset_slice])
# end

slice_multiframe_dataset(Xs, dataset_slice) = map(X->slice_dataset(X, dataset_slice; return_view = false), Xs)


# Multi-frame dataset
concat_datasets(X1::AbstractVector{<:GenericDataset}, X2::AbstractVector{<:GenericDataset}) = map(concat_datasets, X1, X2)

dataset_has_nonevalues(Xs::AbstractVector{<:GenericDataset}) = any([dataset_has_nonevalues(X) for X in Xs])
