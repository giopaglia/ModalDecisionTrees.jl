using StatsBase
using DataFrames
using CSV
using Tables

size(mf::ModalFrame) = (nrow(mf.data),)
getindex(mf::ModalFrame, i::Int) = mf.views[i]

struct ClassificationDataset2
	ldim::Int
	hdim::Int
	frames::Vector{ModalFrame}
	instances::Vector{ModalInstance}
	domains::Dict{Int, Dict{Symbol, Vector{Number}}}
	classes::CategoricalArray
end

function ClassificationDataset2(frames::Vector{ModalFrame}, classes::CategoricalArray)
	instances = ModalInstance[]
	domains = Dict{Int, Dict{Symbol, Vector{Number}}}()

	n_instances = size(frames[1])

	for i in 1:length(frames)
		curr_n_instances = size(frames[i])
		@assert n_instances == curr_n_instances "Each frame must have the same number of (intraframe) instances!"
	end

	@assert n_instances[1] == length(classes) "Each instance must have one and only one class!"

	# Computing the instances.
	# for each instance
	for i in 1:n_instances[1]
		instance = ModalInstance()
		# for each frame
		for f in 1:length(frames)
			# push the instance
			push!(instance.rows, frames[f][i])
		end
		push!(instances, instance)
	end

	# Computing the domains.
	# for each frame
	for f in 1:length(frames)
		attrs = names(frames[f][1])
		# for each attribute
		attr_domain = Dict{Symbol, Vector{Number}}()
		for attr in attrs
			# convert to symbol
			A = Symbol(attr)
			# for each instance
			# for i in 1:n_instances[1]

			# end
			attr_domain[A] = collect(Set([v for i in 1:n_instances[1] for v in frames[f][i][A]]))
			domains[f] = attr_domain
		end
	end

	ldim, hdim = extrema([dimension(i) for i in frames])

	ClassificationDataset2(ldim, hdim, frames, instances, domains, classes)
end

function ClassificationDataset22RunnerDataset(d::ClassificationDataset2)
	Y = map(string, Array(d.classes))
	X = Array{<:Number}[]
	for frame in d.frames
		push!(X, ModalFrame2MatricialDataset(frame))
	end
	(X,Y)
end

instance(cs::ClassificationDataset2, i::Int) = cs.instances[i] # TODO cs -> ds (either sup or unsupervised)
instance(cs::ClassificationDataset2, i::Int, f::Int) = cs.instances[i][f]
function attributes(ds::ClassificationDataset2)
	d = Dict{Int,Array{String,1}}()
	for (fid,frame) in enumerate(ds.frames)
		d[fid] = names(frame.data)
	end
	return d
end
attributes(ds::ClassificationDataset2, fid::Int) = names(ds.frames[fid].data)

length(ds::ClassificationDataset2) = length(ds.instances)

function show(io::IO, ::MIME"text/plain", ds::ClassificationDataset2)
	println(io, "Classification dataset with $(length(ds.instances)) instances")

	for (i, inst) in enumerate(ds.instances)
		println(io, "Instance #$(i):")
		println(io, instance(ds, i))
		println(io)
	end
end

# TODO
# possible test: all([auslan.instances[i].rows[1][attr] === auslan.frames[1].data[i,attr] for i in 1:length(auslan.instances) for attr in attributes(auslan, 1)])
function transform!(ds::ClassificationDataset2, f::Function, fid::Int; kwargs...)
	for attr in attributes(ds, fid)
		for i in 1:nrow(ds.frames[fid].data)
			ds.frames[fid].data[i,attr] = f(ds.frames[fid].data[i,attr]; kwargs...)
		end
	end
end

function minimum(ds::ClassificationDataset2)
	d = Dict{Int,Array{Float64,1}}()

	for (fid,frame) in enumerate(ds.frames)
		frame = ds.frames[fid]
		if frame.dimension == 0
			d[fid] = missing
		else
			d[fid] = [minimum(vcat(frame.data[!,attr]...)) for attr in attributes(ds, fid)]
		end
	end
	return d
end

function maximum(ds::ClassificationDataset2)
	d = Dict{Int,Array{Float64,1}}()

	for (fid,frame) in enumerate(ds.frames)
		frame = ds.frames[fid]
		if frame.dimension == 0
			d[fid] = missing
		else
			d[fid] = [maximum(vcat(frame.data[!,attr]...)) for attr in attributes(ds, fid)]
		end
	end
	return d
end

# TODO fix problem with select_attributes!
function select_attributes!(ds::ClassificationDataset2, frame_attributes::Dict{Int, Array{Int, 1}})
	frames = collect(keys(frame_attributes))

	for i in 1:length(frames)
		ds.frames[frames[i]].data = ds.frames[frames[i]].data[:, frame_attributes[frames[i]]]
	end
end

# function transform!(ds::ClassificationDataset2, f::Function; kwargs...)

# using MultivariateTimeSeries
# X, y = read_data_labeled(joinpath(dirname(pathof(GBDTs)), "..", "data", "auslan_youtube8"));
#
# # adesso voglio passare da MTS a ClassificationDataset2
#
# my_frame = DataFrame()
#
# for attr in names(X)
#     insertcols!(my_frame, attr => Array{Float64,1}[])
# end
#
# # for each instance
# for i in 1:length(X)
#     array_data = Array{Float64, 2}(X[i])
#     push!(my_frame, [array_data[:, j] for j in 1:length(names(X))])
# end
#
# auslan = ClassificationDataset2([ModalFrame(my_frame)], CategoricalArray(y))



function cd_sample(ds::ClassificationDataset2,
		range::Union{UnitRange{Int},Nothing};
		invert::Bool=false)

	n_instances = length(ds.instances)

	@assert range ⊆ 1:n_instances "The range must be a subset of 1:$(n_instances)."

	if ismissing(range)
		return ds
	else

		output_classes = Int[]
		timeseries_df = DataFrame()

		attrs = attributes(ds, 1) # TODO up to now, only timeseries

		for attr in attrs
			insertcols!(timeseries_df, attr => Array{Float64, 1}[]) # TODO also here Array{Float64, 1}[], i.e., timeseries
		end

		if invert == false
			for idx in range
				push!(timeseries_df, ds.instances[idx][1]) # ds.instances[idx][1] <-- [1] means the first frame TODO in future
				push!(output_classes, ds.classes[idx])
			end
		else
			for idx in setdiff(1:n_instances,range)
				push!(timeseries_df, ds.instances[idx][1]) # ds.instances[idx][1] <-- [1] means the first frame TODO in future
				push!(output_classes, ds.classes[idx])
			end
		end
	end

	return ClassificationDataset2([ModalFrame(timeseries_df)], CategoricalArray(output_classes))
end

function cd_sample(ds::ClassificationDataset2,
	percentage::Float64; 
	seed::Int=1,
	bias::Float64=0.0, # bias to uniform
	replace::Bool=true,
	invert::Bool=false)

	@assert 0 ≤ percentage && percentage ≤ 1 "Percentage p must be 0 ≤ p ≤ 1."

	n_instances = length(ds.instances)
	n_instances_per_class = countmap(ds.classes)
	classes = keys(n_instances_per_class)
	n_classes = length(n_instances_per_class)

	instances_per_class = Dict{CategoricalArrays.CategoricalValue{Int64, UInt32}, Array{Int, 1}}()

	for class in classes
		instances_per_class[class] = ModalInstance[]
	end

	instances_to_sample = Dict{CategoricalArrays.CategoricalValue{Int64, UInt32}, Int64}()

	for class in classes
		sample_size = Int(round(percentage * ((1-bias)*n_instances_per_class[class] + bias*n_instances / n_classes)))
		instances_to_sample[class] = sample_size
	end

	for i in 1:n_instances
		push!(instances_per_class[ds.classes[i]], i)
	end

	r = Random.seed!(seed)

	output_classes = Int[]
	timeseries_df = DataFrame()

	attrs = attributes(ds, 1) # TODO up to now, only timeseries

	for attr in attrs
		insertcols!(timeseries_df, attr => Array{Float64, 1}[]) # TODO also here Array{Float64, 1}[], i.e., timeseries
	end

	if replace
		for class in classes
			for i in 1:instances_to_sample[class]
				idx = instances_per_class[class][rand(r, 1:n_instances_per_class[class])]
				push!(timeseries_df, ds.instances[idx][1]) # ds.instances[idx][1] <-- [1] means the first frame TODO in future
				push!(output_classes, ds.classes[idx])
				# @show idx
				# @show ds.classes[idx]
			end
		end
	else
		for class in classes
			n_eligible = length(instances_per_class[class])
			selected = instances_per_class[class]

			# the selected elements are those at the end of then selected array
			for i in 1:instances_to_sample[class]
				idx = rand(r, 1:n_eligible)
				selected[idx], selected[n_eligible] = selected[n_eligible], selected[idx]
				n_eligible -= 1
			end

			# take the last instances_to_sample[class] instances
			if invert == false
				for i in n_instances_per_class[class]-instances_to_sample[class]+1:n_instances_per_class[class]
					idx = selected[i]
					push!(timeseries_df, ds.instances[idx][1]) # ds.instances[idx][1] <-- [1] means the first frame TODO in future
					push!(output_classes, ds.classes[idx])
				end
			# take the first n_instances_per_class[class]-instances_to_sample[class] instances
			else
				for i in 1:n_instances_per_class[class]-instances_to_sample[class]
					idx = selected[i]
					push!(timeseries_df, ds.instances[idx][1]) # ds.instances[idx][1] <-- [1] means the first frame TODO in future
					push!(output_classes, ds.classes[idx])
				end
			end
		end
	end

	return ClassificationDataset2([ModalFrame(timeseries_df)], CategoricalArray(output_classes))
end

# function trip_shutdown(dir::String, day::Int) # day = 4 solo 4th, day = 1 tutti
# 	df = DataFrame()
# 	classes = Int[]
# 	attributes = CSV.File(dir * "/Example_1.csv", drop = [1, 2]) |> DataFrame |> names

# 	for attr in attributes
# 		insertcols!(df, attr => Array{Float64, 1}[])
# 	end

# 	for row in CSV.File(dir * "/DataInfo.csv")
# 		if row.Day >= day
# 			aux = CSV.File(dir * "/Example_" * string(row.ExampleID) * ".csv", drop = [1, 2]) |> Tables.matrix

# 			push!(df, [aux[:, j] for j in 1:length(attributes)])

# 			push!(classes, row.Class)
# 		end
# 	end

# 	ds = ClassificationDataset2([ModalFrame(df)], CategoricalArray(classes))

# 	return ds
# end


function trip_no_trip(dir::String;
		n_machine::Union{Nothing, Int64} = nothing,
		ignore_class0 = true,
		only_consider_trip_days = true,
		mode::Symbol = :classification, # :classification, :regression
		regression_step_in_minutes = 60,
		ignore_last_minutes = 0,
		ma_size = nothing,
		ma_step = nothing,
		)
	X_df = DataFrame()
	Y = []
	attributes = CSV.File(dir * "/Example_1.csv", drop = [1, 2]) |> DataFrame |> names

	println("Attributes: $(attributes)")

	for attr in attributes
		insertcols!(X_df, attr => Array{Float64, 1}[])
	end

	for row in CSV.File(dir * "/DataInfo.csv")
		if ignore_class0 && row.Class == 0
			continue
		elseif only_consider_trip_days && ! (row.Class == 1 && row.Day == 4)
			continue
		else
			if !isnothing(n_machine) && row.Datasource != n_machine
				continue
			end

			aux = CSV.File(dir * "/Example_$(row.ExampleID).csv", drop = [1, 2]) |> Tables.matrix

			# println(size(aux))

			if mode == :classification
				ts = [moving_average(aux[:, j], ma_size, ma_step) for j in 1:length(attributes)]
				push!(X_df, ts)
				push!(Y, (row.Day < 4 ? "no-trip" : "trip"))
			elseif mode == :regression
				for minute in 1:regression_step_in_minutes:((1440-1)-ignore_last_minutes)
					# println("Window: [$(minute):$(minute+regression_step_in_minutes-1)]")
					ts = [
						moving_average(aux[minute:(minute+regression_step_in_minutes-1), j], ma_size, ma_step)
						for j in 1:length(attributes)]
					push!(X_df, ts)
					push!(Y, (1440-(minute+regression_step_in_minutes-1))/60)
				end
			else
				error("Unknown mode: $(mode) (type = $(typeof(mode)))")
			end
		end
	end
	
	ids  = sortperm(Y)
	X_df = X_df[ids,:]
	Y    = Y[ids]

	# println(X_df)
	# println(Y)

	if mode == :classification
		classes = unique(Y)
		cm = StatsBase.countmap(Y)
		class_counts = Tuple([cm[y] for y in unique(Y)])

		println(classes)
		println(cm)
		println(class_counts)
		println()

		ds = ClassificationDataset2([ModalFrame(X_df)], CategoricalArray(Y))
		ds, class_counts
	elseif mode == :regression
		n_instances = nrow(X_df)
		n_attrs = ncol(X_df)
		channel_length = ((length.(X_df) |> unique)[1,:] |> unique)[1]
		X = Array{Float64,3}(undef, channel_length, n_attrs, n_instances)
		for (i_inst, instance) in enumerate(eachrow(X_df))
			for (i_attr, channel) in enumerate(instance)
				X[:, i_attr, i_inst] = channel
			end
		end
		println("size: $(size(X))")
		X, Y
	else
		error("Unknown mode: $(mode) (type = $(typeof(mode)))")
	end
end

function window_slice(x::AbstractArray{T} where T<:Real;
		from::Union{Missing,Int}=missing,
		to::Union{Missing,Int}=missing)
		ismissing(from) && ismissing(to) && return x
		ismissing(to) && return x[from:end]
		ismissing(from) && return x[1:to]
		return x[from:to]
end

function paa(x::AbstractArray{T} where T <: Real;
		n_chunks::Union{Missing,Int}=missing,
		f::Function=mean,
		kwargs...)
	if ismissing(n_chunks) return x
	else
		N = length(x)
		@assert 0 ≤ n_chunks && n_chunks ≤ N "The number of chunks must be in [0,$(N)]"

		y = Array{Float64}(undef, n_chunks) # TODO Float64?
		for i in 1:n_chunks
			l = Int(ceil((N*(i-1)/n_chunks) + 1))
			h = Int(ceil(N*i/n_chunks))
			y[i] = f(x[l:h]; kwargs...)
		end
		return y
	end
end

function SiemensJuneDataset_not_stratified(from, to)
	# Qua fai un nuovo dataset, cambiando il dir, con trip vs no-trip
	ds, class_counts = trip_no_trip(data_dir * "Data_Features"); 
	# Qua si fa una trasformazione del dataset usando la window che va dal punto 1 al punto 120, i.e., le prime due ore.
	transform!(ds, window_slice, 1; from=from, to=to)
	
	ClassificationDataset22RunnerDataset(ds), class_counts
end

function SiemensDataset_regression(datadirname; binning, kwargs...)
	X, Y = trip_no_trip(data_dir * datadirname; mode = :regression, kwargs...);
	function manual_bin(y)
		for (threshold,label) in binning
			y <= threshold && return label
		end
		error("Error! element with label $(y) falls outside binning $(binning)")
	end
	Y = manual_bin.(Y)
	classes = unique(Y)
	cm = StatsBase.countmap(Y)
	class_counts = Tuple([cm[y] for y in unique(Y)])
	(X, Y), class_counts
end
