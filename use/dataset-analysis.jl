using SoleBase
using SoleBase: dimension
using StatsPlots
using Plots.PlotMeasures
using SoleViz
using Measures

using Distributions

function update_ylims!(p)
	_min, _max = Inf, -Inf
	i = 1
	while true
		try p[1][i]
			_min = min(_min,minimum(p[1][i][:y]))
			_max = max(_max,maximum(p[1][i][:y]))
			i += 1
		catch Error
			break
		end
	end
	# println("updated ylims for $(i-1) series")
	tiny_margin(x) = min(x*.1,0.1)
	plot!(p, ylims = (_min-tiny_margin(_min),_max+tiny_margin(_max))); # TODO improve
end

function average_plot(plots)
	p_avg = Base.deepcopy(plots[length(plots)]);
	i = 1
	p_avg[1][i][:y] = [mean(filter(!isnan, [p_i[1][i][:y][i_attr] for p_i in plots])) for i_attr in 1:length(plots[1][1][i][:y])]
	update_ylims!(p_avg);
	p_avg[1][:title] = "mean"
	p_avg;
end

function normalize_plot(p)
	p_norm = Base.deepcopy(p);
	i = 1
	while true
		try p_norm[1][i]
			p_norm[1][i][:y] /= sum(p_norm[1][i][:y])
			i += 1
		catch Error
			break
		end
	end
	update_ylims!(p_norm);
	p_norm;
end

function compute_mfd(X)
	n_attrs = size(X,2)[end]
	n_insts = size(X)[end]
	columns = []
	for i_attr in 1:n_attrs
		push!(columns, ([X[:,i_attr,i_inst] for i_inst in 1:n_insts]))
	end
	colnames = [string(i) for i in 1:n_attrs]
	df = DataFrame(columns, colnames)
	mfd = MultiFrameDataset([1:ncol(df)], df)
end

function single_frame_blind_feature_selection((X, Y)::Tuple{AbstractArray{T,3},AbstractVector}, attribute_names::AbstractVector, grouped_descriptors, run_file_prefix, n_desired_attributes, n_desired_features; savefigs = false, descriptors_abbr = descriptors_abbr, attributes_abbr = attributes_abbr) where {T}

	@assert length(attribute_names) == size(X)[end-1] "$(length(attribute_names)) != $(size(X)[end-1])"

	_savefig = savefigs ? savefig : (x...)->(;)
	
	mfd = compute_mfd(X);
	# ClassificationMultiFrameDataset(Y, mfd)

	descriptors = collect(Iterators.flatten([values(grouped_descriptors)...]))

	println("Performing dataset pre-analysis!")
	println("n_desired_attributes = $(n_desired_attributes)")
	println("n_desired_features   = $(n_desired_features)")

	println()
	println("size(X) = $(size(X))")

	# run_name = "example"

	t = 1
	p = SoleViz.plotdescription(mfd, descriptors = descriptors, windows = [[[(t,0,0)]]])
	println(size(p))
	p = p[:]

	best_attributes_idxs = begin

		############################################################################
		############################################################################
		############################################################################
		@assert length(descriptors) == 25 "TODO change layout"
		
		_savefig(plot(p...; layout = (5,5), size = (1920, 1080), margin = 5mm), "$(run_file_prefix).png");

		p_avg = average_plot(p);
		_savefig(plot(p_avg; size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-avg.png");
		############################################################################
		############################################################################
		############################################################################
		perm = sortperm(p_avg[1][1][:y], rev=true);
		p_sorted = Base.deepcopy(p);
		for (p_i_idx,p_i) in enumerate(collect(p_sorted))
			i = 1
			while true
				try p_i[1][i]
					p_i[1][i][:y] = p_i[1][i][:y][perm]
					i += 1
				catch Error
					break
				end
			end
			if p_i_idx == length(p_sorted)
				plot!(p_i, xticks = (1:length(perm), [isnothing(attributes_abbr) ? attribute_names[t] : attributes_abbr[t] for t in perm]));
			else
				plot!(p_i, xticks = false)
			end
		end
		_savefig(plot(p_sorted...; layout = (5,5), size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-sorted.png");

		p_sorted_avg = Base.deepcopy(p_avg);
		p_sorted_avg[1][1][:y] = p_sorted_avg[1][1][:y][perm]
		plot!(p_sorted_avg, xticks = (1:length(perm), [isnothing(attributes_abbr) ? attribute_names[t] : attributes_abbr[t] for t in perm]));
		# plot!(p_sorted_avg, xticks = (1:length(perm), [isnothing(attributes_abbr) ? attribute_names[t] : attributes_abbr[t] for t in perm]));
		_savefig(plot(p_sorted_avg; size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-sorted-avg.png");
		############################################################################
		############################################################################
		############################################################################
		p_norm = normalize_plot.(p);
		_savefig(plot(p_norm...; layout = (5,5), size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-norm.png");

		p_norm_avg = average_plot(p_norm);
		_savefig(plot(p_norm_avg; size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-norm-avg.png");
		############################################################################
		############################################################################
		############################################################################
		perm = sortperm(p_norm_avg[1][1][:y], rev=true);
		p_sorted = Base.deepcopy(p_norm);
		for (p_i_idx,p_i) in enumerate(collect(p_sorted))
			i = 1
			while true
				try p_i[1][i]
					p_i[1][i][:y] = p_i[1][i][:y][perm]
					i += 1
				catch Error
					break
				end
			end
			if p_i_idx == length(p_sorted)
				plot!(p_i, xticks = (1:length(perm), [isnothing(attributes_abbr) ? attribute_names[t] : attributes_abbr[t] for t in perm]));
			else
				plot!(p_i, xticks = false)
			end
		end
		_savefig(plot(p_sorted...; layout = (5,5), size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-norm-sorted.png");

		p_sorted_avg = Base.deepcopy(p_norm_avg);
		for i in 1:length(p_sorted_avg[1])
			p_sorted_avg[1][i][:y] = p_sorted_avg[1][i][:y][perm]
		end
		plot!(p_sorted_avg, xticks = (1:length(perm), [isnothing(attributes_abbr) ? attribute_names[t] : attributes_abbr[t] for t in perm]));
		# plot!(p_sorted_avg, xticks = (1:length(perm), [isnothing(attributes_abbr) ? attribute_names[t] : attributes_abbr[t] for t in perm]));
		_savefig(plot(p_sorted_avg; size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-norm-sorted-avg.png");
		############################################################################
		############################################################################
		############################################################################

		# display(p_norm_avg)

		p = SoleViz.plotdescription(mfd, descriptors = grouped_descriptors, windows = [[[(t,0,0)]]])
		# println(size(p))
		p = p[:]

		############################################################################
		############################################################################
		############################################################################
		@assert length(grouped_descriptors) == 8 "TODO change layout"


		for (p_i_idx,p_i) in enumerate(collect(p))
			plot!(p_i, xticks = false)
		end

		_savefig(plot(p...; layout = (4,2), size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-grouped.png");

		############################################################################
		############################################################################
		############################################################################
		p_sorted = Base.deepcopy(p);
		for (p_i_idx,p_i) in enumerate(collect(p_sorted))
			i = 1
			while true
				try p_i[1][i]
					p_i[1][i][:y] = p_i[1][i][:y][perm]
					i += 1
				catch Error
					break
				end
			end
			if (p_i_idx % 4) == 0
				plot!(p_i, xticks = (1:length(perm), [isnothing(attributes_abbr) ? attribute_names[t] : attributes_abbr[t] for t in perm]));
			else
				plot!(p_i, xticks = false)
			end
		end
		_savefig(plot(p_sorted...; layout = (4,2), size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-grouped-sorted.png");

		############################################################################
		############################################################################
		############################################################################
		p_norm = normalize_plot.(p);
		_savefig(plot(p_norm...; layout = (4,2), size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-grouped-norm.png");

		############################################################################
		############################################################################
		############################################################################
		perm = sortperm(p_norm_avg[1][1][:y], rev=true)
		p_sorted = Base.deepcopy(p_norm);
		for (p_i_idx,p_i) in enumerate(collect(p_sorted))
			i = 1
			while true
				try p_i[1][i]
					p_i[1][i][:y] = p_i[1][i][:y][perm]
					i += 1
				catch Error
					break
				end
			end
			if (p_i_idx % 4) == 0
				plot!(p_i, xticks = (1:length(perm), [isnothing(attributes_abbr) ? attribute_names[t] : attributes_abbr[t] for t in perm]));
			else
				plot!(p_i, xticks = false)
			end
		end
		_savefig(plot(p_sorted...; layout = (4,2), size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-grouped-norm-sorted.png");

		############################################################################
		############################################################################
		############################################################################
		perm[1:n_desired_attributes]
	end

	X_sub = X[:,best_attributes_idxs,:]
	Y_sub = Y[:]

	println("Selecting $(length(best_attributes_idxs)) attributes: $(best_attributes_idxs)...")
	println(["$(t): $(attribute_names[t])" for t in best_attributes_idxs])

	mfd_sub = compute_mfd(X_sub);

	p = SoleViz.plotdescription(mfd_sub, descriptors = descriptors, windows = [[[(t,0,0)]]], on_x_axis = :descriptors)
	println(size(p))
	p = p[:]

	best_descriptors = begin

		############################################################################
		############################################################################
		############################################################################
		
		for i in 1:(length(p)-1)
			plot!(p[i]; xticks = false)
		end

		_savefig(plot(p...; layout = (5,1), size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-transposed.png");

		p_avg = average_plot(p);
		_savefig(plot(p_avg; size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-transposed-avg.png");
		############################################################################
		############################################################################
		############################################################################
		perm = sortperm(p_avg[1][1][:y], rev=true)
		p_sorted = Base.deepcopy(p);
		for (p_i_idx,p_i) in enumerate(collect(p_sorted))
			i = 1
			while true
				try p_i[1][i]
					p_i[1][i][:y] = p_i[1][i][:y][perm]
					i += 1
				catch Error
					break
				end
			end
			if (p_i_idx % 4) == 0
				plot!(p_i, xticks = (1:length(perm), [isnothing(descriptors_abbr) ? string(descriptors[t]) : descriptors_abbr[descriptors[t]] for t in perm]));
			else
				plot!(p_i, xticks = false)
			end
		end
		_savefig(plot(p_sorted...; layout = (5,1), size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-transposed-sorted.png");

		############################################################################
		############################################################################
		############################################################################
		p_norm = normalize_plot.(p);
		_savefig(plot(p_norm...; layout = (5,1), size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-transposed-norm.png");

		p_norm_avg = average_plot(p_norm);
		_savefig(plot(p_norm_avg; size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-transposed-norm-avg.png");
		############################################################################
		############################################################################
		############################################################################
		perm = sortperm(p_norm_avg[1][1][:y], rev=true)
		p_sorted = Base.deepcopy(p_norm);
		for (p_i_idx,p_i) in enumerate(collect(p_sorted))
			i = 1
			while true
				try p_i[1][i]
					p_i[1][i][:y] = p_i[1][i][:y][perm]
					i += 1
				catch Error
					break
				end
			end
			if (p_i_idx % 4) == 0
				plot!(p_i, xticks = (1:length(perm), [isnothing(descriptors_abbr) ? string(descriptors[t]) : descriptors_abbr[descriptors[t]] for t in perm]));
			else
				plot!(p_i, xticks = false)
			end
		end
		_savefig(plot(p_sorted...; layout = (5,1), size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-transposed-norm-sorted.png");

		p_sorted_avg = Base.deepcopy(p_norm_avg);
		for i in 1:length(p_sorted_avg[1])
			p_sorted_avg[1][i][:y] = p_sorted_avg[1][i][:y][perm]
		end
		plot!(p_sorted_avg, xticks = (1:length(perm), [isnothing(descriptors_abbr) ? string(descriptors[t]) : descriptors_abbr[descriptors[t]] for t in perm]));
		# plot!(p_sorted_avg, xticks = (1:length(perm), [isnothing(descriptors_abbr) ? string(descriptors[t]) : descriptors_abbr[descriptors[t]] for t in perm]));
		_savefig(plot(p_sorted_avg; size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-transposed-norm-sorted-avg.png");
		############################################################################
		############################################################################
		############################################################################

		descriptors[perm[1:n_desired_features]]
	end
	
	println("Selecting $(length(best_descriptors)) descriptors: $(best_descriptors)...")
	println()

	best_attributes_idxs, best_descriptors
end

# descriptors = best_descriptors
# single_frame_target_aware_analysis((X, Y), attribute_names, descriptors, run_file_prefix; savefigs = false, descriptors_abbr = descriptors_abbr, attributes_abbr = attributes_abbr)

function single_frame_target_aware_analysis((X, Y)::Tuple{AbstractArray{T,3},AbstractVector{<:String}}, attribute_names::AbstractVector, descriptors, run_file_prefix; savefigs = false, descriptors_abbr = descriptors_abbr, attributes_abbr = attributes_abbr, plot_normals = false) where {T}
	
	@assert length(attribute_names) == size(X)[end-1] "$(length(attribute_names)) != $(size(X)[end-1])"

	mfd = compute_mfd(X);

	_savefig = savefigs ? savefig : (x...)->(;)

	win = 1

	descriptions = SoleBase.describe(mfd; desc = descriptors, t = [[(win,0,0)]])[1]

	class_names = unique(Y)

	idxs_per_class = [Y .== class_name for class_name in class_names]
	
	values = Array{NTuple{length(class_names)}}(undef, length(descriptors), length(attribute_names))

	for (i_descriptor,descriptor) in enumerate(descriptors)
		# println("Analysing $(descriptor)...")
		description = descriptions[:,descriptor]
		for (i_attribute,(attribute_name,attr_description)) in enumerate(zip(attribute_names,description))
			values[i_descriptor,i_attribute] = Tuple([attr_description[class_idxs] for class_idxs in idxs_per_class])
		end
	end

	plots = []
	for (i_attribute,attribute_name) in enumerate(attribute_names)
		for (i_descriptor,descriptor) in enumerate(descriptors)
			subp = plot() # title = string(descriptor))

			# class_is_normal = []

			# histogram!(subp, values[i_descriptor,i_attribute]...)
			for (v, class_name) in zip(values[i_descriptor,i_attribute], class_names)

				# histogram!(subp, v)
				density!(subp, v, legend = false)
				if plot_normals
					plot!(subp, fit(Normal, v)) #, fill=(0, .5,:orange))
				end

				# push!(class_is_normal, )

			end
			# "$(string(descriptors))($(i_attribute))"
			plots[i_descriptor, i_attribute] = subp
		end
	end

	p = plot!(plots..., size = (1920, 1080), margin = 5mm, showaxis = false, grid = false, labels = class_names)
	
	# p = plot!(p..., title = "$(attribute_name)", size = (1920, 1080), margin = 5mm)
	# display(p)
	# _savefig(plot!(p..., size = (1920, 1080), margin = 5mm), "$(run_file_prefix)-$(attribute_name).png");
	# readline()

	_savefig(p, "$(run_file_prefix)-all.png");
end
