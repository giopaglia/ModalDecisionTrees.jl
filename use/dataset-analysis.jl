using SoleBase
using SoleBase: dimension
using StatsPlots
using Plots.PlotMeasures
using SoleViz
using Measures
using SimpleCaching
SimpleCaching.settings.log = true

using Distributions

function show_vector_sans_type(v::AbstractVector)
  out_str = "["
  for (i, elt) in enumerate(v)
    i > 1 && (out_str *= ", ")
    out_str *= "$(elt)"
  end
  out_str *= "]"
  out_str
end

function latex_df_push_column!(df, col_name, val)
	df[!,replace(string(col_name), "_" => "")] = val
end


# sort but leave NaN last, or ignore them
function _sortperm(a; nan_mode = :last, kwargs...)
	___perm = sortperm(a; kwargs...)
	# println(p)
	# println(a[p])
	if nan_mode == :ignore
		filter((idx)->!isnan(a[idx]), ___perm)
	elseif nan_mode == :last
		[filter((idx)->!isnan(a[idx]), ___perm)..., filter((idx)->isnan(a[idx]), ___perm)...]
	else
		error("Unknown nan_mode: $(nan_mode)")
	end
end

function update_ylims!(p)
	_min, _max = Inf, -Inf
	i = 1
	while true
		try p[1][i]
			_min = min(_min,minimum(filter(!isnan, p[1][i][:y])))
			_max = max(_max,maximum(filter(!isnan, p[1][i][:y])))
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
	try p_avg[1][2]
		error("Can't compute average plot with more than one series!")
	catch Error
	end
	p_avg[1][i][:y] = [mean(filter(!isnan, [p_i[1][i][:y][x] for p_i in plots])) for x in 1:length(plots[length(plots)][1][i][:y])]
	update_ylims!(p_avg);
	p_avg[1][i][:label] = ""
	p_avg[1][:title] = "mean"
	p_avg
end

function normalize_plot(p)
	p_norm = Base.deepcopy(p);
	i = 1
	while true
		try p_norm[1][i]
			_s = sum(filter(!isnan, p_norm[1][i][:y]))
			if _s != 0
				p_norm[1][i][:y] /= _s
			end
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

# TODO use PGFPlotsX?
function single_frame_blind_feature_selection(
		(X, Y)::Tuple{AbstractArray{T,3},AbstractVector},
		attribute_names::AbstractVector,
		grouped_descriptors,
		file_prefix::AbstractString,
		n_desired_attributes::Integer,
		n_desired_features::Integer;
		savefigs::Bool = false,
		descriptor_abbrs = nothing,
		attribute_abbrs = nothing,
		export_csv::Bool = false,
		precomputed_description::Union{<:AbstractVector{<:AbstractDataFrame},Nothing} = nothing,
		cache_description_in::Union{<:AbstractString,Nothing} = nothing,
		join_plots::AbstractVector{Symbol} = Symbol[], # [:attributes], # array of :attributes, :descriptors
	) where {T}

	@assert length(attribute_names) == size(X)[end-1] "$(length(attribute_names)) != $(size(X)[end-1])"

	@assert !(export_csv && !isempty(join_plots)) "export_csv requires empty join_plots"

	_savefig = savefigs ? (plot, plot_name)->begin
			savefig(plot, "$(file_prefix)$((plot_name == "" ? "" : "-$(plot_name)")).png")
		end : (x...)->(;)

	mfd = compute_mfd(X);
	# ClassificationMultiFrameDataset(Y, mfd)

	descriptors = collect(Iterators.flatten([values(grouped_descriptors)...]))

	println("Performing dataset pre-analysis!")
	println("n_desired_attributes = $(n_desired_attributes)")
	println("n_desired_features   = $(n_desired_features)")

	println()
	println("size(X) = $(size(X))")

	# file_prefix = "example"

	best_attributes_idxs = begin

		if export_csv
			df_xattributes = DataFrame()
			latex_df_push_column!(df_xattributes, :index, collect(0:(length(attribute_names)-1)))
			# latex_df_push_column!(df_xattributes, :attribute_names, attribute_names)
			# latex_df_push_column!(df_xattributes, :attribute_abbrs, begin
			# 		if isnothing(attribute_abbrs)
			# 			["A$i" for i in 1:length(attribute_names)]
			# 		else
			# 			attribute_abbrs
			# 		end
			# 	end
			# )
		end

		t = 1
		descriptions_args = (mfd,)
		descriptions_kwargs = (desc = descriptors, t = [[(t,0,0)]])

		descriptions =
			if !isnothing(precomputed_description)
				precomputed_description
			elseif !isnothing(cache_description_in)
				@cachefast "plotdescription" joinpath(cache_description_in, "plotdescription") descriptions_args descriptions_kwargs SoleBase.describe
			else
				SoleBase.describe(descriptions_args...; descriptions_kwargs...)
			end

		p = SoleViz.plotdescription(
			[descriptions],
			group_descriptors = descriptors,
			attribute_names = attribute_names,
			cache_stats = !isnothing(cache_description_in) ? joinpath(cache_description_in, "stats") : nothing,
			join_plots = :descriptors in join_plots,
		)
		p = p[:]

		############################################################################
		############################################################################
		############################################################################
		@assert length(descriptors) == 25 "TODO change layout"

		_savefig(plot(p...; layout = (5,5), size = (1920, 1080), margin = 5mm), "main");

		p_avg = average_plot(p);
		_savefig(plot(p_avg; size = (1920, 1080), margin = 5mm), "avg");
		############################################################################
		############################################################################
		############################################################################
		perm = _sortperm(p_avg[1][1][:y], rev=true, nan_mode = :last);
		# perm = sortperm(p_avg[1][1][:y], rev=true);
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
				plot!(p_i, xticks = (1:length(perm), [isnothing(attribute_abbrs) ? attribute_names[t] : attribute_abbrs[t] for t in perm]));
			else
				plot!(p_i, xticks = false)
			end
		end
		_savefig(plot(p_sorted...; layout = (5,5), size = (1920, 1080), margin = 5mm), "sorted");

		p_sorted_avg = Base.deepcopy(p_avg);
		p_sorted_avg[1][1][:y] = p_sorted_avg[1][1][:y][perm]
		plot!(p_sorted_avg, xticks = (1:length(perm), [isnothing(attribute_abbrs) ? attribute_names[t] : attribute_abbrs[t] for t in perm]));
		# plot!(p_sorted_avg, xticks = (1:length(perm), [isnothing(attribute_abbrs) ? attribute_names[t] : attribute_abbrs[t] for t in perm]));
		_savefig(plot(p_sorted_avg; size = (1920, 1080), margin = 5mm), "sorted-avg");
		############################################################################
		############################################################################
		############################################################################
		p_norm = normalize_plot.(p);
		_savefig(plot(p_norm...; layout = (5,5), size = (1920, 1080), margin = 5mm), "norm");

		p_norm_avg = average_plot(p_norm);
		_savefig(plot(p_norm_avg; size = (1920, 1080), margin = 5mm), "norm-avg");
		############################################################################
		############################################################################
		############################################################################
		perm = _sortperm(p_norm_avg[1][1][:y], rev=true, nan_mode = :last);
		if export_csv
			_attributenames = attribute_names
			_attributeabbrs = begin
					if isnothing(attribute_abbrs)
						["A$i" for i in 1:length(attribute_names)]
					else
						attribute_abbrs
					end
				end
			latex_df_push_column!(df_xattributes, :attribute_names_norm_sorted, _attributenames[perm])
			latex_df_push_column!(df_xattributes, :attribute_abbrs_norm_sorted, _attributeabbrs[perm])
		end

		p_sorted = Base.deepcopy(p_norm);
		for (p_i_idx,p_i) in enumerate(collect(p_sorted))
			i = 1
			while true
				try p_i[1][i]
					p_i[1][i][:y] = p_i[1][i][:y][perm]
					if export_csv
						series_name = string(p_i[1][i][:label]) * "_norm_sorted"
						println("Exporting series \"$(series_name)\"...")
						latex_df_push_column!(df_xattributes, series_name, p_i[1][i][:y])
					end
					i += 1
				catch Error
					break
				end
			end
			if p_i_idx == length(p_sorted)
				plot!(p_i, xticks = (1:length(perm), [isnothing(attribute_abbrs) ? attribute_names[t] : attribute_abbrs[t] for t in perm]));
			else
				plot!(p_i, xticks = false)
			end
		end
		_savefig(plot(p_sorted...; layout = (5,5), size = (1920, 1080), margin = 5mm), "norm-sorted");

		p_sorted_avg = Base.deepcopy(p_norm_avg);
		for i in 1:length(p_sorted_avg[1])
			p_sorted_avg[1][i][:y] = p_sorted_avg[1][i][:y][perm]
			if export_csv
				series_name = string(p_sorted_avg[1][:title]) * "_norm_sorted"
				# series_name = Symbol(string(p_sorted_avg[1][:title]) * "_norm_sorted_$(p_sorted_avg[1][i][:label])")
				println("Exporting series \"$(series_name)\"...")
				latex_df_push_column!(df_xattributes, series_name, p_sorted_avg[1][i][:y])
			end
		end
		plot!(p_sorted_avg, xticks = (1:length(perm), [isnothing(attribute_abbrs) ? attribute_names[t] : attribute_abbrs[t] for t in perm]));
		# plot!(p_sorted_avg, xticks = (1:length(perm), [isnothing(attribute_abbrs) ? attribute_names[t] : attribute_abbrs[t] for t in perm]));
		_savefig(plot(p_sorted_avg; size = (1920, 1080), margin = 5mm), "norm-sorted-avg");
		############################################################################
		############################################################################
		############################################################################

		# display(p_norm_avg)

		p = SoleViz.plotdescription(
			[descriptions],
			group_descriptors = grouped_descriptors,
			attribute_names = attribute_names,
			cache_stats = !isnothing(cache_description_in) ? joinpath(cache_description_in, "stats") : nothing,
			join_plots = :descriptors in join_plots,
		)
		# println(size(p))
		p = p[:]

		############################################################################
		############################################################################
		############################################################################
		@assert length(grouped_descriptors) == 8 "TODO change layout"


		for (p_i_idx,p_i) in enumerate(collect(p))
			plot!(p_i, xticks = false)
		end

		_savefig(plot(p...; layout = (4,2), size = (1920, 1080), margin = 5mm), "grouped");

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
			if p_i_idx in [7,8]
				plot!(p_i, xticks = (1:length(perm), [isnothing(attribute_abbrs) ? attribute_names[t] : attribute_abbrs[t] for t in perm]));
			else
				plot!(p_i, xticks = false)
			end
		end
		_savefig(plot(p_sorted...; layout = (4,2), size = (1920, 1080), margin = 5mm), "grouped-sorted");

		############################################################################
		############################################################################
		############################################################################
		p_norm = normalize_plot.(p);
		_savefig(plot(p_norm...; layout = (4,2), size = (1920, 1080), margin = 5mm), "grouped-norm");

		############################################################################
		############################################################################
		############################################################################
		perm = _sortperm(p_norm_avg[1][1][:y], rev=true, nan_mode = :last)
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
			if p_i_idx in [7,8]
				plot!(p_i, xticks = (1:length(perm), [isnothing(attribute_abbrs) ? attribute_names[t] : attribute_abbrs[t] for t in perm]));
			else
				plot!(p_i, xticks = false)
			end
		end
		_savefig(plot(p_sorted...; layout = (4,2), size = (1920, 1080), margin = 5mm), "grouped-norm-sorted");

		############################################################################
		############################################################################
		############################################################################

		CSV.write("$(file_prefix)-xattributes.csv",  df_xattributes)

		perm[1:n_desired_attributes]
	end

	X_sub = X[:,best_attributes_idxs,:]
	Y_sub = Y[:]

	println("Selecting $(length(best_attributes_idxs)) attributes: $(best_attributes_idxs)...")
	attribute_names_sub = attribute_names[best_attributes_idxs]
	println(["$(t): $(n)" for (t,n) in zip(best_attributes_idxs,attribute_names_sub)])

	mfd_sub = compute_mfd(X_sub);

	best_descriptors = begin

		if export_csv
			df_xdescriptors = DataFrame()
			latex_df_push_column!(df_xdescriptors, :index, collect(0:(length(descriptors)-1)))
			# latex_df_push_column!(df_xdescriptors, :descriptor_names, string.(descriptors))
			# latex_df_push_column!(df_xdescriptors, :descriptor_abbrs, begin
			# 		if isnothing(descriptor_abbrs)
			# 			string.(descriptors)
			# 		else
			# 			[descriptor_abbrs[d] for d in descriptors]
			# 		end
			# 	end
			# )
		end

		descriptions_sub_args = (mfd_sub,)

		descriptions_sub =
			if !isnothing(cache_description_in)
				@cachefast "plotdescription" joinpath(cache_description_in, "plotdescription") descriptions_sub_args descriptions_kwargs SoleBase.describe
			else
				SoleBase.describe(descriptions_sub_args...; descriptions_kwargs...)
			end

		p = SoleViz.plotdescription(
			[descriptions_sub],
			group_descriptors = descriptors,
			on_x_axis = :descriptors,
			attribute_names = attribute_names_sub,
			cache_stats = !isnothing(cache_description_in) ? joinpath(cache_description_in, "stats") : nothing,
			join_plots = :attributes in join_plots,
		)
		println(size(p))
		p = p[:]

		############################################################################
		############################################################################
		############################################################################

		for i in 1:(length(p)-1)
			plot!(p[i]; xticks = false)
		end

		_savefig(plot(p...; layout = (length(p),1), size = (1920, 1080), margin = 5mm), "transposed");

		p_avg = average_plot(p);
		_savefig(plot(p_avg; size = (1920, 1080), margin = 5mm), "transposed-avg");
		############################################################################
		############################################################################
		############################################################################
		perm = _sortperm(p_avg[1][1][:y], rev=true, nan_mode = :last)
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
			if (p_i_idx % 5) == 0
				plot!(p_i, xticks = (1:length(perm), [isnothing(descriptor_abbrs) ? string(descriptors[t]) : descriptor_abbrs[descriptors[t]] for t in perm]));
			else
				plot!(p_i, xticks = false)
			end
		end
		_savefig(plot(p_sorted...; layout = (length(p_sorted),1), size = (1920, 1080), margin = 5mm), "transposed-sorted");

		############################################################################
		############################################################################
		############################################################################
		p_norm = normalize_plot.(p);
		_savefig(plot(p_norm...; layout = (length(p_norm),1), size = (1920, 1080), margin = 5mm), "transposed-norm");

		p_norm_avg = average_plot(p_norm);
		_savefig(plot(p_norm_avg; size = (1920, 1080), margin = 5mm), "transposed-norm-avg");
		############################################################################
		############################################################################
		############################################################################
		perm = _sortperm(p_norm_avg[1][1][:y], rev=true, nan_mode = :last)
		if export_csv
			_descriptornames = string.(descriptors)
			_descriptorabbrs = begin
				if isnothing(descriptor_abbrs)
					string.(descriptors)
				else
					[descriptor_abbrs[d] for d in descriptors]
				end
			end
			latex_df_push_column!(df_xdescriptors, :descriptor_names_norm_sorted, _descriptornames[perm])
			latex_df_push_column!(df_xdescriptors, :descriptor_abbrs_norm_sorted, _descriptorabbrs[perm])
		end
		p_sorted = Base.deepcopy(p_norm);
		for (p_i_idx,p_i) in enumerate(collect(p_sorted))
			i = 1
			while true
				try p_i[1][i]
					p_i[1][i][:y] = p_i[1][i][:y][perm]
					if export_csv
						series_name = string(p_i[1][i][:label]) * "_norm_sorted"
						println("Exporting series \"$(series_name)\"...")
						latex_df_push_column!(df_xdescriptors, series_name, p_i[1][i][:y])
					end
					i += 1
				catch Error
					break
				end
			end
			if (p_i_idx % 5) == 0
				plot!(p_i, xticks = (1:length(perm), [isnothing(descriptor_abbrs) ? string(descriptors[t]) : descriptor_abbrs[descriptors[t]] for t in perm]));
			else
				plot!(p_i, xticks = false)
			end
		end
		_savefig(plot(p_sorted...; layout = (length(p_sorted),1), size = (1920, 1080), margin = 5mm), "transposed-norm-sorted");

		p_sorted_avg = Base.deepcopy(p_norm_avg);
		for i in 1:length(p_sorted_avg[1])
			p_sorted_avg[1][i][:y] = p_sorted_avg[1][i][:y][perm]
			if export_csv
				series_name = string(p_sorted_avg[1][:title]) * "_norm_sorted"
				println("Exporting series \"$(series_name)\"...")
				latex_df_push_column!(df_xdescriptors, series_name, p_sorted_avg[1][i][:y])
				# df_xdescriptors[!,string(p_sorted_avg[1][:title]) * "_norm_sorted_$(p_sorted_avg[1][i][:label])"] = p_sorted_avg[1][i][:y]
			end
		end
		plot!(p_sorted_avg, xticks = (1:length(perm), [isnothing(descriptor_abbrs) ? string(descriptors[t]) : descriptor_abbrs[descriptors[t]] for t in perm]));
		# plot!(p_sorted_avg, xticks = (1:length(perm), [isnothing(descriptor_abbrs) ? string(descriptors[t]) : descriptor_abbrs[descriptors[t]] for t in perm]));
		_savefig(plot(p_sorted_avg; size = (1920, 1080), margin = 5mm), "transposed-norm-sorted-avg");
		############################################################################
		############################################################################
		############################################################################

		best_descriptors = descriptors[perm[1:n_desired_features]]

		CSV.write("$(file_prefix)-xdescriptors.csv", df_xdescriptors)

		best_descriptors
	end

	println("Selecting $(length(best_descriptors)) descriptors: $(best_descriptors)...")
	println()

	best_attributes_idxs, best_descriptors
end

# descriptors = best_descriptors
# single_frame_target_aware_analysis((X, Y), attribute_names, descriptors, file_prefix; savefigs = false, descriptor_abbrs = descriptor_abbrs, attribute_abbrs = attribute_abbrs)

function single_frame_target_aware_analysis(
		(X, Y)::Tuple{AbstractArray{T,3},<:Union{<:AbstractVector{<:AbstractString},<:AbstractVector{<:Real}}},
		attribute_names::AbstractVector,
		descriptors::AbstractVector{Symbol},
		file_prefix::AbstractString;
		make_bins::Union{<:NTuple{N,<:L},<:Tuple{<:NTuple{N,<:L},<:NTuple{M,<:AbstractString}},Nothing} = nothing,
		savefigs::Bool = false,
		descriptor_abbrs = nothing,
		attribute_abbrs = nothing,
		export_csv::Bool = false,
		plot_normals::Bool = false,
	) where {T,N,M,L<:Real}

	if !isnothing(make_bins)
		if make_bins isa Tuple{<:Tuple,<:Tuple}
			@assert [make_bins[1]...] == sort([make_bins[1]...]) "split positions in `make_bins` have to be in increasing order"
			@assert length(make_bins[1]) == length(unique(make_bins[1])) "split positions in `make_bins` have to be unique"
			@assert M == N+1 "The number of class names in `make_bins` should equals the length " *
				"of the split positions +1: $(M) != $(N) + 1"
		else
			@assert [make_bins...] == sort([make_bins...]) "split positions in `make_bins` have to be in increasing order"
			@assert length(make_bins) == length(unique(make_bins)) "split positions in `make_bins` have to be unique"
		end
	end

	@assert length(attribute_names) == size(X)[end-1] "$(length(attribute_names)) != $(size(X)[end-1])"

	mfd = compute_mfd(X);

	n_instances = size(X)[end]

	_savefig = savefigs ? (plot, plot_name)->begin
			savefig(plot, "$(file_prefix)$((plot_name == "" ? "" : "-$(plot_name)")).png")
		end : (x...)->(;)

	win = 1

	descriptions = SoleBase.describe(mfd; desc = descriptors, t = [[(win,0,0)]])[1]

	# TODO: probably should be changed this variable name due to its different meaning when eltype(Y) <: Real
	class_names, idxs_per_class =
		if !isnothing(make_bins) && eltype(Y) <: Real
			cns =
				if typeof(make_bins) <:Tuple{<:NTuple{N,<:L},<:NTuple{M,<:AbstractString}}
					new_labels = [make_bins[2]...]
					make_bins = make_bins[1]
					new_labels
				else
					[string(i) for i in 1:(length(make_bins)+1)]
				end

			function locate_ordered(n::Real)
				if n < make_bins[1]
					return 1
				elseif n ≥ make_bins[end]
					return length(make_bins)+1
				else
					for (i_v, (v1, v2)) in [(i, (make_bins[i-1], make_bins[i])) for i in 2:length(make_bins)]
						if v1 ≤ n < v2
							return i_v
						end
					end
				end
			end

			new_Y = [cns[locate_ordered(curr_Y)] for curr_Y in Y]

			cns, [new_Y .== class_name for class_name in cns]
		else
			if !isnothing(make_bins)
				@warn "Making BINs is supported only for numeric labels: ignoring..."
			end

			cns =
				if eltype(Y) <: AbstractString
					unique(Y)
				elseif eltype(Y) <: Integer
					# NOTE: strong assumption about the meaning of an Integer label
					ext = extrema(Y)
					collect(ext[1]:ext[2])
					# collect(range(extrema(Y)...)) # not supported in julia < 1.7
				else
					sort(unique(Y))
				end

			cns, [Y .== class_name for class_name in cns]
		end


	values = Array{NTuple{length(class_names)}}(undef, length(descriptors), length(attribute_names))

	for (i_descriptor,descriptor) in enumerate(descriptors)
		# println("Analysing $(descriptor)...")
		description = descriptions[:,descriptor]
		for (i_attribute,(attribute_name,attr_description)) in enumerate(zip(attribute_names,description))
			values[i_descriptor,i_attribute] = Tuple([attr_description[class_idxs] for class_idxs in idxs_per_class])
		end
	end

	# print vector Python-style https://discourse.julialang.org/t/printing-an-array-without-element-type/6731

	df_distribution = DataFrame()
	df_values = DataFrame(:class_names => class_names)
	plots = Matrix{Any}(undef, length(descriptors), length(attribute_names))
	for (i_attribute,attribute_name) in enumerate(attribute_names)
		for (i_descriptor,descriptor) in enumerate(descriptors)
			subp = plot() # title = string(descriptor))
			att = (isnothing(attribute_abbrs) ? attribute_name : attribute_abbrs[i_attribute])
			desc = (isnothing(descriptor_abbrs) ? descriptor : descriptor_abbrs[descriptor])
			class_vs = values[i_descriptor,i_attribute]
			# class_is_normal = []

			# histogram!(subp, class_vs...)
			for (i_class,(v, class_name)) in enumerate(zip(class_vs, class_names))

				try
					# histogram!(subp, v)
					density!(subp, v, legend = false)

					if export_csv
						latex_df_push_column!(df_distribution, "$(att)-$(desc)-$(class_name)-x", deepcopy(collect(subp[1][i_class][:x])))
						latex_df_push_column!(df_distribution, "$(att)-$(desc)-$(class_name)-y", deepcopy(collect(subp[1][i_class][:y])))

						# CSV.write("$(file_prefix)-distribution-$(att)-$(desc)-$(class_name).csv", df_distribution)
					end
				catch ex
					println("ERROR! ($(typeof(ex))) In plotting density for attribute $(attribute_name) and descriptor $(descriptor): $(v)")
					if ex isa ArgumentError
						println(" └─ $(ex.msg)")
					elseif ex isa BoundsError
						println(" └─ attempt to access $(typeof(ex.a)) of size $(size(ex.a)) at index $(ex.i)")
					end
				end

			end

			# for (i_class,(v, class_name)) in enumerate(zip(values[i_descriptor,i_attribute], class_names))
				# if plot_normals
				# 	# ERROR: if we plot normals on subp, export_csv fails (it just has to be fixed)
				# 	plot!(subp, fit(Normal, v)) #, fill=(0, .5,:orange))
				# end

				# push!(class_is_normal, )
			# end

			if export_csv
				df_values[!,"$(att)-$(desc)"] = show_vector_sans_type.(collect(class_vs))
			end
			# "$(string(descriptors))($(i_attribute))"
			plots[i_descriptor, i_attribute] = subp
		end
	end

	CSV.write("$(file_prefix)-xdistributions.csv", df_distribution)
	CSV.write("$(file_prefix)-values.csv", df_values)

 	p = plot!(plots..., size = (1920, 1080), margin = 5mm, labels = class_names)
	# p = plot!(p..., title = "$(attribute_name)", size = (1920, 1080), margin = 5mm)
	# display(p)
	# _savefig(plot!(p..., size = (1920, 1080), margin = 5mm), "$(attribute_name)");
	# readline()

 	println("single_frame_target_aware_analysis:")
 	println("- all-plot attributes: $(attribute_names)")
 	println("- all-plot descriptors: $(descriptors)")


	_savefig(p, "all");
end
