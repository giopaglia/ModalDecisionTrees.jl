using MAT
import Base: getindex, values
import Random
using JSON

data_dir = "../datasets/" # TODO move this to an .env file?

include("wav2stft_time_series.jl")
include("wave-utils/wav-process.jl")

# include("datasets/dummy.jl")
# include("datasets/eduard.jl")
# include("datasets/siemens-0.jl")

# https://stackoverflow.com/questions/59562325/moving-average-in-julia
moving_average(vs::AbstractArray{T,1},n,st=1) where {T} = [sum(@view vs[i:(i+n-1)])/n for i in 1:st:(length(vs)-(n-1))]
moving_average(vs::AbstractArray{T,2},n,st=1) where {T} = mapslices((x)->(@views moving_average(x,n,st)), vs, dims=1)
# (sum(w) for w in partition(1:9, 3, 2))
# moving_average_np(vs,num_out_points,st) = moving_average(vs,length(vs)-num_out_points*st+1,st)
# moving_average_np(vs,num_out_points,o) = (w = length(vs)-num_out_points*(1-o/w)+1; moving_average(vs,w,1-o/w))
# moving_average_np(vs,t,o) = begin
# 	N = length(vs);
# 	s = floor(Int, (N+1)/(t+(1/(1-o))))
# 	w = ceil(Int, s/(1-o))
# 	# moving_average(vs,w,1-ceil(Int, o/w))
# end

"""
	searchdir(path, key; exclude = [], recursive = false, results_limit = 0)

Search the directory at `path` for files containing `key`.

Exceptions can be specified using `exclude` and the amount
of results can be limited by specifying a value greater than 0
for `results_limit`.

The search can be recurisve by setting `recursive` to `true`.
"""
function searchdir(
			path          :: String,
			key           :: Union{Vector{String},String};
			exclude       :: Union{Vector{String},String}  = Vector{String}(),
			recursive     :: Bool                          = false,
			results_limit :: Int64                         = 0
		)::Vector{String}

	function contains_str(str::String, match::String)::Bool
		occursin(match, str)
	end
	function contains_str(str::String, match::Vector{String})::Bool
		length(findall(contains_str(str, m) for m in match)) > 0
	end
	function matches_key(x::String)::Bool
		occursin(key, x) && !isdir(path * "/" * x) && !contains_str(x, exclude)
	end

	results = Vector{String}()

	dir_content = readdir(path)
	append!(results, map(res -> path * "/" * res, filter(matches_key, dir_content)))

	if recursive
		for d in filter(x -> isdir(path * "/" * x), dir_content)
			if results_limit > 0 && length(results) > results_limit break end
			append!(results, searchdir(path * "/" * d, key; exclude = exclude, recursive = recursive, results_limit = results_limit))
		end
	end

	if results_limit > 0 && length(results) > results_limit
		deepcopy(results[1:results_limit])
	else
		results
	end
end

"""
	draw_wavs_for_partitioned_dataset(original_dataset_dir, partitioned_dataset_dir)

Given the `original_dataset_dir` and its partitioned version `original_dataset_dir`
draw in stack the original wav and all the resulting partitions.
"""
function draw_wavs_for_partitioned_dataset(original_dataset_dir::String, partitioned_dataset_dir::String)
	function splitted_name(orig::String, n::Integer)::String
		replace(orig, original_dataset_dir => partitioned_dataset_dir; count = 1) * "-split.$(n).wav"
	end

	if !isdir(original_dataset_dir) && !startswith(original_dataset_dir, data_dir)
		original_dataset_dir = data_dir * original_dataset_dir
	end

	if !isdir(partitioned_dataset_dir) && !startswith(partitioned_dataset_dir, data_dir)
		partitioned_dataset_dir = data_dir * partitioned_dataset_dir
	end

	for original_path in searchdir(original_dataset_dir, ".wav"; recursive = true)
		files_to_draw = [ original_path ]
		i = 1
		while isfile(splitted_name(original_path, i))
			push!(files_to_draw, splitted_name(original_path, i))
			i += 1
		end
		if length(files_to_draw) > 1
			println("Drawing partitioned \"$(original_path)\"")
			for i in 2:(length(files_to_draw)-1)
				println(" ├─ \"$(splitted_name(original_path, i-1))\"")
			end
			println(" └─ \"$(splitted_name(original_path, length(files_to_draw)-1))\"")

			draw_wav(files_to_draw)
			savefig(replace(splitted_name(original_path, 0), "-split.0.wav" => "-splits.png"))
		end
	end
end
# just a debugging function
# function generate_splitted_wavs_dataset(
#             path      :: String;
#             exclude   :: Union{Vector{String},String}  = Vector{String}(),
#             draw_wavs :: Bool    = false,
#             limit     :: Int64   = 0
#         )::Nothing
#     dest = rstrip(path, '/') * "-split-wavs/"
#     mkpath(dest)

#     for wav_found in searchdir(path, ".wav"; exclude = exclude, recursive = true, results_limit = limit)
#         mkpath(dirname(wav_found))


#         wavs, sr = partitionwav(wav_found)
#         mkpath(dirname(replace(wav_found, path => dest; count = 1)))

#         Threads.@threads for (i, w) in collect(enumerate(wavs))
#             wavwrite(w, replace(replace(wav_found, ".wav" => ".split.$(i).wav"), path => dest; count = 1); Fs = sr)
#         end

#         if draw_wavs
#             plts = []
#             wav_orig, sr_orig = wavread(wav_found)
#             wav_orig = merge_channels(wav_orig)

#             orig_name = replace(basename(wav_found), ".wav" => "")
#             push!(plts, draw_wav(wav_orig, sr_orig; title = orig_name))

#             for (i, w) in enumerate(wavs)
#                 push!(plts, draw_wav(w, sr; title = orig_name * ".split.$(i)"))
#             end

#             final_plot = plot(plts..., layout = (length(plts), 1), size = (1000, 150 * length(plts)))
#             savefig(final_plot, replace(replace(wav_found, ".wav" => ".graph.png"), path => dest; count = 1))
#         end
#     end

# end

"""
	partition_wavs_and_get_names(samples_n_samplerates, paths; partitioning_kwargs = ())
	partition_wavs_and_get_names(paths; partitioning_kwargs = ())

Call [`partitionwav`](@ref) on all inputed WAVs and then return a filepath
for each slice of the original wave
"""
function partition_wavs_and_get_names(
			samples_n_samplerates :: AbstractVector{<:Tuple{<:AbstractVector{<:T},SR}},
			paths                 :: AbstractVector{String};
			partitioning_kwargs   :: NamedTuple                    = NamedTuple()
		)::Tuple{Vector{Tuple{Vector{T},Real}},Vector{String}} where {T<:AbstractFloat,SR<:Real}

	results_sns = Vector{Tuple{Vector{T},Real}}()
	results_paths = Vector{String}()

	sns_lock = Threads.Condition()
	paths_lock = Threads.Condition()
	# using these locks make sense because most of each iteration time will be spent in 'partitionwav`
	# Threads.@threads
	for (sns, path) in collect(zip(samples_n_samplerates, paths))
		wavs, samplerate = partitionwav(sns; partitioning_kwargs...)
		lock(sns_lock)
		append!(results_sns, collect(zip(wavs, fill(samplerate, length(wavs)))))
		unlock(sns_lock)
		lock(paths_lock)
		# append!(results_paths, [ replace(path, r".wav$" => ".split.$(i).wav") for i in 1:length(wavs) ])
		append!(results_paths, [ "$(path)-split.$(i).wav" for i in 1:length(wavs) ])
		unlock(paths_lock)
	end

	@assert length(results_sns) == length(results_paths) "Mismatching length: length(results_sns) vs length(results_paths): $(length(results_sns)) != $(length(results_paths))"

	results_sns, results_paths
end
function partition_wavs_and_get_names(paths::Vector{String}; kwargs...)::Tuple{Tuple{Vector{AbstractFloat},Real},Vector{String}}
	sns = Vector{Tuple{Vector{AbstractFloat},Real}}(undef, length(paths))
	Threads.@threads for (i, p) in collect(enumerate(paths))
		samples, samplerate = wavread(p)
		sns[i] = (merge_channels(samples), samplerate)
	end
	split_wavs(sns, paths; kwargs...)
end

"""
	process_dataset(dataset_dir_name, wav_paths; preprocess = [], postprocess = [], partition_instances = false, partitioning_kwargs = (preprocess = [], postprocess = []))

Process a dataset named `dataset_dir_name` using
`preprocess` splitting instances (if `partition_instances`
is `true`) and then apply to all new wavs `postprocess`.
"""
function process_dataset(
			dataset_dir_name     :: String,
			filepaths            :: Vector{String};
			out_dataset_dir_name :: Union{String,Nothing}              = nothing,
			preprocess           :: Vector{Tuple{Function,NamedTuple}} = Tuple{Function,NamedTuple}[],
			postprocess          :: Vector{Tuple{Function,NamedTuple}} = Tuple{Function,NamedTuple}[],
			partition_instances  :: Bool                               = false,
			partitioning_kwargs  :: NamedTuple                         = (
				cut_original = Val(true),
				preprocess   = Function[],
				postprocess  = Function[],
			)
		)

	@assert length(filepaths) > 0 "No wav path passed"

	function name_processing(type::String, pp::Vector{Tuple{Function,NamedTuple}})
		result = "$(type)["
		for p in pp
			result *= string(p[1])
			if length(pp[2]) > 0
				result *= string("{", pp[2], "}")
			end
			result *= ","
		end
		result = rstrip(result, ',')
		result *= "]"

		result
	end

	outdir = rstrip(data_dir, '/') * "/" * (
		!isnothing(out_dataset_dir_name) ? out_dataset_dir_name :
			dataset_dir_name *
			(partition_instances ? "-partitioned" : "") *
			"-" * name_processing("PREPROC", preprocess) *
			"-" * name_processing("POSTPROC", postprocess) *
			""
		)
	println("outdir = $(outdir)")

	# read wavs
	Threads.@threads for (i, filepath) in collect(enumerate(filepaths))

		samples, samplerate = wavread(rstrip(data_dir, '/') * "/" * dataset_dir_name * "/" * filepath)
		samples = merge_channels(samples)

		# apply pre-process
		for (prepf, prepkwargs) in preprocess
			prepf(samples, prepkwargs...)
		end

		# partition
		new_sampless_n_samplerates, new_filepaths =
			if partition_instances
				partition_wavs_and_get_names([(samples, samplerate)], [filepath]; partitioning_kwargs = partitioning_kwargs)
			else
				[(samples, samplerate)], [filepath]
		end

		for ((new_samples, new_samplerate),new_filepath) in zip(new_sampless_n_samplerates,new_filepaths)
			# apply post-process
			for (postpf, postpkwargs) in postprocess
				postpf(new_samples, postpkwargs...)
			end

			# save
			mkpath(outdir * "/" * dirname(new_filepath))
			wavwrite(new_samples, outdir * "/" * new_filepath; Fs = new_samplerate)
		end
	end
end
# process_dataset(dataset_dir_name::String, wav_paths_func::Function; kwargs...) = process_dataset(dataset_dir_name, wav_paths_func(); kwargs...)

# process_dataset("KDD", x -> load_from_jsonKDD(n_version, n_task), )

include("datasets/Multivariate_arff.jl")
include("datasets/KDD.jl")
include("datasets/land-cover.jl")
include("datasets/ComParE2021.jl")
include("datasets/LoadModalDataset.jl")
