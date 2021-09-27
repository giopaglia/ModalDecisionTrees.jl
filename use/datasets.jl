using MAT
import Base: getindex, values
import Random
using JSON

include("local.jl")
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

The search can be recurisve by setting `recursive` to `ture`.
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

function generate_splitted_wavs_dataset(
            path      :: String;
            exclude   :: Union{Vector{String},String}  = Vector{String}(),
            draw_wavs :: Bool    = false,
            limit     :: Int64   = 0
        )::Nothing
    dest = rstrip(path, '/') * "-split-wavs/"
    mkpath(dest)

    for wav_found in searchdir(path, ".wav"; exclude = exclude, recursive = true, results_limit = limit)
        mkpath(dirname(wav_found))


        wavs, sr = splitwav(wav_found)
        mkpath(dirname(replace(wav_found, path => dest; count = 1)))

        Threads.@threads for (i, w) in collect(enumerate(wavs))
            wavwrite(w, replace(replace(wav_found, ".wav" => ".split.$(i).wav"), path => dest; count = 1); Fs = sr)
        end

        if draw_wavs
            plts = []
            wav_orig, sr_orig = wavread(wav_found)
            wav_orig = merge_channels(wav_orig)

            orig_name = replace(basename(wav_found), ".wav" => "")
            push!(plts, draw_wav(wav_orig, sr_orig; title = orig_name))

            for (i, w) in enumerate(wavs)
                push!(plts, draw_wav(w, sr; title = orig_name * ".split.$(i)"))
            end

            final_plot = plot(plts..., layout = (length(plts), 1), size = (1000, 150 * length(plts)))
            savefig(final_plot, replace(replace(wav_found, ".wav" => ".graph.png"), path => dest; count = 1))
        end
    end

end

# """
#     process_dataset(dataset_dir_name, wav_paths; preprocess = [], postprocess = [], split_instances = false)

# Process a dataset named `dataset_dir_name` using
# `preprocess` splitting instances (if `split_instances`
# is true) and then 
# """
# function process_dataset(
#         dataset_dir_name :: String,
#         wav_paths        :: Vector{String};
#         preprocess       :: Vector{Tuple{Function,NamedTuple}} = Vector{Tuple{Function,NamedTuple}}(),
#         postprocess      :: Vector{Tuple{Function,NamedTuple}} = Vector{Tuple{Function,NamedTuple}}(),
#         split_instances  :: Bool                               = false
#     )

#     @assert length(wav_paths) > 0 "No wav path passed"

#     function name_processing(type::String, pp::Vector{Tuple{Function,NamedTuple}})
#         if length(pp) == 0 return "" end

#         result = "$(type)["
#         for p in pp
#             result *= string(p[1], ",")
#         end
#         result = rstrip(result, ",")
#         result *= "]"

#         result
#     end

#     outdir = rstrip(data_dir, "/") * "/" * dataset_dir_name * "-" * name_processing("PREPROC", preprocess) * "-" * name_processing("POSTPROC", postprocess) * (split_instances ? "-splitted" : "")


#     samples_n_samplerates = Vector{Tuple{Vector{T},Real}}(undef, length(wav_paths))
#     Threads.@threads for (i, path) in collect(enumerate(wav_paths))
#         samples, samplerate = wavread(path)
#         samples = merge_channels(samples)

#         samples_n_samplerates[i] = (samples, samplerate)
#     end

#     for (prepf, prepkwargs) in preprocess
#         prepf(dataset, prepkwargs...)
#     end

#     # do the eventual splitting
#     if split_instances
#     end

#     for (postpf, postpkwargs) in postprocess
#         postpf(dataset, postpkwargs...)
#     end

#     # save dataset in a new directory named accordingly to all parameters
#     mkpath(outdir)
# end
# process_dataset(dataset_dir_name::String, wav_paths_func::Function; kwargs...) = process_dataset(dataset_dir_name, wav_paths_func(); kwargs...)

# process_dataset("KDD", x -> load_from_jsonKDD(n_version, n_task), )

include("datasets/Multivariate_arff.jl")
include("datasets/KDD.jl")
include("datasets/land-cover.jl")
include("datasets/ComParE2021.jl")
include("datasets/LoadModalDataset.jl")
