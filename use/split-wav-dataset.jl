
include("wav-filtering.jl")

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
            wavwrite(w, sr, replace(replace(wav_found, ".wav" => ".$(i).split.wav"), path => dest; count = 1))
        end
        
        if draw_wavs
            plts = []
            wav_orig, sr_orig = wavread(wav_found)
            wav_orig = merge_channels(wav_orig)

            orig_name = replace(basename(wav_found), ".wav" => "")
            push!(plts, draw_wav(wav_orig, sr_orig; title = orig_name))

            for (i, w) in enumerate(wavs)
                push!(plts, draw_wav(w, sr; title = orig_name * ".$(i).split"))
            end

            final_plot = plot(plts..., layout = (length(plts), 1), size = (1000, 150 * length(plts)))
            savefig(final_plot, replace(replace(wav_found, ".wav" => ".graph.png"), path => dest; count = 1))
        end
    end
    
end

generate_splitted_wavs_dataset("../datasets/KDD/"; exclude = [ "aug", "mono", "pitch" ], draw_wavs = true, limit = 10)
