
include("wav-filtering.jl")

function searchdir(path::String, key::String; recursive::Bool = false)::Vector{String}
    results = Vector{String}()

    dir_content = readdir(path)
    append!(results, map(res -> path * "/" * res, filter(x -> occursin(key, path * "/" * x), dir_content)))
    
    if recursive
        for d in filter(x -> isdir(path * "/" * x), dir_content)
            append!(results, searchdir(path * "/" * d, key; recursive = recursive))
        end
    end

    results
end

function generate_splitted_wavs_dataset(path::String; draw_wavs::Bool = false)::Nothing
    dest = rstrip(path, '/') * "-split-wavs/"
    mkpath(dest)

    for wav_found in searchdir(path, ".wav"; recursive = true)
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

# generate_splitted_wavs_dataset("../datasets/KDD/"; draw_wavs = true)
