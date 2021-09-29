
include("scanner.jl")
include("wav2stft_time_series.jl")
include("local.jl")
include("lib.jl")

using DecisionTree
using DecisionTree.ModalLogic
using JSON

include("wave-utils/wav-lib.jl")
include("wave-utils/wav-process.jl")
include("wave-utils/wav-drawing.jl")

"""
TODO: docs
"""
mutable struct InstancePathInTree{S}
    file_name    :: String
    label        :: S
    tree         :: Union{Nothing,DTree{T}} where T
    predicted    :: Union{Nothing,S}
    path         :: DecisionPath
    dataset_info :: Any
    InstancePathInTree{S}(file_name::String, label::S) where S = new(file_name, label, nothing, nothing, [], ())
    InstancePathInTree{S}(file_name::String, label::S, dataset_info) where S = new(file_name, label, nothing, nothing, [], dataset_info)
    InstancePathInTree{S}(file_name::String, label::S, tree::DTree) where S = new(file_name, label, tree, nothing, [], ())
    InstancePathInTree{S}(file_name::String, label::S, tree::DTree, dataset_info) where S = new(file_name, label, tree, nothing, [], dataset_info)
end

"""
    is_correctly_classified(inst)

Check if an `InstancePathInTree` was correctly classified.

Note: returns `false` even if the prediction was not performed
(`inst.predicted` is `nothing`).
"""
is_correctly_classified(inst::InstancePathInTree)::Bool = !isnothing(inst.predicted) && inst.label === inst.predicted

"""
TODO: docs
"""
function apply_tree_to_datasets_wavs(
        tree_hash                             :: String,
        tree                                  :: DTree{S},
        dataset                               :: GenericDataset,
        wav_paths                             :: Vector{String},
        labels                                :: Vector{S};
        # MODE OPTIONS
        filter_type                           :: Symbol             = :dynamic,  # :static
        mode                                  :: Symbol             = :bandpass, # :emphasize
        # OUTPUT OPTIONS
        only_files                            :: Vector{String}     = Vector{String}(),
        files_already_generated               :: Bool               = false,
        use_original_dataset_filesystem_tree  :: Bool               = false,
        destination_dir                       :: String             = "filtering-results/filtered",
        remove_from_path                      :: String             = "",
        save_single_band_tracks               :: Bool               = true,
        generate_json                         :: Bool               = true,
        json_file_name                        :: String             = "files.json",
        verbose                               :: Bool               = false,
        draw_worlds_on_file                   :: Bool               = true,
        # WAV POST-PROCESS
        postprocess_wavs                      :: Vector{Function}   = [ trim_wav!,      normalize! ],
        postprocess_wavs_kwargs               :: AbstractVector     = [ (level = 0.0,), (level = 1.0,) ],
        # FILTER OPTIONS
        filter_kwargs                         :: NamedTuple         = (nbands = 40,),
        window_f                              :: Function           = triang,
        # SPECTROGRAM OPTIONS
        generate_spectrogram                  :: Bool               = true,
        spectrograms_kwargs                   :: NamedTuple         = NamedTuple(),
        spec_show_only_found_bands            :: Bool               = true,
        # ANIM AND VIDEO OPTIONS (note: can't create video without animation)
        draw_anim_for_instances               :: Vector{Int64}      = Vector{Int64}(),
        anim_kwargs                           :: NamedTuple         = NamedTuple(),
        normalize_before_draw_anim            :: Bool               = false,
        generate_video_from_gif               :: Bool               = true,
        video_kwargs                          :: NamedTuple         = NamedTuple(),
        # SAMPLERATE ADJUST PARAMETERS FOR WORLDS
        wintime                               :: Real               = 0.025,
        steptime                              :: Real               = 0.01,
        movingaverage_size                    :: Int64              = 1,
        movingaverage_step                    :: Int64              = 1
    ) where {S}

    @assert mode in [ :bandpass, :emphasize ] "Got unsupported mode Symbol $(mode). It can be $([ :bandpass, :emphasize ])."
    @assert filter_type in [ :dynamic, :static ] "Got unsupported filter_type Symbol $(filter_type). It can be $([ :dynamic, :static ])."

    @assert :nbands in keys(filter_kwargs) "Need to pass at least nbands as parameter for filter_kwargs. ex: (nbands = 40,)"

    # remove nbands from filter_kwargs
    nbands = filter_kwargs[:nbands]
    filt_kw_dict = Dict{Symbol,Any}()
    for k in keys(filter_kwargs)
        if k == :nbands continue end
        filt_kw_dict[k] = filter_kwargs[k]
    end
    filter_kwargs = (; filt_kw_dict...)
    println(filter_kwargs)

    # TODO:
    if mode == :emphasize
        @warn "The value :emphasize for 'mode` is still experimental!"
    end
    if filter_type == :dynamic
        @warn "The value :dynamic for 'filter_type` is still experimental!"
    end

    n_instances = n_samples(dataset)

    println()
    println("Applying tree $(tree_hash):")
    print_tree(tree)
    println()
    print_apply_tree(tree, dataset, labels)
    println()

    if length(postprocess_wavs) == 0
        empty!(postprocess_wavs_kwargs)
    end

    @assert n_instances == length(wav_paths) "dataset and wav_paths length mismatch! $(n_instances) != $(length(wav_paths))"
    @assert n_instances == length(labels) "dataset and labels length mismatch! $(n_instances) != $(length(labels))"

    @assert length(postprocess_wavs) == length(postprocess_wavs_kwargs) "length(postprocess_wavs) != length(postprocess_wavs_kwargs): $(length(postprocess_wavs)) != $(length(postprocess_wavs_kwargs))"

    if dataset isa MultiFrameModalDataset
        @assert n_frames(dataset) == 1 "MultiFrameModalDataset with more than one frame is still not supported! n_frames(dataset): $(n_frames(dataset))"
    end

    json_archive = Dict{String, Vector{String}}()

    results = Vector{InstancePathInTree}(undef, n_instances)
    predictions = apply_tree(tree, dataset)
    paths = get_path_in_tree(tree, dataset)
    Threads.@threads for i in 1:n_instances
        results[i] = InstancePathInTree{S}(wav_paths[i], labels[i], tree)
        
        results[i].predicted = predictions[i]
        results[i].path = paths[i]
    end

    originals = Vector{Vector{Float64}}(undef, n_instances)
    samplerates = Vector{Number}(undef, n_instances)
    Threads.@threads for i in 1:n_instances
        curr_orig, samplerates[i] = wavread(wav_paths[i])
        originals[i] = merge_channels(curr_orig)
    end

    if :maxfreq in keys(filter_kwargs)
        min_sr = minimum(samplerates)
        @assert (min_sr / 2) >= filter_kwargs[:maxfreq] "maxfreq ($(filter_kwargs[:maxfreq])) is too high: lower samplerate in dataset is $min_sr (Nyquist freq: $(min_sr / 2))"
    end

    #############################################################
    ########## FILTER USING TREE FEATURES #######################
    #############################################################

    filtered = Vector{Vector{Float64}}(undef, n_instances)
    Threads.@threads for i in 1:n_instances
        if length(only_files) > 0 && !(wav_paths[i] in only_files)
            if verbose println("Skipping file $(wav_paths[i]) because it is not in the list...") end
            continue
        end
        if !is_correctly_classified(results[i])
            # TODO: handle not correctly classified instances
            if verbose println("Skipping file $(wav_paths[i]) because it was not correctly classified...") end
            continue
        end
        n_features = length(results[i].path)
        bands = Vector{Int64}(undef, n_features)
        weights = Vector{AbstractFloat}(undef, n_features)
        for j in 1:n_features
            if mode == :bandpass
                weights[j] = 1.0
            elseif mode == :emphasize
                # TODO: set a proper way to choose the threshold
                # TODO: real behaviour to emphasizes
                if (in(results[i].path[j].test_operator, [ >, >= ]) && results[i].path[j].taken) ||
                   (in(results[i].path[j].test_operator, [ <, <= ]) && !results[i].path[j].taken)
                    if results[i].path[j].threshold <= 1 # > than low
                        0.5
                    else # > than high
                        1.0
                    end
                else
                    if results[i].path[j].threshold <= 1 # < than low
                        0.25
                    else # < than high
                        0.5
                    end
                end
            end
            bands[j] = results[i].path[j].feature.i_attribute
        end
        println("Applying filter to file $(wav_paths[i]) with bands $(string(collect(zip(bands, weights))))...")
        if filter_type == :dynamic
            filter = dynamic_multibandpass_digitalfilter_mel(results[i].path, samplerates[i], floor(Int64, wintime * movingaverage_size * samplerates[i]), floor(Int64, steptime * movingaverage_step * samplerates[i]), nbands; filter_kwargs...)
            filtered[i] = apply_filter(filter, originals[i])
        elseif filter_type == :static
            filter = multibandpass_digitalfilter_mel(bands, samplerates[i], window_f; weights = weights, nbands = nbands, filter_kwargs...)
            filtered[i] = filt(filter, originals[i])
        end
    end

    #############################################################
    ######## APPLY POST-PROCESS AND SAVE FILTERED ###############
    #############################################################

    mk_tree_path(tree_hash, tree; path = destination_dir)

    real_destination = destination_dir * "/" * tree_hash
    mkpath(real_destination)
    heatmap_png_path = Vector{String}(undef, n_instances)
    json_lock = Threads.Condition()
    Threads.@threads for i in 1:n_instances
        if length(only_files) > 0 && !(wav_paths[i] in only_files)
            if verbose println("Skipping file $(wav_paths[i]) because it is not in the list...") end
            continue
        end
        if !is_correctly_classified(results[i])
            if verbose println("Skipping file $(wav_paths[i]) because it was not correctly classified...") end
            continue
        end
        save_path = replace(wav_paths[i], remove_from_path => "")
        if use_original_dataset_filesystem_tree
            while startswith(save_path, "../")
                save_path = replace(save_path, "../" => "")
            end
            save_dir = real_destination
        else
            save_dir = get_tree_path_as_dirpath(tree_hash, tree, results[i].path; path = destination_dir)
        end
        filtered_file_path = save_dir * "/" * replace(save_path, ".wav" => ".filt.wav")
        original_file_path = save_dir * "/" * replace(save_path, ".wav" => ".orig.wav")
        heatmap_png_path[i] = save_dir * "/" * replace(save_path, ".wav" => ".spectrogram.png")
        mkpath(dirname(filtered_file_path))
        for (i_pp, pp) in enumerate(postprocess_wavs)
            pp(filtered[i]; (postprocess_wavs_kwargs[i_pp])...)
            pp(originals[i]; (postprocess_wavs_kwargs[i_pp])...)
        end
        if files_already_generated
            continue
        end
        println("Saving filtered file $(filtered_file_path)...")
        wavwrite(filtered[i], filtered_file_path; Fs = samplerates[i])
        wavwrite(originals[i], original_file_path; Fs = samplerates[i])
        if generate_json
            if !haskey(json_archive, string(results[i].label))
                json_archive[string(results[i].label)] = []
            end
            Threads.lock(json_lock)
            push!(json_archive[string(results[i].label)], filtered_file_path)
            Threads.unlock(json_lock)
        end
    end

    #############################################################
    ######## GENERATE JSON WITH FILTERED WAV PATHS ##############
    #############################################################

    if generate_json && !files_already_generated
        try
            # TODO: handle "only_files" situation
            f = open(real_destination * "/" * json_file_name, "w")
            write(f, JSON.json(json_archive))
            close(f)
        catch
            println(stderr, "Unable to write file $(real_destination * "/" * json_file_name)")
        end
    end

    #############################################################
    ################ GENERATE SPECTROGRAMS ######################
    #############################################################

    spectrograms = Vector{NamedTuple{(:original, :filtered), Tuple{Plots.Plot, Plots.Plot}}}(undef, n_instances)
    if generate_spectrogram
        println("Generating spectrograms...")
        for i in 1:n_instances
            if length(only_files) > 0 && !(wav_paths[i] in only_files) continue end
            if !is_correctly_classified(results[i]) continue end
            println("Generating spectrogram $(heatmap_png_path[i])...")
            freqs =
                if spec_show_only_found_bands
                    [ path_node.feature.i_attribute for path_node in results[i].path ]
                else
                    :all
                end
            spectrograms[i] = (
                original = draw_spectrogram(originals[i], samplerates[i]; title = "Original", only_bands = freqs, spectrograms_kwargs...),
                filtered = draw_spectrogram(filtered[i], samplerates[i]; title = "Filtered", only_bands = freqs, spectrograms_kwargs...)
            )
            plot(spectrograms[i].original, spectrograms[i].filtered, layout = (1, 2))
            savefig(heatmap_png_path[i])
        end
    end

    #############################################################
    ######################## DRAW WORLDS ########################
    #############################################################

    if draw_worlds_on_file
        println("Drawing worlds...")
        for i in 1:n_instances
            if length(only_files) > 0 && !(wav_paths[i] in only_files) continue end
            if !is_correctly_classified(results[i]) continue end

            generic_track_name = basename(replace(heatmap_png_path[i], ".spectrogram.png" => ""))
            output_filename = replace(heatmap_png_path[i], ".spectrogram.png" => ".worlds.png")
            println("Drawing worlds $(output_filename)...")

            n_features = length(results[i].path)
            plts = []
            for j in 1:n_features
                feat = results[i].path[j].feature.i_attribute
                worlds = results[i].path[j].worlds

                winsize = round(Int64, (wintime * movingaverage_size * samplerates[i]))
                stepsize = round(Int64, (steptime * movingaverage_step * samplerates[i]))

                wav_descriptor, points, seconds = get_points_and_seconds_from_worlds(worlds, winsize, stepsize, length(originals[i]), samplerates[i])

                push!(plts, draw_worlds(wav_descriptor, samplerates[i]; title = "A$(feat)"))
            end
            plot(plts..., layout = (n_features, 1), size = (1000, 150 * n_features))
            savefig(output_filename)
        end
    end

    #############################################################
    ########### SIGNLE BANDS, ANIMATION AND VIDEO ###############
    #############################################################

    single_band_tracks = Vector{Vector{Tuple{Int64,Vector{Float64}}}}(undef, n_instances)
    if save_single_band_tracks
        println("Generating single band tracks...")
        for i in 1:n_instances
            if length(only_files) > 0 && !(wav_paths[i] in only_files) continue end
            if !is_correctly_classified(results[i]) continue end
            single_band_tracks[i] = Vector{Tuple{Int64,Vector{Float64}}}(undef, length(results[i].path))
            n_features = length(results[i].path)
            dyn_filter = 
                if filter_type == :dynamic
                    dynamic_multibandpass_digitalfilter_mel(results[i].path, samplerates[i], floor(Int64, wintime * movingaverage_size * samplerates[i]), floor(Int64, steptime * movingaverage_step * samplerates[i]), nbands; filter_kwargs...)
                else
                    nothing
                end
            for j in 1:n_features
                feat = results[i].path[j].feature.i_attribute
                one_band_sample =
                    if filter_type == :dynamic
                        apply_filter(dyn_filter, originals[i], feat)
                    elseif filter_type == :static
                        filt(multibandpass_digitalfilter_mel([feat], samplerates[i], window_f; nbands = nbands, filter_kwargs...), originals[i])
                    end
                single_band_tracks[i][j] = (feat, one_band_sample)
            end
        end
        features_colors = [ RGB(1 - (1 * (i/(nbands - 1))), 0, 1 * (i/(nbands - 1))) for i in 0:(nbands-1) ]
        for i in 1:n_instances
            if length(only_files) > 0 && !(wav_paths[i] in only_files) continue end
            if !is_correctly_classified(results[i]) continue end
            for (feat, samples) in single_band_tracks[i]
                save_path = replace(heatmap_png_path[i], ".spectrogram.png" => ".A$(feat).wav")
                println("Saving single band track to file $(save_path)...")
                wavwrite(samples, save_path; Fs = samplerates[i])
            end
            if i in draw_anim_for_instances
                gifout = replace(heatmap_png_path[i], ".spectrogram.png" => ".anim.gif")
                wws = 
                    if normalize_before_draw_anim
                        orig_norm = normalize!(deepcopy(originals[i]))
                        norm_rate = maximum(abs, orig_norm) / maximum(abs, originals[i])
                        collect(zip(
                            [ orig_norm, filtered[i] * norm_rate, [single_band_tracks[i][j][2] * norm_rate for j in 1:length(single_band_tracks[i])]... ],
                            fill(samplerates[i], 2 + length(single_band_tracks[i]))
                        ))
                    else
                        [ (originals[i], samplerates[i]), (filtered[i], samplerates[i]), [ (single_band_tracks[i][j][2], samplerates[i]) for j in 1:length(single_band_tracks[i]) ]... ]
                    end
                draw_wave_anim(
                    wws;
                    labels = [ "Original", "Filtered", [ string("A", single_band_tracks[i][j][1]) for j in 1:length(single_band_tracks[i]) ]... ],
                    colors = [ RGB(.3, .3, 1), RGB(1, .3, .3), features_colors[ [ b[1] for b in single_band_tracks[i] ] ]...],
                    outfile = gifout,
                    anim_kwargs...
                )
                if generate_video_from_gif
                    orig = replace(gifout, ".anim.gif" => ".orig.wav")
                    filt = replace(gifout, ".anim.gif" => ".filt.wav")
                    generate_video(gifout, [ orig, filt ]; video_kwargs...)
                end
            end
        end
    end

    return results, spectrograms
end
