
################################################################################
################################################################################
################################################################################
# task 1: YES/NO_CLEAN_HISTORY_AND_LOW_PROBABILITY
#   ( 66 user (141 sample) / 220 users (298 samples) in total)
# - v1: USING COUGH
# - v2: USING BREATH
# - v3: USING COUGH + BREATH
# task 2: YES_WITH_COUGH/NO_CLEAN_HISTORY_AND_LOW_PROBABILITY
#   ( 23 user (54 sample) / 29 users (32 samples) in total)
# - v1: USING COUGH
# - v2: USING BREATH
# - v3: USING COUGH + BREATH
# task 3: YES_WITH_COUGH/NO_CLEAN_HISTORY_AND_LOW_PROBABILITY_WITH_ASTHMA_AND_COUGH_REPORTED
#   ( 23 user (54 sample) / 18 users (20 samples) in total)
# - v1: USING COUGH
# - v2: USING BREATH
# - v3: USING COUGH + BREATH

function KDDDataset_not_stratified((n_task,n_version),
		audio_kwargs; ma_size = 1,
		ma_step = 1,
		max_points = -1,
		use_full_mfcc = false,
		preprocess_wavs = [],
		use_augmentation_data = true,
		# rng = Random.GLOBAL_RNG :: Random.AbstractRNG
	)
	@assert n_task in [1,2,3] "KDDDataset: invalid n_task: {$n_task}"
	@assert n_version in [1,2,3] "KDDDataset: invalid n_version: {$n_version}"

	kdd_data_dir = data_dir * "KDD/"
	
	task_to_folders = [
		[
			["covidandroidnocough", "covidandroidwithcough", "covidwebnocough", "covidwebwithcough"],
			["healthyandroidnosymp", "healthywebnosymp"],
			["YES", "NO_CLEAN_HISTORY_AND_LOW_PROBABILITY"]
		],
		[
			["covidandroidwithcough", "covidwebwithcough"],
			["healthyandroidwithcough", "healthywebwithcough"],
			["YES_WITH_COUGH", "NO_CLEAN_HISTORY_AND_LOW_PROBABILITY"]
		],
		[
			["covidandroidwithcough", "covidwebwithcough"],
			["asthmaandroidwithcough", "asthmawebwithcough"],
			["YES_WITH_COUGH", "NO_CLEAN_HISTORY_AND_LOW_PROBABILITY_WITH_ASTHMA_AND_COUGH_REPORTED"]
		],
	]

	has_subfolder_structure = ["asthmawebwithcough", "covidwebnocough", "covidwebwithcough", "healthywebnosymp", "healthywebwithcough"]

	files_to_ignore = [
		# Missing
		"asthmawebwithcough/2020-04-07-18_49_21_155697/audio_file_breathe.wav_aug_amp1.wav",
		# Ignore to square up
		"asthmawebwithcough/2020-04-07-18_49_21_155697/audio_file_cough.wav_aug_amp1.wav"
	]

	has_augmentation_data = [
		"healthyandroidwithcough",
		"healthywebwithcough",
		"asthmaandroidwithcough",
		"asthmawebwithcough",
	]
	augmentation_file_suffixes = [
		"",
		"_aug_amp1.wav",
		"_aug_amp2.wav",
		"_aug_noise1.wav",
		"_aug_noise2.wav",
		"_aug_pitchspeed1.wav",
		"_aug_pitchspeed2.wav",
	]

	cough = ("cough","cough","cough_")
	breath = ("breath","breathe","breaths_")
	dir_infos =
		if n_version == 1
			(cough,)
		elseif n_version == 2 
			(breath,)
		else
			(cough, breath)
		end

	folders_Y, folders_N, class_labels = task_to_folders[n_task]

	
	function readFiles(folders, subfolder, file_suffix, file_prefix)
		files_map = JSON.parsefile(kdd_data_dir * "files.json")
		# println(folders)
		# https://stackoverflow.com/questions/59562325/moving-average-in-julia
		moving_average(vs::AbstractArray{T,1},n,st=1) where {T} = [sum(@view vs[i:(i+n-1)])/n for i in 1:st:(length(vs)-(n-1))]
		moving_average(vs::AbstractArray{T,2},n,st=1) where {T} = mapslices((x)->(@views moving_average(x,n,st)), vs, dims=2)
		# (sum(w) for w in partition(1:9, 3, 2))
		# moving_average_np(vs,num_out_points,st) = moving_average(vs,length(vs)-num_out_points*st+1,st)
		# moving_average_np(vs,num_out_points,o) = (w = length(vs)-num_out_points*(1-o/w)+1; moving_average(vs,w,1-o/w))
		# moving_average_np(vs,t,o) = begin
		# 	N = length(vs);
		# 	s = floor(Int, (N+1)/(t+(1/(1-o))))
		# 	w = ceil(Int, s/(1-o))
		# 	# moving_average(vs,w,1-ceil(Int, o/w))
		# end
		n_samples = 0
		# timeseries = Vector{Vector{Array{Float64, 2}}}[]
		timeseries = []
		for folder in folders

			# Groups of filepaths
			file_paths = files_map[folder]
			
			# Correct with augmentated data (if present)
			file_paths = 
				if use_augmentation_data && folder in has_augmentation_data && ! (folder in has_subfolder_structure)
					map((file_path_arr)->["$(file_path)$(suff)" for suff in augmentation_file_suffixes for file_path in file_path_arr], file_paths)
				else
					file_paths
				end

			cur_folder_timeseries = Vector{Vector{Array{Float64, 2}}}(undef, length(file_paths))

			# collect is necessary because the threads macro only supports arrays
			# https://stackoverflow.com/questions/57633477/multi-threading-julia-shows-error-with-enumerate-iterator
			Threads.@threads for (i_samples, samples) in collect(enumerate(file_paths))

				# Correct with augmentated data (if present)
				samples = 
					if folder in has_subfolder_structure
						samples = map((subfoldname)->"$folder/$subfoldname/audio_file_$(file_suffix).wav", samples)
						if use_augmentation_data && folder in has_augmentation_data
							samples = ["$(file_path)$(suff)" for suff in augmentation_file_suffixes for file_path in samples]
						end
						samples
					else
						# println(samples)
						filter!((filename)->startswith(filename,file_prefix), samples)
						map((filename)->"$folder/$subfolder/$filename", samples)
					end
				# println(samples)

				# Filter out files that are known to be missing
				filter!((filename)->! (filename in files_to_ignore), samples)

				cur_file_timeseries = Vector{Array{Float64, 2}}(undef, length(samples))
				valid_i_filenames = []
				for (i_filename, filename) in collect(enumerate(samples))
					# println(filename)
					filepath = kdd_data_dir * "$filename"
					ts = wav2stft_time_series(filepath, audio_kwargs; preprocess_sample = preprocess_wavs, use_full_mfcc = use_full_mfcc)
					# Ignore instances with NaN (careful! this may leave with just a few instances)
					# if any(isnan.(ts))
					# 	@warn "Instance with NaN values was ignored"
					# 	continue
					# end
					ts = moving_average(ts, ma_size, ma_step)
					# Drop first point
					ts = @views ts[:,2:end]
					# println(size(ts))
					if max_points != -1 && size(ts,2)>max_points
						ts = ts[:,1:max_points]
					end
					# println(size(ts))
					# readline()
					# println(size(wav2stft_time_series(filepath, audio_kwargs)))
					cur_file_timeseries[i_filename] = ts
					push!(valid_i_filenames, i_filename)
					n_samples += 1
				end
				cur_folder_timeseries[i_samples] = cur_file_timeseries[valid_i_filenames]
				# break
			end
			append!(timeseries, [j for i in cur_folder_timeseries for j in i])
		end
		# @assert n_samples == tot_n_samples "KDDDataset: unmatching tot_n_samples: {$n_samples} != {$tot_n_samples}"
		# timeseries[1:5]
		timeseries
	end

	function getTimeSeries(folders::NTuple{N,AbstractVector{String}}, dir_infos::NTuple{3,String}) where N
		subfolder,file_suffix,file_prefix = dir_infos
		pos = readFiles(folders[1], subfolder, file_suffix, file_prefix)
		neg = readFiles(folders[2], subfolder, file_suffix, file_prefix)

		println("POS={$(length(pos))}, NEG={$(length(neg))}")
		#n_per_class = min(length(pos), length(neg))

		#pos = pos[Random.randperm(rng, length(pos))[1:n_per_class]]
		#neg = neg[Random.randperm(rng, length(neg))[1:n_per_class]]

		#println("Balanced -> {$n_per_class}+{$n_per_class}")

		# Stratify
		# timeseries = vec(hcat(pos,neg)')
		# Y = vec(hcat(ones(Int,length(pos)),zeros(Int,length(neg)))')

		# print(size(pos))
		# print(size(neg))
		timeseries = [[p' for p in pos]..., [n' for n in neg]...]
		# print(size(timeseries))
		# print(size(timeseries[1]))
		# Y = [ones(Int, length(pos))..., zeros(Int, length(neg))...]
		# Y = [zeros(Int, length(pos))..., ones(Int, length(neg))...]
		Y = [fill(class_labels[1], length(pos))..., fill(class_labels[2], length(neg))...]
		# print(size(Y))

		# println([size(ts, 1) for ts in timeseries])
		max_timepoints = maximum(size(ts, 1) for ts in timeseries)
		n_unique_freqs = unique(size(ts, 2) for ts in timeseries)
		@assert length(n_unique_freqs) == 1 "KDDDataset: length(n_unique_freqs) != 1: {$n_unique_freqs} != 1"
		n_unique_freqs = n_unique_freqs[1]
		X = zeros((max_timepoints, n_unique_freqs, length(timeseries)))
		for (i,ts) in enumerate(timeseries)
			# println(size(ts))
			X[1:size(ts, 1),:,i] = ts
		end
		
		((X,Y), length(pos), length(neg))
	end

	function getTimeSeries(folders::NTuple{N,AbstractVector{String}}, dir_infos::NTuple{M,NTuple{3,String}}) where {N,M}
		datasets = Vector(undef, length(dir_infos))
		n_pos = Vector(undef, length(dir_infos))
		n_neg = Vector(undef, length(dir_infos))
		for (i, dir_info) in enumerate(dir_infos)
			datasets[i],n_pos[i],n_neg[i] = getTimeSeries(folders, dir_info)
			@assert datasets[1][2] == datasets[i][2] "mismatching classes:\n\tY1 = $(datasets[1][2])\n\tY2 = $(datasets[i][2])"
		end

		#@assert length(unique(n_pos)) == 1 "n_pos mismatch across frames: $(length.(n_pos))"
		#@assert length(unique(n_neg)) == 1 "n_neg mismatch across frames: $(length.(n_neg))"

		((getindex.(datasets, 1),datasets[1][2]), (n_pos[1], n_neg[1]))
	end

	getTimeSeries((folders_Y, folders_N), dir_infos)
end
################################################################################
################################################################################
################################################################################
