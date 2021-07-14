using CSV
using DataFrames
using StatsBase

function ComParE2021Dataset(;
		subchallenge = "CCS",
		use_lowSR = true,
		mode = :development,
		include_static_data = false,
		treat_as_single_attribute_2D_context = false,
		#
		audio_kwargs,
		ma_size = 1,
		ma_step = 1,
		max_points = -1,
		use_full_mfcc = false,
		preprocess_wavs = [],
	)
	dataset_dir = data_dir * "ComParE2021-CCS-CSS-Data/"

	@assert mode in [:development, :testing]
	@assert subchallenge in ["CCS", "CSS"] "Unknown subchallenge: $(subchallenge)"
	
	@assert subchallenge == "CCS" "Currently, only subchalleng=CCS is supported."
	@assert include_static_data == false "Currently, static data is not supported"

	records = CSV.read("$(dataset_dir)metaData_$(subchallenge).csv", DataFrame)
	samples_folder = "$(dataset_dir)ComParE2021_$(subchallenge)/dist/wav/"

	to_ignore = use_lowSR ? [] : [row.filename_x for row in CSV.File("$(dataset_dir)lowSR_$(subchallenge).list")]

	is_to_ignore(filename) = begin
		filename in to_ignore ||
		(mode == :testing && (startswith(filename, "devel_") || startswith(filename, "train_"))) ||
		(mode == :development && startswith(filename, "test_"))
	end

	records = records[map(!,(is_to_ignore.(records.filename))), :]

	sort!(records, [:label], alg=Base.Sort.InsertionSort)

	# display(records)
	class_names = unique(map((record)->record[:label], eachrow(records)))

	class_counts = [class_name=>0 for class_name in class_names] |> Dict

	timeseries = Vector{Array{Float64, 2}}(undef, nrow(records))
	labels = Vector{String}(undef, nrow(records))

	max_ts_length         = []
	max_ts_with_ma_length = []
	max_ts_cut_length     = []

	# println(records |> typeof)
	# println(eachrow(records))
	# println(enumerate(eachrow(records)))
	Threads.@threads for (i_record,record) in collect(enumerate(eachrow(records)))
		# println(i_record)
		# println(record)
		# readline()
		filepath = "$(samples_folder)$(record[:filename])"
		label = record[:label]
		
		class_counts[label] +=1
		# println(filepath, label)

		ts = wav2stft_time_series(filepath, audio_kwargs; preprocess_sample = preprocess_wavs, use_full_mfcc = use_full_mfcc)

		push!(max_ts_length, size(ts,1))

		# display(ts)
		# ts = @views ts[:,2:end]
		# display(ts)
		ts = moving_average(ts, ma_size, ma_step)
		# display(ts)

		push!(max_ts_with_ma_length, size(ts,1))

		if max_points != -1 && size(ts,1)>max_points
			ts = ts[1:max_points,:]
		end
		# display(ts)
		# display(size(ts,1))
		
		push!(max_ts_cut_length, size(ts,1))
		
		timeseries[i_record] = ts
		labels[i_record] = label
		# readline()
	end

	println("max_ts_length         = (max = $(StatsBase.maximum(max_ts_length)), min = $(StatsBase.minimum(max_ts_length)), mean = $(StatsBase.mean(max_ts_length)), std = $(StatsBase.std(max_ts_length)))")
	println("max_ts_with_ma_length = (max = $(StatsBase.maximum(max_ts_with_ma_length)), min = $(StatsBase.minimum(max_ts_with_ma_length)), mean = $(StatsBase.mean(max_ts_with_ma_length)), std = $(StatsBase.std(max_ts_with_ma_length)))")
	println("max_ts_cut_length     = (max = $(StatsBase.maximum(max_ts_cut_length)), min = $(StatsBase.minimum(max_ts_cut_length)), mean = $(StatsBase.mean(max_ts_cut_length)), std = $(StatsBase.std(max_ts_cut_length)))")

	max_timepoints = maximum(size(ts, 1) for ts in timeseries)
	n_unique_freqs = unique(size(ts, 2) for ts in timeseries)
	@assert length(n_unique_freqs) == 1 "KDDDataset: length(n_unique_freqs) != 1: {$n_unique_freqs} != 1"
	n_unique_freqs = n_unique_freqs[1]
	X = zeros((max_timepoints, n_unique_freqs, length(timeseries)))
	for (i,ts) in enumerate(timeseries)
		# println(size(ts))
		X[1:size(ts, 1),:,i] = ts
	end

	if treat_as_single_attribute_2D_context
		X = reshape(X, (size(X)[1], size(X)[2], 1, size(X)[3]))
	end

	class_counts = [class_counts[label] for label in class_names] |> Tuple

	(X, labels), class_counts;
end
