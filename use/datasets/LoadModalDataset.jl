using Base.Filesystem

function LoadModalDataset_v1(folder_name)
	
	dataset_dir = joinpath(data_dir, folder_name)

	function parse_metadata_file(filepath)
		d = Dict{String,Any}()
		for (i_line,line) in enumerate(readlines(filepath))
			pieces = rsplit(line, "=")
			@assert length(pieces) == 2 "Couldn't parse metadata file: $(filepath), line $(i_line)"
			# map!(s->strip(s, ['"', '\'']), pieces)
			pieces = map(s->strip(s, '"'), pieces)
			d[pieces[1]] = pieces[2]
		end
		d
	end

	metadata = parse_metadata_file(joinpath(dataset_dir, "Metadata.txt"))

	@assert haskey(metadata, "name")           "Couldn't find 'name' dataset property!"
	@assert haskey(metadata, "supervised")     "Couldn't find 'supervised' dataset property!"
	@assert haskey(metadata, "num_frames")     "Couldn't find 'num_frames' dataset property!"

	@assert metadata["name"]       isa AbstractString  "Property 'name' should be of type AbstractString, got $(typeof(metadata["name"]      )) instead!"
	@assert metadata["supervised"] isa AbstractString    "Property 'supervised' should be of type AbstractString, got $(typeof(metadata["supervised"])) instead!"
	@assert metadata["supervised"] in ["false", "true"]  "Property 'supervised' should be either 'false' or 'true', got '$(typeof(metadata["supervised"]      ))' instead!"
	metadata["supervised"] = (metadata["supervised"] == "true")
	metadata["num_frames"] = parse(Int, metadata["num_frames"])
	@assert metadata["num_frames"] isa Integer "Property 'num_frames' should be of type Integer, got $(typeof(metadata["num_frames"])) instead!"
	
	println(metadata)

	num_frames = metadata["num_frames"]
	frame_dims = Integer[]
	
	println("num_frames: $(num_frames)")

	for i_frame in 1:num_frames
		@assert haskey(metadata, "frame$(i_frame)")     "$(num_frames) frames are expected, but couldn't find 'frame$(i_frame)' dataset property!"
		metadata["frame$(i_frame)"] = parse(Int, metadata["frame$(i_frame)"])
		@assert metadata["frame$(i_frame)"] isa Integer  "Property 'frame$(i_frame)' should be of type Integer, got $(typeof(metadata["frame$(i_frame)"])) instead!"
		push!(frame_dims, metadata["frame$(i_frame)"])
	end
	
	INTuple = NTuple{N,Integer} where N
	
	@assert metadata["supervised"] == true "Only supervised datasets are currently allowed."

	instance_names = filter(isdir, readdir(dataset_dir, sort=true))

	println("Instances:")
	println(instance_names)
	println()

	instance_names_enumerator = 
		if metadata["supervised"]
			throw_n_log("TODO")
			# TODO read joinpath(dataset_dir, "Labels.csv") and associate each instance_name to its label
			# labels = ...TODO
			@assert length(instance_names) == length(labels)
			zip(instance_names,labels)
		else
			zip(instance_names,Iterators.repeated(nothing))
		end

	class_counts = Dict{String,Integer}()
	Y = String[]

	for (example_dir,label) in instance_names
		instance_metadata = parse_metadata_file(joinpath(dataset_dir, example_dir, "Metadata.txt"))
		println(instance_metadata)
		instance_frame_sizes = INTuple[]

		for i_frame in 1:num_frames
			@assert haskey(instance_metadata, "frame$(i_frame)")     "$(num_frames) frames are expected, but couldn't find 'frame$(i_frame)' dataset property!"
			@assert instance_metadata["frame$(i_frame)"] isa AbstractString  "Property 'frame$(i_frame)' should be of type AbstractString, got $(typeof(instance_metadata["frame$(i_frame)"])) instead!"
			instance_metadata["frame$(i_frame)"] = eval(Meta.parse(instance_metadata["frame$(i_frame)"]))
			if instance_metadata["frame$(i_frame)"] isa Integer
				instance_metadata["frame$(i_frame)"] = tuple(instance_metadata["frame$(i_frame)"])
			end
			@assert instance_metadata["frame$(i_frame)"] isa INTuple  "Property 'frame$(i_frame)' should be of type INTuple, got $(typeof(instance_metadata["frame$(i_frame)"])) instead!"
			push!(instance_frame_sizes, instance_metadata["frame$(i_frame)"])

			throw_n_log("TODO")
			# TODO leggi joinpath(dataset_dir, example_dir, "/Frame $(i_frame).csv") e carica i datasets!
		end

		if metadata["supervised"]
			if !haskey(class_counts, label)
				class_counts[label] = 0
			end
			class_counts[label] +=1
			push!(Y, label)
		end

	end

	# NOTE: assuming categorical values
	if metadata["supervised"]

		# TODO sort instances by class and derive class_names.
		class_counts = [class_counts[label] for label in class_names] |> Tuple

		(X,Y), class_counts
	else
		X
	end;
end

# LoadModalDataset_v1("Example-Dataset-v1")

# TODO give instances a name, different from "Example n"
# TODO give frames a name
# frame_names = String[]
# for i_frame in 1:num_frames
# 	@assert metadata["frame$(i_frame)"] isa String  "Property 'frame$(i_frame)' should be of type Integer, got $(typeof(instance_metadata["frame$(i_frame)"])) instead!"
# 	push!(frame_names, metadata["frame$(i_frame)"])
# end
# for frame_name in frame_names
# 	"$(frame_name).csv"
# end

# LoadModalDataset_v2("Example-Dataset-v2")
