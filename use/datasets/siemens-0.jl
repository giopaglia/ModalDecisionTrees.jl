SplatSiemensDataset(rseed::Int, nmeans::Int, hour::Int, distance::Int; subdir = "Siemens") = begin

	readDataset(filepath::String) = open(filepath, "r") do io
		insts = Array{Array{Float64}}[]
		# labels = Int64[]
		labels = String[]

		numattributes = 0

		flag = false

		lines = readlines(io)

		i = 1

		if flag == false
			while isempty(lines[i]) || split(lines[i])[1] != "@data"
				i = i + 1
				if !isempty(lines[i]) && split(lines[i])[1] == "@data"
					break
				end
			end
			flag = true
			i = i+1
		end

		for j = i:length(lines)
			# line = lines[j]
			# # println("line=" * line)
			# split_line = split(line, "\',\'")
			# split_line[1] = split_line[1][2:end]    # removing the first '
			#
			# # println("split_line[1]")
			# # println(split_line[1])
			# # println("split_line[2]")
			# # println(split_line[2])
			# # println("split_line[10]")
			# # println(split_line[10])
			# # series = split(split_line[1], "','")    # splitting the series based on \n
			# #class = split_line[11]                   # the class
			#
			# temp = split(split_line[10], "\',")
			# class = temp[2]
			# split_line[10] = temp[1]
			#
			# series = [split_line[i] for i ∈ 1:10]

			line = lines[j]
			split_line = split(line, "\',")
			split_line[1] = split_line[1][2:end]    # removing the first '
			series = split(split_line[1], "\\n")    # splitting the series based on \n
			class = split_line[2]                   # the class

			if length(series) ≥ 1
				numserie = [parse(Float64, strval) for strval ∈ split(series[1], ",")]
				numseries = Array{Float64}[]
				push!(numseries, numserie)

				for i ∈ 2:length(series)
					numserie = [parse(Float64, strval) for strval ∈ split(series[i], ",")]
					push!(numseries, numserie)
				end
				# @show class
				push!(insts, numseries)
				# push!(labels, parse(Int, class))
				push!(labels, class)
			end

			if numattributes === 0
				numattributes = length(series)
			end
		end

		(insts,labels)
	end

	fileWithPath = data_dir * subdir * "/TRAIN_seed" * string(rseed) * "_nMeans" * string(nmeans) * "_hour" * string(hour) * "_distanceFromEvent" * string(distance) * ".arff"
	println(fileWithPath)
	insts,classes = readDataset(fileWithPath)

	n_attributes = 10

	X_train = Array{Float64,3}(undef, nmeans*hour, length(insts), n_attributes)

	for i in 1:length(insts)
		# println()
		# println("i=" * string(i))
		# println("size(X_train)=" * string(size(X_train)))
		# println("size(insts[i])=" * string(size(insts[i])))
		# println("size(hcat(insts[i]...))=" * string(size(hcat(insts[i]...))))
		# println("size(X_train[:, i, :])=" * string(size(X_train[:, i, :])))
		X_train[:, i, :] .= hcat(insts[i]...)
	end

	Y_train = classes

	fileWithPath = data_dir * subdir * "/TEST_seed" * string(rseed) * "_nMeans" * string(nmeans) * "_hour" * string(hour) * "_distanceFromEvent" * string(distance) * ".arff"
	insts,classes = readDataset(fileWithPath)

	X_test = Array{Float64,3}(undef, nmeans*hour, length(insts), n_attributes)

	for i in 1:length(insts)
		X_test[:, i, :] .= hcat(insts[i]...)
	end

	Y_test = classes

	(X_train,Y_train),(X_test,Y_test)
end

SiemensDataset_not_stratified(nmeans::Int, hour::Int, distance::Int; subdir = "Siemens") = begin

	readDataset(filepath::String) = open(filepath, "r") do io
		insts = Array{Array{Float64}}[]
		labels = Int64[]

		numattributes = 0

		flag = false

		lines = readlines(io)

		i = 1

		if flag == false
			while isempty(lines[i]) || split(lines[i])[1] != "@data"
				i = i + 1
				if !isempty(lines[i]) && split(lines[i])[1] == "@data"
					break
				end
			end
			flag = true
			i = i+1
		end

		for j = i:length(lines)
			# line = lines[j]
			# # println("line=" * line)
			# split_line = split(line, "\',\'")
			# split_line[1] = split_line[1][2:end]    # removing the first '
			#
			# # println("split_line[1]")
			# # println(split_line[1])
			# # println("split_line[2]")
			# # println(split_line[2])
			# # println("split_line[10]")
			# # println(split_line[10])
			# # series = split(split_line[1], "','")    # splitting the series based on \n
			# #class = split_line[11]                   # the class
			#
			# temp = split(split_line[10], "\',")
			# class = temp[2]
			# split_line[10] = temp[1]
			#
			# series = [split_line[i] for i ∈ 1:10]

			line = lines[j]
			split_line = split(line, "\',")
			split_line[1] = split_line[1][2:end]    # removing the first '
			series = split(split_line[1], "\\n")    # splitting the series based on \n
			class = split_line[2]                   # the class

			if length(series) ≥ 1
				numserie = [parse(Float64, strval) for strval ∈ split(series[1], ",")]
				numseries = Array{Float64}[]
				push!(numseries, numserie)

				for i ∈ 2:length(series)
					numserie = [parse(Float64, strval) for strval ∈ split(series[i], ",")]
					push!(numseries, numserie)
				end
				# @show class
				push!(insts, numseries)
				push!(labels, parse(Int, class))
			end

			if numattributes === 0
				numattributes = length(series)
			end
		end

		(insts,labels)
	end

	fileWithPath = data_dir * subdir * "/nMeans" * string(nmeans) * "_hour" * string(hour) * "_distanceFromEvent" * string(distance) * ".arff"
	println(fileWithPath)
	insts,classes = readDataset(fileWithPath)

	n_attributes = 10

	X = Array{Float64,3}(undef, nmeans*hour, length(insts), n_attributes)
	Y = Vector{String}(undef, length(insts))

	pos_idx = []
	neg_idx = []
	for (i,class) in enumerate(classes)
		if class == 0
			push!(neg_idx, i)
		elseif class == 1
			push!(pos_idx, i)
		else
			error("Unknown class: $(class)")
		end
	end

	progressive_i = 1
	for i in [pos_idx..., neg_idx...]
		X[:, progressive_i, :] .= hcat(insts[i]...)
		Y[progressive_i] = (i == 0 ? "negative" : "positive")
		progressive_i+=1
	end

	((X,Y), length(pos_idx), length(neg_idx))
end
