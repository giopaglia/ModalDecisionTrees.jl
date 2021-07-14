function SplatEduardDataset(N)
	readDataset(filepath::String, ::Val{N}) where {N} = open(filepath, "r") do io
		insts = Array{Array{Float64}}[]
		labels = String[]

		numattributes = 0

		lines = readlines(io)
		for line ∈ lines
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

				push!(insts, numseries)
				push!(labels, class)
			end

			if numattributes === 0
				numattributes = length(series)
			end
		end
		(insts,labels)
		# domains = Array{Float64}[]
		# for j ∈ 1:numattributes
		# 	# sort all values for each timeserie (i.e., row of size(insts)[1] rows) for the j-th attribute; that is, compute the sorted domain of each attribute
		# 	push!(domains, sort!(collect(Set([val for i ∈ 1:size(insts)[1] for val ∈ insts.values[i][j]]))))
		# end

		# classes = collect(Set(map((a,b)->(b), insts)))
	end

	insts,classes = readDataset(data_dir * "test-Eduard/Train-$N.txt", Val(N))

	n_samples = length(insts)
	n_attributes = 2
	
	X_train = Array{Float64,3}(undef, N, n_samples, n_attributes)

	for i in 1:length(insts)
		X_train[:, i, :] .= hcat(insts[i]...)
	end

	Y_train = copy(classes)
	
	insts,classes = readDataset(data_dir * "test-Eduard/Test-$N.txt", Val(N))

	n_samples = length(insts)
	n_attributes = 2
	
	X_test = Array{Float64,3}(undef, N, n_samples, n_attributes)

	for i in 1:length(insts)
		X_test[:, i, :] .= hcat(insts[i]...)
	end

	Y_test = copy(classes)

	(X_train,Y_train),(X_test,Y_test)
	# [val for i ∈ 1:3 for val ∈ ds.values[i][:,1]] <-- prendo tutti i valori del primo attributo
	# sort!(collect(Set([val for i ∈ 1:3 for val ∈ ds.values[i][:,1]]))) <-- ordino il dominio
	# size(ds.values[1])    <-- dimensione della prima serie
end
