
IndianPinesDataset() = begin
	X = matread(data_dir * "indian-pines/Indian_pines_corrected.mat")["indian_pines_corrected"]
	Y = matread(data_dir * "indian-pines/Indian_pines_gt.mat")["indian_pines_gt"]
	(X, Y) = map(((x)->round.(Int,x)), (X, Y))
end

SalinasDataset() = begin
	X = matread(data_dir * "salinas/Salinas_corrected.mat")["salinas_corrected"]
	Y = matread(data_dir * "salinas/Salinas_gt.mat")["salinas_gt"]
	(X, Y) = map(((x)->round.(Int,x)), (X, Y))
end

SalinasADataset() = begin
	X = matread(data_dir * "salinas-A/SalinasA_corrected.mat")["salinasA_corrected"]
	Y = matread(data_dir * "salinas-A/SalinasA_gt.mat")["salinasA_gt"]
	(X, Y) = map(((x)->round.(Int,x)), (X, Y))
end

PaviaCentreDataset() = begin
	X = matread(data_dir * "paviaC/Pavia.mat")["pavia"]
	Y = matread(data_dir * "paviaC/Pavia_gt.mat")["pavia_gt"]
	(X, Y) = map(((x)->round.(Int,x)), (X, Y))
end

PaviaDataset() = begin
	X = matread(data_dir * "paviaU/PaviaU.mat")["paviaU"]
	Y = matread(data_dir * "paviaU/PaviaU_gt.mat")["paviaU_gt"]
	(X, Y) = map(((x)->round.(Int,x)), (X, Y))
end

SampleLandCoverDataset(dataset::String,
												n_samples_per_label::Int,
												sample_size::Union{Int,NTuple{2,Int}}
												;
												stratify = true,
												n_attributes::Int = -1,
												flattened::Union{Bool,Symbol} = false,
												rng = Random.GLOBAL_RNG :: Random.AbstractRNG) = begin
	if sample_size isa Int
		sample_size = (sample_size, sample_size)
	end
	@assert isodd(sample_size[1]) && isodd(sample_size[2])
	(Xmap, Ymap), class_labels_map = 	if dataset == "IndianPines"
									IndianPinesDataset(),["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"]
								elseif dataset == "Salinas"
									SalinasDataset(),["Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow", "Fallow_rough_plow", "Fallow_smooth", "Stubble", "Celery", "Grapes_untrained", "Soil_vinyard_develop", "Corn_senesced_green_weeds", "Lettuce_romaine_4wk", "Lettuce_romaine_5wk", "Lettuce_romaine_6wk", "Lettuce_romaine_7wk", "Vinyard_untrained", "Vinyard_vertical_trellis"]
								elseif dataset == "Salinas-A"
									SalinasADataset(),Dict(1 => "Brocoli_green_weeds_1", 10 => "Corn_senesced_green_weeds", 11 => "Lettuce_romaine_4wk", 12 => "Lettuce_romaine_5wk", 13 => "Lettuce_romaine_6wk", 14 => "Lettuce_romaine_7wk")
								elseif dataset == "PaviaCentre"
									PaviaCentreDataset(),["Water", "Trees", "Asphalt", "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows", "Meadows", "Bare Soil"]
								elseif dataset == "Pavia"
									PaviaDataset(),["Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets", "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows"]
								else
									error("Unknown land cover dataset")
	end

	# print(size(Xmap))
	# readline()
	existingLabels = sort(filter!(l->l≠0, unique(Ymap)))
	# print(existingLabels)
	# readline()
	n_labels = length(existingLabels)

	if n_labels != length(class_labels_map)
		error("Unexpected number of labels in dataset: $(n_labels) != $(length(class_labels_map))")
	end

	class_counts = Dict(y=>0 for y in existingLabels)
	no_class_counts = 0
	for y in Ymap
		if y ≠ 0
			class_counts[y] += 1
		else
			no_class_counts += 1
		end
	end
	# println(zip(class_labels_map,class_counts) |> collect)
	# println(no_class_counts)
	# readline()

	class_is_to_ignore = Dict(y=>(class_counts[y] < n_samples_per_label) for y in existingLabels)

	if sum(values(class_is_to_ignore)) != 0
		# println("Warning! The following classes will be ignored in order to balance the dataset:")
		ignored_existingLabels = filter(y->(class_is_to_ignore[y]), existingLabels)
		non_ignored_existingLabels = map(y->!(class_is_to_ignore[y]), existingLabels)
		filter(y->(class_is_to_ignore[y]), existingLabels)
		# println([class_labels_map[y] for y in ignored_existingLabels])
		# println([class_counts[y] for y in ignored_existingLabels])
		n_labels = sum(non_ignored_existingLabels)
		# println(class_labels_map)
		# println(class_counts)
		# println(n_labels)
		# readline()
	end

	n_samples = n_samples_per_label*n_labels
	# print(n_samples)
	# cols = sort!([1:size(M,2);], by=i->(v1[i],v2[i]));

	X,Y = size(Xmap)[1], size(Xmap)[2]
	tot_attributes = size(Xmap)[3]
	inputs = Array{eltype(Xmap),4}(undef, sample_size[1], sample_size[2], n_samples, tot_attributes)
	labels = Vector{eltype(Ymap)}(undef, n_samples)
	sampled_class_counts = Dict(y=>0 for y in existingLabels)

	already_sampled = fill(false, X, Y)
	for i in 1:n_samples
		# print(i)
		while (x = rand(rng, 1:(X-sample_size[1])+1);
					y = rand(rng, 1:(Y-sample_size[2])+1);
					exLabel = Ymap[x+floor(Int,sample_size[1]/2),y+floor(Int,sample_size[2]/2)];
					exLabel == 0 || class_is_to_ignore[exLabel] || already_sampled[x,y] || sampled_class_counts[exLabel] == n_samples_per_label
					)
		end
		# print( Xmap[x:x+sample_size[1]-1,y:y+sample_size[2]-1,:] )
		# print( size(inputs[:,:,i,:]) )
		# readline()
		# print(label)
		# print(sampled_class_counts)
		# println(x,y)
		# println(x,x+sample_size[1]-1)
		# println(y,y+sample_size[2]-1)
		# println(already_sampled[x,y])
		# readline()

		inputs[:,:,i,:] .= Xmap[x:x+sample_size[1]-1,y:y+sample_size[2]-1,:]
		already_sampled[x,y] = true
		labels[i] = findfirst(x->x==exLabel, existingLabels)
		sampled_class_counts[exLabel] += 1
		# readline()
	end
	if n_attributes != -1
		# new_inputs = Array{eltype(Xmap),4}(undef, sample_size[1], sample_size[2], n_samples, n_attributes)
		n_attributes
		inputs = inputs[:,:,:,1:floor(Int, tot_attributes/n_attributes):tot_attributes]
		# new_inputs[:,:,:,:] = inputs[:,:,:,:]
		# inputs = new_inputs
	end

	if (sum(already_sampled) != n_samples)
		error("ERROR! Sampling failed! $(n_samples) $(sum(already_sampled))")
	end
	# println(labels)

	sp = sortperm(labels)
	labels = labels[sp]
	inputs = inputs[:,:,sp,:]
	
	if stratify
		labels = reshape(transpose(reshape(labels, (n_samples_per_label,n_labels))), n_samples)
		inputs = reshape(permutedims(reshape(inputs, (size(inputs, 1),size(inputs, 2),n_samples_per_label,n_labels,size(inputs, 4))), [1,2,4,3,5]), (size(inputs, 1),size(inputs, 2),n_samples,size(inputs, 4)))
	end

	if flattened != false
		if flattened == :flattened
			inputs = reshape(inputs, (n_samples,(size(inputs, 1)*size(inputs, 2)*size(inputs, 4))))
		elseif flattened == :averaged
			inputs = sum(inputs, dims=(1,2))./(size(inputs, 1)*size(inputs, 2))
			inputs = dropdims(inputs; dims=(1,2))
		else
			error("Unexpected value for flattened: $(flattened)")
		end
		inputs = permutedims(inputs, [2,1])
	elseif (size(inputs, 1), size(inputs, 2)) == (1, 1)
		inputs = dropdims(inputs; dims=(1,2))
		inputs = permutedims(inputs, [2,1])
	else
		inputs = permutedims(inputs, [1,2,4,3])
	end
	

	# println([class_labels_map[y] for y in existingLabels])
	# println(labels)
	class_labels = [class_labels_map[y] for y in existingLabels]
	ys = [class_labels[y] for y in labels]
	
	dataset = inputs, ys

	if stratify
		dataset
	else
		dataset, Tuple(fill(n_samples_per_label, n_labels))
	end
end
