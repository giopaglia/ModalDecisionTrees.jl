
simpleDataset(200,n_attributes = 50,rng = my_rng())
simpleDataset2(200,n_attributes = 5,rng = my_rng())

simpleDataset(n_samp::Int, N::Int, rng = Random.GLOBAL_RNG :: Random.AbstractRNG) = begin
	X = Array{Int,3}(undef, N, n_samp, 1);
	Y = Vector{String}(undef, n_samp);
	for i in 1:n_samp
		instance = fill(2, N)
		y = rand(rng, 0:1)
		if y == 0
			instance[3] = 1
		else
			instance[3] = 2
		end
		X[:,i,1] .= instance
		Y[i] = string(y)
	end
	(X,Y)
end

simpleDataset2(n_samp::Int, N::Int, rng = Random.GLOBAL_RNG :: Random.AbstractRNG) = begin
	X = Array{Int,3}(undef, N, n_samp, 1);
	Y = Vector{String}(undef, n_samp);
	for i in 1:n_samp
		instance = fill(0, N)
		y = rand(rng, 0:1)
		if y == 0
			instance[3] += 1
		else
			instance[1] += 1
		end
		X[:,i,1] .= instance
		Y[i] = string(y)
	end
	(X,Y)
end