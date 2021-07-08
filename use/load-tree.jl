include("runner.jl")

using FileIO

trees_directory = "./siemens/trees"
output_directory = "./siemens/results-load-trees"

mkpath(trees_directory)
mkpath(output_directory)

# tree_files = map(str -> trees_directory * "/" * str, filter!(endswith(".jld"), readdir(trees_directory)))

# function load_tree(file_name::String)::DTree
# 	println("Loading $(file_name)...")
# 	load(string(file_name))["T"]
# end

# function load_all_trees(file_names::Vector{String})::DTree
# 	arr = []
# 	for f in file_names
# 		push!(arr, load_tree(f))
# 	end
# 	arr
# end

# testing_tree = load_tree(tree_files[1])
# print_tree(testing_tree)

# seeds = [
# 	"7933197233428195239",
# 	"1735640862149943821",
# 	"3245434972983169324",
# 	"1708661255722239460",
# 	"1107158645723204584"
# ]

# configs = [
# 	"30.45.30",
# 	"30.75.50"
# ]

# tree_401a21a62325fa736d2c57381d6b6a1f63a92e665220ac3c63ccca89efe7fb0c
# tree_b7269f8be54c95a26094482fcd29b38d9611ddc2fcd9a14015cbdf4fab67aa28
# tree_72ce352289cb551233bc542db93dd90ab643b82e182e9b1a0d7fd532c479b4e2
# tree_014838ab30d9d70e2d5939ee4da07c21e07fe69960cbfdffe0348ecd6f167c71

# for seed in seeds
#     for config in configs

# for (dataset_str,tree_filepath) in [
# 	("1708661255722239460.1.1.40.(30.25.15)","tree_401a21a62325fa736d2c57381d6b6a1f63a92e665220ac3c63ccca89efe7fb0c"),
# 	("1708661255722239460.1.2.40.(30.45.30)","tree_b7269f8be54c95a26094482fcd29b38d9611ddc2fcd9a14015cbdf4fab67aa28"),
# 	("1708661255722239460.1.2.20.(30.75.50)","tree_72ce352289cb551233bc542db93dd90ab643b82e182e9b1a0d7fd532c479b4e2"),
# 	("1107158645723204584.1.2.20.(30.75.50)","tree_014838ab30d9d70e2d5939ee4da07c21e07fe69960cbfdffe0348ecd6f167c71")]

obj = Dict{String,Vector{String}}(
	"dataset_9ecb9dee626e2a57ad035a211d693139f3e0063696cb67846075e40d0a343329.jld" => [
	"tree_e14b482914a90b3e159786d691dfe0e121306930832256a5ac65348391b770d7.jld",
])
# For each dataset
for (dataset_filepath,tree_filepaths) in obj
	# Test all of these trees
	for tree_filepath in tree_filepaths
		dataset = nothing
		tree = nothing
		JLD2.@load "./results-audio-scan/datasets/$(dataset_filepath)" dataset
		tree = load("./results-audio-scan/trees/$(tree_filepath).jld")["T"]
		# JLD2.@load "./results-audio-scan/datasets/$(seed).1.1.60.($(config)).jld" dataset

		X, Y = dataset[1], dataset[2]
		# Y = 1 .- Y

		# print_tree(tree, n_tot_inst = 226)
		print_tree(tree)
		regenerated_tree = print_apply_tree(tree, X, Y)
		# readline()
		# print_tree(regenerated_tree)
	end
end

#     end
# end
