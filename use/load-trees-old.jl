include("scanner.jl")

using FileIO

input_directory = "./covid-august"
output_directory = "$(input_directory)/interesting-trees"

mkpath(output_directory)

trees_directory = "./covid-august/trees"
mkpath(trees_directory)
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

trees = [
	# (2,true),1
	"06c50dcd07e13172a0af62e573fcdb85f1a64e6a00d683dce4a02dea04b41e30",
	"eca519bde8f283382742da2805cf8073283cb659edb699226d3cf846b5e45a03",
	"19c8985bf7e163af09c3bafaf744d25670215aaf9a5763d9cc4dd7c6cfee8e0d",
	"116917b7dc8ed036f8a39636f6891b869799e25384c704592749db38b961e061",
	"1c94e85d0abbdb763ac2d8b8e20de0ba7e8c4e6b31a85d5ac2ed716c26d637a3",
	"8778a72bc08dc538c128729062e79de0b500d08988fbeba441390594d6a10ece",
	"df385e792e0fd6d99160967f955b052f193ed14d494f1790ba914cb89f64bada",
	"d5cce8625a82b7c1e5360a4d055175425302db80dc57e55695e32d4d782c6ac5",
	"f10b42fcb933c01ee82ed020ad68111dc6570720f5e935d36951a5c8c49ecf16",
	"10aca278db9bcdf5cf7e4ec532da3d1b06ec122dd62f948da7ac6ff4213c2fcc",
	"",
	# (2,true),2
	"37fba47129079631ac57d29887ea17fc4a838609d1c346bc51885c2a71316744",
	"52a2c3ce36108879fdce59c13541e197bc6cf0f85222c8b38e0ba4d2058348da",
	"8576e6cdad4bae34f78d005ef54e7cb28f9b874f11f7be4ef766a953cca08c8b",
	"a9308351c47eb02f5fc807f8f5837ef350d0bb512a62408cd4f1f164a8024929",
	"51cd914d4e3a89a119ccccca5f227b109cd22bd7ecc41c2b1be176e5a4cb29fd",
	"e53e0e0271e306f8fcb5f686396088b4c1922024ab1faf3230efac7908c4d73f",
	"c39e2ebeae1892478b1d2e96df956c8453d5245980620dabf148472027c1581a",
	"3d25ae3ec5b9a1d9c2a43b06ac52238637a8908307486e0e377841264d30d9cf",
	"",
	# (2,true),3
	"6d67d112bf381171a749b5352363171b3fc5d95671542bd904d64e0c2129c15d",
	"424356a52db67e3da39cf2a713e6e4382503e76d5a9d67cc6a06e9f1507cc3e8",
	"70801717b5bf277ad3655c0b5c26d009e4f4bedb874bcb6c7b6ee18286d31d18",
	"0720aaf0e1e902d1a446711f12685f0ccfc42d6583bc778ac01e8c975f854871",
	"80c2872b9f8d2af2936ca9d770b77c35a5faf360e9157084b3bcf21dc016d163",
	"8bde0f31c96ace7ba9d45d1b09cdb3b5a9de2e91d67e400bc23a2eef040773d9",
	"",
	# others
	"27cf73a1fc2cf6bae7f00b75c621a7cb1e910f03bdfb2fae4f667034d01dfab7",
	"",
	"49e923aa4d3698122b47cc8da3dfaa0864ca56332f71f1a7f9def3c7d0f5cfc0",
	"",
	"99d3943d1e1775a713be25cee3c73edc628d1190bb4e9122c41232c92f646fbd",
	"",
	"da8abc8bc9b68c42077a82016c38ab0a7549a8a3c0c269e01e458b5cb42d1766",
	"",
]

# For each tree
for tree_hash in trees

	println()
	println()

	if tree_hash == ""
		println("STOP")
		continue
	end

	tree = load("$(input_directory)/trees/tree_$(tree_hash).jld")["T"]

	# print_tree(tree, n_tot_inst = 226)
	print_tree(tree)
	# regenerated_tree = print_apply_tree(tree, stump_fmd, Y)

	# readline()
	# print_tree(regenerated_tree)
end

# # (dataset hash,stump_fmd hash) -> List of tree hashes to test against
# datasets_n_trees = Dict{Tuple{String,String},Vector{String}}(
# 	("9ecb9dee626e2a57ad035a211d693139f3e0063696cb67846075e40d0a343329", "d2f1e0557490c40d00d77685d5da9e77c4ff9626a3749e7a279cff0f65c20904") => [
# 	"e14b482914a90b3e159786d691dfe0e121306930832256a5ac65348391b770d7",
# ])

# # a = load_cached_obj("dataset", "$(input_directory)/cache", "9ecb9dee626e2a57ad035a211d693139f3e0063696cb67846075e40d0a343329")

# # For each dataset
# for ((dataset_hash,stump_fmd_hash),tree_hashes) in datasets_n_trees

# 	dataset = load_cached_obj("dataset", "$(input_directory)/cache", dataset_hash)
# 	(_,Y),_ = dataset
	
# 	stump_fmd = load_cached_obj("stump_fmd", "$(input_directory)/cache", stump_fmd_hash)
	
# 	# Test all of these trees
# 	for tree_hash in tree_hashes
		
# 		tree = load("$(input_directory)/trees/tree_$(tree_hash).jld")["T"]

# 		# print_tree(tree, n_tot_inst = 226)
# 		print_tree(tree)
# 		regenerated_tree = print_apply_tree(tree, stump_fmd, Y)

# 		# readline()
# 		# print_tree(regenerated_tree)
# 	end
# end

