include("scanner.jl")

using FileIO

results_dir = "./covid-august-v2"
data_savedir = "$(results_dir)/cache"
trees_directory = "$(results_dir)/trees"

output_directory = "$(results_dir)/interesting-trees"

mkpath(output_directory)
mkpath(trees_directory)

################################################################################
################################################################################
################################################################################

# tree_files = map(str -> trees_directory * "/" * str, filter!(endswith(".jld"), readdir(trees_directory)))
test_operators = [TestOpGeq_80, TestOpLeq_80]
ontology       = getIntervalOntologyOfDim(Val(1))

# max_sample_rate = 48000
max_sample_rate = 16000
# max_sample_rate = 8000
nbands = 40
use_full_mfcc = false

dataset_kwargs =  (
						 	ma_size = 100,
						 	ma_step = 75,
						)

preprocess_wavs = 
	# ["Normalize"],
	# [],
	["NG", "Normalize"]


audio_kwargs_partial_mfcc = (
	wintime = 0.025, # in ms          # 0.020-0.040
	steptime = 0.010, # in ms         # 0.010-0.015
	fbtype = :mel,                    # [:mel, :htkmel, :fcmel]
	window_f = DSP.hamming, # [DSP.hamming, (nwin)->DSP.tukey(nwin, 0.25)]
	pre_emphasis = 0.97,              # any, 0 (no pre_emphasis)
	nbands = 40,                      # any, (also try 20)
	sumpower = false,                 # [false, true]
	dither = false,                   # [false, true]
	# bwidth = 1.0,                   # 
	# minfreq = 0.0,
	maxfreq = max_sample_rate/2, # Fix this
	# maxfreq = (sr)->(sr/2),
	# usecmp = false,
)

audio_kwargs_full_mfcc = (
	wintime=0.025,
	steptime=0.01,
	numcep=13,
	lifterexp=-22,
	sumpower=false,
	preemph=0.97,
	dither=false,
	minfreq=0.0,
	maxfreq = max_sample_rate/2, # Fix this
	# maxfreq=sr/2,
	nbands=20,
	bwidth=1.0,
	dcttype=3,
	fbtype=:htkmel,
	usecmp=false,
	modelorder=0
)


cur_audio_kwargs = merge(
	if use_full_mfcc
		audio_kwargs_full_mfcc
	else
		audio_kwargs_partial_mfcc
	end
	, (nbands=nbands,))



preprocess_wavs_dict = Dict(
	"NG" => noise_gate!,
	"Normalize" => normalize!,
)


cur_preprocess_wavs = [ preprocess_wavs_dict[k] for k in preprocess_wavs ]

################################################################################
################################################################################
################################################################################

dataset_function = (
	((n_task,use_aug),
		n_version,
		cur_audio_kwargs,
		dataset_kwargs,
		cur_preprocess_wavs,
		use_full_mfcc,)->
	KDDDataset_not_stratified(
		(n_task,n_version),
		cur_audio_kwargs;
		dataset_kwargs...,
		preprocess_wavs = cur_preprocess_wavs,
		use_full_mfcc = use_full_mfcc,
		return_filepaths = true,
		# use_augmentation_data = false,
		# force_monolithic_dataset = true,
		use_augmentation_data = use_aug,
		force_monolithic_dataset = :train_n_test,
	)
)

################################################################################
################################################################################
################################################################################

dataset_to_trees = [
	((
		(2,true),
		1,
		cur_audio_kwargs,
		dataset_kwargs,
		cur_preprocess_wavs,
		use_full_mfcc,
	), [
	"f1d6649dc960b07979a8cebe06a4f6b06061f64acb3b3aa880c2da62e45bb501",
	"c521f47f19e6a7c1bbcd2cce77461e86451e8f1b30ad3a0ecac28de995cd99d0",
	"4ef0f59238c19fb138b7bfae70e969f5c4d30074dde2e6af9cd4539957aeec21",
	"d655f61138131d05dea6debb3f8405feb727447d0d978f66271b8f3bb0adbcc8",
	"c31eb8ef897c0ae83c203bbbaf7947fb1cfad5de9969229099bb7d0a4d94b84c",
	"60872a3b4e149c5cac1eeb859217451cfaef6ab13c02d3cef1c3e50d49ef25c1",
	"3d6af89d7c21da6daec1c85284c7bbb72a2b1ffa7d3ecb4a1daaa878e2dea536",
	"006adae36df611a86333904d0293e10a98ed871401c4e4882dd4f6fce557441b",
	"726fd20bb146f11bf5b592bf8a5a073fad09dee7998e099ccce12db001a02a16",

	]), ((
		(2,true),
		2,
		cur_audio_kwargs,
		dataset_kwargs,
		cur_preprocess_wavs,
		use_full_mfcc,
	), [
	"ad35bdc78212e9112ce5ddfded3ece8ba2aad2fa803f6047661caef6676e2cb3",
	"1921e8cf8ab0c454912d31e805a2583dfcd49aa52f9257d4b6706df70018917b",
	"ad81fcd26440cf577b5883153fa3d0f1853b1f83e943032420e7c0f12747c2c3",
	"e4ec7be70c69b04071aa0dcde23f7d8f85e37bcfebb0e50c27501ad8f8fff03e",
	"a52f08b4473fa39c9e6731a778238d2ec4b81786078b2578c8e3e55e3a7565a0",
	"905b64dae6fe5cc5e557a71f3943a5065d779463633192b3c6bc5b28ea2d3685",
	"c5b6ef208dce8efa1ee6eccd20bfa5e668a100fb03b995c518406ebee4472b55",
	"f54ad9ddb18b735db460bd2ed8a616fab192b0ae8759ae3c5d4627f5b7bfc369",
	"479895a95bfe20561029a305080219640cd1eceb4bf2765d5645a4dc996ba529",
	"9ebc93efe84536aa1e11ffc7987cf29258ffefdb7f13b9e0359c6a529f4ca227",

	]), ((
		(2,true),
		3,
		cur_audio_kwargs,
		dataset_kwargs,
		cur_preprocess_wavs,
		use_full_mfcc,
	), [
	# "5d55e2939bf4cfe3486a6f0684b39182e2743d8408f9fe6f2bfdde2aad5a752b",
	# "3a228177284b56b21162b380a870902a084f0f4de3fee66a9fe167faddcb84ea",
	# "27ae9963f29c56d9a227eaf22627b64401518742a2f53d944ccf0207e335f67d",
	# "da51d66c6d064d666049137fdff5f0ea2182a6208a3b19cc7e9d2c3c576cb5af",
	# "55cf86c487e4ecea2a921b244d1a86636ea3a75ac24ffa96bc54f46f24f12163",
	# "970f0d21b6624959c22d67ee99038eb17981fdaf1be181e648cd400ad680987a",
	# "2eb6209b99fa2e75bfc0d6bc27ca59fcbdb13594f77819901b9277d0ffbd9035",
	# "b0b0afb6a6eb2b7ebb2023aa459c901a60b77b08357b9512191411b79552b93f",
	# "0364adc2159fa25f440cab268faa2c03e2dc11b8408884392dead7fd92b8f868",
	# "a37fd5c4441cdbd96f5200b0ff39fbbe554172b28ac678c04f25d926f0e7d744",

	]), ((
		(3,true),
		1,
		cur_audio_kwargs,
		dataset_kwargs,
		cur_preprocess_wavs,
		use_full_mfcc,
	), [
	"96b9e3dd5e389037f65300f4b59174d6885b6b568f8021c8d36fcf094a9f3e8d",
	"9c5e1bbecf74cedc8869ddcdd7553f4c09db73f90896fdd331aba851f84781f7",

	]), ((
		(3,true),
		2,
		cur_audio_kwargs,
		dataset_kwargs,
		cur_preprocess_wavs,
		use_full_mfcc,
	), [
	"5e99de5ba4df5ce36a002b3aa870a1786a47c9ead6abfad62fd9c8052f51ac0c",
	"fe3a51880fbfe7ee9b953196cfeb0af98f5d4aed70d14ec141b25ec4bde634ba",
	"2ee113048b00da5aeedd2e78af4d77c3de4a0f9fc1402239d6f8255f8b0579ca",
	"18a0a6b44d3d2745244b44c14b4b47072b3aaf78bc70bc92dec69e1f8dbf1756",

	]), ((
		(3,true),
		3,
		cur_audio_kwargs,
		dataset_kwargs,
		cur_preprocess_wavs,
		use_full_mfcc,
	), [
	# "1a38a8a876927761c496115ff3120b64d2e0111b6aeaac5ae8dd9965ce68dff0",
	# "3cdbb1e510193eda537fa001aed51f6efddd70b9c7ff25f5c0544b3c45aa0cc9",
	# "8a23e17e2904e44683d1c4e276eda0210edc4fbeca66e0154ec566a8589a40d7",
	# "22208529c86258f070daa630cdcd21926012d9a77dae302309d6659492344b40",

	])
]

################################################################################
################################################################################
################################################################################


MakeOntologicalDataset(Xs) = begin
	MultiFrameModalDataset([
		begin
			features = FeatureTypeFun[]

			for i_attr in 1:n_attributes(X)
				for test_operator in test_operators
					if test_operator == TestOpGeq
						push!(features, ModalLogic.AttributeMinimumFeatureType(i_attr))
					elseif test_operator == TestOpLeq
						push!(features, ModalLogic.AttributeMaximumFeatureType(i_attr))
					elseif test_operator isa _TestOpGeqSoft
						push!(features, ModalLogic.AttributeSoftMinimumFeatureType(i_attr, test_operator.alpha))
					elseif test_operator isa _TestOpLeqSoft
						push!(features, ModalLogic.AttributeSoftMaximumFeatureType(i_attr, test_operator.alpha))
					else
						error("Unknown test_operator type: $(test_operator), $(typeof(test_operator))")
					end
				end
			end

			featsnops = Vector{<:TestOperatorFun}[
				if any(map(t->isa(feature,t), [AttributeMinimumFeatureType, AttributeSoftMinimumFeatureType]))
					[≥]
				elseif any(map(t->isa(feature,t), [AttributeMaximumFeatureType, AttributeSoftMaximumFeatureType]))
					[≤]
				else
					error("Unknown feature type: $(feature), $(typeof(feature))")
					[≥, ≤]
				end for feature in features
			]

			OntologicalDataset(X, ontology, features, featsnops)
		end for X in Xs])
end


################################################################################
################################################################################
################################################################################

# For each tree
for (dataset_fun_sub_params,trees) in dataset_to_trees
	
	if length(trees) == 0
		continue
	end
	
	println()
	println()
	println()
	println()
	println(dataset_fun_sub_params)

	dataset = @cachefast "dataset" data_savedir dataset_fun_sub_params dataset_function
	# dataset = dataset_function(dataset_fun_sub_params...)
	
	(X, Y, filepaths), (n_pos, n_neg) = dataset
	
	# TODO should not need these at test time. Instead, extend functions so that one can use a MatricialDataset instead of an OntologicalDataset
	X = MakeOntologicalDataset(X)


	for tree_hash in trees

		println()
		println()
		println("Loading tree...")
		println(tree_hash)
		
		T = load("$(trees_directory)/tree_$(tree_hash).jld")["T"]

		# print_tree(tree, n_tot_inst = 226)
		print_tree(T)

		preds = apply_tree(T, X);
		cm = confusion_matrix(Y, preds)

		println()

		# regenerated_tree = print_apply_tree(T, X, Y)
		regenerated_tree = print_apply_tree(T, X, Y; print_relative_confidence = true)

		println(cm)
		
		# readline()
		# print_tree(regenerated_tree)
	end
	# readline()
end
