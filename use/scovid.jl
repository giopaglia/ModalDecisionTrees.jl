################################################################################
################################################################################
################################## Scan script #################################
################################################################################
################################################################################

include("scanner.jl")

train_seed = 2

################################################################################
#################################### FOLDERS ###################################
################################################################################

results_dir = "./covid/journal-v8-base_freq"

iteration_progress_json_file_path = results_dir * "/progress.json"
data_savedir  = results_dir * "/cache"
model_savedir = results_dir * "/trees"

# dry_run = false
# dry_run = :dataset_only
# dry_run = :model_study
dry_run = true

skip_training = false

# save_datasets = true
save_datasets = false

perform_consistency_check = false

################################################################################
##################################### TREES ####################################
################################################################################

# Optimization arguments for single-tree
tree_args = [
#	(
#		loss_function = DecisionTree.util.entropy,
#		min_samples_leaf = 1,
#		min_purity_increase = 0.01,
#		min_loss_at_leaf = 0.6,
#	)
]

for loss_function in [DecisionTree.util.entropy]
	for min_samples_leaf in [2,4] # [1,2]
		for min_purity_increase in [0.01] # [0.01, 0.001]
			for min_loss_at_leaf in [0.4, 0.5, 0.6] # [0.4, 0.6]
				push!(tree_args, 
					(
						loss_function       = loss_function,
						min_samples_leaf    = min_samples_leaf,
						min_purity_increase = min_purity_increase,
						min_loss_at_leaf    = min_loss_at_leaf,
						perform_consistency_check = perform_consistency_check,
					)
				)
			end
		end
	end
end

println(" $(length(tree_args)) trees")

################################################################################
#################################### FORESTS ###################################
################################################################################

forest_runs = 5
optimize_forest_computation = false
# optimize_forest_computation = true


forest_args = []

# for n_trees in [51,101]
for n_trees in [51]
	for n_subfeatures in [half_f]
		for n_subrelations in [id_f]
			for partial_sampling in [0.7]
				push!(forest_args, (
					n_subfeatures       = n_subfeatures,
					n_trees             = n_trees,
					partial_sampling    = partial_sampling,
					n_subrelations      = n_subrelations,
					# Optimization arguments for trees in a forest (no pruning is performed)
					loss_function       = DecisionTree.util.entropy,
					# min_samples_leaf    = 1,
					# min_purity_increase = 0.0,
					# min_loss_at_leaf    = 0.0,
					perform_consistency_check = perform_consistency_check,
				))
			end
		end
	end
end


println(" $(length(forest_args)) forests " * (length(forest_args) > 0 ? "(repeated $(forest_runs) times)" : ""))

################################################################################
################################## MODAL ARGS ##################################
################################################################################

modal_args = (;
	initConditions = DecisionTree.startWithRelationGlob,
	# initConditions = DecisionTree.startAtCenter,
	# useRelationGlob = true,
	useRelationGlob = false,
)

data_modal_args = (;
	# ontology = getIntervalOntologyOfDim(Val(1)),
	# ontology = getIntervalOntologyOfDim(Val(2)),
	# ontology = Ontology{ModalLogic.Interval}([ModalLogic.IA_A]),
	# ontology = Ontology{ModalLogic.Interval}([ModalLogic.IA_A, ModalLogic.IA_L, ModalLogic.IA_Li, ModalLogic.IA_D]),
)


################################################################################
##################################### MISC #####################################
################################################################################

# log_level = Logging.Warn
log_level = DecisionTree.DTOverview
# log_level = DecisionTree.DTDebug
# log_level = DecisionTree.DTDetail

# timing_mode = :none
timing_mode = :time
# timing_mode = :btime
#timing_mode = :profile

round_dataset_to_datatype = false
# round_dataset_to_datatype = UInt8
# round_dataset_to_datatype = UInt16
# round_dataset_to_datatype = UInt32
# round_dataset_to_datatype = UInt64
# round_dataset_to_datatype = Float16
# round_dataset_to_datatype = Float32
# round_dataset_to_datatype = Float64

split_threshold = 0.8
# split_threshold = 1.0
# split_threshold = false

# use_training_form = :dimensional
# use_training_form = :fmd
# use_training_form = :stump
# use_training_form = :stump_with_memoization

test_flattened = false
test_averaged  = false

legacy_gammas_check = false
# legacy_gammas_check = true

prefer_nonaug_data = true

################################################################################
##################################### SCAN #####################################
################################################################################

# For generating KDD-partitioned dataset
# generate_KDD_partitioned() = begin
# 	include("wav-filtering.jl")
# 	process_dataset("KDD", KDD_getSamplesList(; rel_path = true);
# 	out_dataset_dir_name = "KDD-norm-partitioned",
# 	partition_instances = true,
# 	partitioning_kwargs = (
# 			cut_original        = Val(true),
# 			preprocess          = Function[noise_gate!, normalize!],
# 			# preprocess_kwargs   = NamedTuple[ (level = 0.005,), (level = 1.0,) ],
# 			postprocess         = Function[],
# 		)
# 	)
# 	run(`rm "../datasets/KDD-norm-partitioned/covidwebnocough/2020-04-26-16_20_02_616687/audio_file_cough.wav-split.1.wav"`)
# end


# # For generating KDD-partitioned dataset
# generate_KDD_partitioned() = begin
# 	include("wav-filtering.jl")
#   process_dataset("KDD", KDD_getSamplesList(; rel_path = true);
#   out_dataset_dir_name = "KDD-partitioned",
#   partition_instances = true,
#   partitioning_kwargs = (
#       cut_original = Val(true),
#       preprocess   = Function[],
#       postprocess  = Function[],
#     )
#   )
# 	# Bad (e.g. short?)
# 	run(`rm "../datasets/KDD-partitioned/covidwebnocough/2020-04-26-16_20_02_616687/audio_file_cough.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidwebnocough/2020-04-26-16_20_02_616687/audio_file_cough.wav-split.5.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebnosymp/2020-04-08-08_16_43_888185/audio_file_cough.wav-split.2.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-14-19_05_58_854062/audio_file_breathe.wav_aug_pitchspeed2.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidwebwithcough/2020-04-13-18_37_12_935759/audio_file_breathe.wav-split.6.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/breath/breaths_2DDMc0SESm_1587299604514.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/breath/breaths_zv02Ygabqh_1587712162421.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-14-19_05_58_854062/audio_file_breathe.wav_aug_pitchspeed1.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidwebwithcough/2020-04-13-18_37_12_935759/audio_file_breathe.wav-split.3.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/breath/breaths_wI0AtSFKGI_1586891988754.wav-split.7.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-14-19_05_58_854062/audio_file_breathe.wav_aug_noise1.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/breath/breaths_RZPXvUslJL_1589782005941.wav-split.3.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-10-16_31_45_878885/audio_file_breathe.wav_aug_pitchspeed2.wav-split.8.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-14-19_05_58_854062/audio_file_breathe.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-14-19_05_58_854062/audio_file_breathe.wav_aug_amp1.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-14-19_05_58_854062/audio_file_breathe.wav_aug_amp2.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-08-19_36_24_576603/audio_file_breathe.wav-split.5.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-08-19_36_24_576603/audio_file_breathe.wav_aug_amp1.wav-split.5.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-08-19_36_24_576603/audio_file_breathe.wav_aug_amp2.wav-split.5.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-08-19_36_24_576603/audio_file_breathe.wav_aug_noise1.wav-split.5.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/breath/breaths_wI0AtSFKGI_1587123341332.wav-split.7.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-08-19_36_24_576603/audio_file_breathe.wav_aug_noise1.wav-split.4.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-20-10_53_52_205272/audio_file_breathe.wav_aug_pitchspeed1.wav-split.3.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-20-10_53_52_205272/audio_file_breathe.wav_aug_pitchspeed2.wav-split.3.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidwebwithcough/2020-04-13-18_37_12_935759/audio_file_breathe.wav-split.7.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/breath/breaths_crxRiqIPHi_1587826550848.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/breath/breaths_RZPXvUslJL_1589436750953.wav-split.9.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidwebwithcough/2020-04-08-19_50_32_255545/audio_file_breathe.wav-split.9.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-09-13_18_07_699528/audio_file_breathe.wav_aug_pitchspeed2.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-08-19_36_24_576603/audio_file_breathe.wav-split.4.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-08-19_36_24_576603/audio_file_breathe.wav_aug_amp1.wav-split.4.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-08-19_36_24_576603/audio_file_breathe.wav_aug_amp2.wav-split.4.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-10-16_31_45_878885/audio_file_breathe.wav_aug_pitchspeed1.wav-split.6.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-07-20_32_56_249415/audio_file_breathe.wav_aug_pitchspeed2.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/breath/breaths_2DDMc0SESm_1587299604514.wav-split.2.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/breath/breaths_wI0AtSFKGI_1587123341332.wav-split.4.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthyandroidwithcough/breath/breaths_yfW1c2NlCx_1586974588197.wav-split.11.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthyandroidwithcough/breath/breaths_yfW1c2NlCx_1586974588197.wav_aug_amp1.wav-split.11.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthyandroidwithcough/breath/breaths_yfW1c2NlCx_1586974588197.wav_aug_amp2.wav-split.11.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthyandroidwithcough/breath/breaths_yfW1c2NlCx_1586974588197.wav_aug_pitchspeed1.wav-split.10.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/breath/breaths_wI0AtSFKGI_1587123341332.wav-split.5.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-14-14_29_49_101189/audio_file_breathe.wav_aug_pitchspeed2.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/breath/breaths_crxRiqIPHi_1587826550848.wav-split.6.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthyandroidwithcough/breath/breaths_yfW1c2NlCx_1586974588197.wav_aug_noise1.wav-split.11.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthyandroidwithcough/breath/breaths_yfW1c2NlCx_1586974588197.wav_aug_noise2.wav-split.11.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthyandroidwithcough/breath/breaths_yfW1c2NlCx_1586974588197.wav_aug_pitchspeed2.wav-split.10.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-14-14_29_49_101189/audio_file_breathe.wav_aug_pitchspeed1.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-20-10_53_52_205272/audio_file_breathe.wav_aug_pitchspeed1.wav-split.2.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-20-10_53_52_205272/audio_file_breathe.wav_aug_pitchspeed2.wav-split.2.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/breath/breaths_FLwelKnRhk_1587323996895.wav-split.4.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/breath/breaths_wI0AtSFKGI_1587123341332.wav-split.6.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-20-10_53_52_205272/audio_file_breathe.wav-split.3.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebwithcough/2020-04-20-10_53_52_205272/audio_file_breathe.wav_aug_amp1.wav-split.3.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthyandroidnosymp/cough/cough_988JFCXXIs_1586971301985.wav-split.5.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebnosymp/2020-04-15-17_40_04_078640/audio_file_cough.wav-split.2.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidwebnocough/2020-04-13-18_36_14_654135/audio_file_cough.wav-split.2.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/cough/cough_crxRiqIPHi_1588687422136.wav-split.5.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebnosymp/2020-04-19-17_45_04_814914/audio_file_cough.wav-split.2.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidnocough/cough/cough_CNz7PwFNQz_1588314464492.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthyandroidnosymp/cough/cough_PWUKaMIaFV_1588224156757.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthyandroidnosymp/cough/cough_r57l51XPwo_1587967960517.wav-split.3.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthyandroidnosymp/cough/cough_Yarb5WSBeD_1588832877170.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthywebnosymp/2020-04-19-17_45_04_814914/audio_file_cough.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/cough/cough_crxRiqIPHi_1587914038132.wav-split.3.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/cough/cough_crxRiqIPHi_1588604042078.wav-split.3.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidwebwithcough/2020-04-26-19_33_46_515975/audio_file_cough.wav-split.2.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/cough/cough_2DDMc0SESm_1588143205284.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidwebnocough/2020-04-21-12_31_25_239458/audio_file_cough.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidwebnocough/2020-04-26-10_36_54_607302/audio_file_cough.wav-split.2.wav"`)
# 	# Too long (TODO cut them properly)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/breath/breaths_RZPXvUslJL_1589525867100.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/breath/breaths_RZPXvUslJL_1589264294638.wav-split.1.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/covidandroidwithcough/breath/breaths_RZPXvUslJL_1589144967244.wav-split.2.wav"`)
# 	# TODO figure out # run(`rm "../datasets/KDD-partitioned/covidwebwithcough/2020-04-13-18_37_12_935759/audio_file_breathe.wav-split.8.wav"`)
# 	run(`rm "../datasets/KDD-partitioned/healthyandroidnosymp/breath/breaths_QiX6eemjYb_1587126241264.wav-split.2.wav"`)
# end

# generate_KDD_partitioned()

exec_dataseed = 1:10

exec_max_sample_rate = [8000] # , 16000]

# exec_dataset_dir = ["KDD", "KDD-partitioned"]

exec_ignore_low_sr_samples = [false] # , true]


# exec_use_training_form = [:dimensional]
exec_use_training_form = [:stump_with_memoization]

exec_n_ver_n_task_use_aug_dataset_dir_preprocess = [
	#
	# ("c",1,false,"KDD",["NG", "Normalize"]),
	# ("b",2,true,"KDD",["Normalize"]),
	# ("b",3,true,"KDD",["Normalize"]),
	#
	("c",1,false,"KDD-norm-partitioned-v1",["NG", "Normalize", "RemSilence"]),
	("c",2,true,"KDD-norm-partitioned-v1",["NG", "Normalize", "RemSilence"]),
	("c",3,true,"KDD-norm-partitioned-v1",["NG", "Normalize", "RemSilence"]),
	# ("b",1,false,"KDD-norm-partitioned-v1",["Normalize", "RemSilence"]),
	# ("b",2,true,"KDD-norm-partitioned-v1",["Normalize", "RemSilence"]),
	# ("b",3,true,"KDD-norm-partitioned-v1",["Normalize", "RemSilence"]),
]
# exec_n_task_use_aug = [
# 	(1, false),
# 	(2, true),
# 	(3, true),
# 	#(2, false),
# 	#(3, false),
# ]
# exec_n_versions = [3] #1:3

exec_fbtype = [:fcmel] #, :mel, :htkmel] #, :semitone]

exec_base_freq     = [:fft, :autocor]
exec_base_freq_min = [200, 300]
exec_base_freq_max = [600, 700]


exec_nbands = [30] # [20,40,60]
# exec_nbands = [40] # [20,40,60]

exec_dataset_kwargs =   [(
	# 	max_points = 50,
	# 	ma_size = 15,
	# 	ma_step = 10,
	# ),(
		max_points = 50,
		ma_size = 30,
		ma_step = 20,
	# ),(
	# 	max_points = 50,
	# 	ma_size = 45,
	# 	ma_step = 30,
	# ),(# max_points = 30,
	# 	ma_size = 120,
	# 	ma_step = 100,
	# ),(#max_points = 30,
	# 	max_points = 50,
	# 	ma_size = 100,
	# 	ma_step = 75,
	# ),(# max_points = 30,
	# 	ma_size = 90,
	# 	ma_step = 60,
	# ),(# max_points = 30,
	# 	ma_size = 75,
	# 	ma_step = 50,
	)
]

audio_kwargs_partial_mfcc(max_sample_rate, nbands, fbtype, base_freq, base_freq_min, base_freq_max) = (
	wintime = 0.025, # in ms          # 0.020-0.040
	steptime = 0.010, # in ms         # 0.010-0.015
	fbtype = fbtype, # :mel                   # [:mel, :htkmel, :fcmel]
	# window_f = DSP.hamming, # [DSP.hamming, (nwin)->DSP.tukey(nwin, 0.25)]
	window_f = DSP.triang,
	pre_emphasis = 0.97,              # any, 0 (no pre_emphasis)
	nbands = nbands,                      # any, (also try 20)
	sumpower = false,                 # [false, true]
	dither = false,                   # [false, true]
	# bwidth = 1.0,                   # 
	# minfreq = 0.0,
	maxfreq = max_sample_rate/2,
	# usecmp = false,
	base_freq     = base_freq,
	base_freq_min = base_freq_min,
	base_freq_max = base_freq_max,
)

audio_kwargs_full_mfcc(max_sample_rate, nbands, fbtype, base_freq, base_freq_min, base_freq_max) = (
	wintime=0.025,
	steptime=0.01,
	numcep=13,
	lifterexp=-22,
	sumpower=false,
	preemph=0.97,
	dither=false,
	minfreq=0.0,
	maxfreq = max_sample_rate/2,
	# maxfreq=sr/2,
	nbands=nbands,
	bwidth=1.0,
	dcttype=3,
	fbtype=fbtype, # :htkmel
	usecmp=false,
	modelorder=0,
	base_freq     = base_freq,
	base_freq_min = base_freq_min,
	base_freq_max = base_freq_max,
)

exec_use_full_mfcc = [false]


wav_preprocessors = Dict(
	"NoOp"       => identity,
	"NG"         => noise_gate!,
	"Trim"       => trim_wav!,
	"Normalize"  => normalize!,
	"RemSilence" => remove_long_silences!,
)

exec_preprocess_wavs = [
	["Normalize"],
	# ["NoOp"],
	["NG", "Normalize"],
	# ["NG", "Trim", "Normalize"],
]

# https://github.com/JuliaIO/JSON.jl/issues/203
# https://discourse.julialang.org/t/json-type-serialization/9794
# TODO: make test operators types serializable
# exec_test_operators = [ "TestOp" ]
exec_test_operators = [ "TestOp_80" ]

test_operators_dict = Dict(
	"TestOp_70" => [TestOpGeq_70, TestOpLeq_70],
	"TestOp_80" => [TestOpGeq_80, TestOpLeq_80],
	"TestOp"    => [TestOpGeq,    TestOpLeq],
)

exec_ontology = [ "IA", ] # "IA7", "IA3", 

ontology_dict = Dict(
	"-"     => ModalLogic.OneWorldOntology,
	"RCC8"  => getIntervalRCC8OntologyOfDim(Val(2)),
	"RCC5"  => getIntervalRCC5OntologyOfDim(Val(2)),
	"IA"    => getIntervalOntologyOfDim(Val(1)),
	"IA7"   => Ontology{ModalLogic.Interval}(ModalLogic.IA7Relations),
	"IA3"   => Ontology{ModalLogic.Interval}(ModalLogic.IA3Relations),
	"IA2D"  => getIntervalOntologyOfDim(Val(2)),
	# "o_ALLiDxA" => Ontology{ModalLogic.Interval2D}([ModalLogic.IA_AA, ModalLogic.IA_LA, ModalLogic.IA_LiA, ModalLogic.IA_DA]),
)


exec_ranges = (; # Order: faster-changing to slower-changing
	base_freq            = exec_base_freq,
	base_freq_min        = exec_base_freq_min,
	base_freq_max        = exec_base_freq_max,
	fbtype               = exec_fbtype,
	exec_max_sample_rate = exec_max_sample_rate,
	# exec_dataset_dir       = exec_dataset_dir,
	exec_ignore_low_sr_samples = exec_ignore_low_sr_samples,
	use_training_form    = exec_use_training_form,
	exec_n_ver_n_task_use_aug_dataset_dir_preprocess       = exec_n_ver_n_task_use_aug_dataset_dir_preprocess,
	# n_task_use_aug       = exec_n_task_use_aug,
	# n_version            = exec_n_versions,
	nbands               = exec_nbands,
	dataset_kwargs       = exec_dataset_kwargs,
	use_full_mfcc        = exec_use_full_mfcc,
	# preprocess_wavs      = exec_preprocess_wavs,
	test_operators       = exec_test_operators,
	ontology             = exec_ontology,
)

dataset_function = (
	(n_task,
		use_aug,
		n_version,
		cur_audio_kwargs,
		dataset_kwargs,
		cur_preprocess_wavs,
		use_full_mfcc,
		ignore_low_sr_samples,
		max_sample_rate,
		dataset_dir,
		)->
	KDDDataset(
		(n_task,n_version),
		cur_audio_kwargs;
		dataset_kwargs...,
		use_augmentation_data = use_aug,
		preprocess_wavs = cur_preprocess_wavs,
		use_full_mfcc = use_full_mfcc,
		ignore_samples_with_sr_less_than = (ignore_low_sr_samples ? max_sample_rate : -Inf),
		dir = "$(data_dir)$(dataset_dir)/",
	)
)

################################################################################
################################### SCAN FILTERS ###############################
################################################################################

# TODO let iteration_white/blacklist a decision function and not a "in-array" condition?
iteration_whitelist = [
	# TASK 1
	# (
	# 	n_version = 1,
	# 	nbands = 40,
	# 	dataset_kwargs = (max_points = 30, ma_size = 75, ma_step = 50),
	# ),
	# (
	# 	n_version = 1,
	# 	nbands = 60,
	# 	dataset_kwargs = (max_points = 30, ma_size = 75, ma_step = 50),
	# ),
	# # TASK 2
	# (
	# 	n_version = 2,
	# 	nbands = 20,
	# 	dataset_kwargs = (max_points = 30, ma_size = 45, ma_step = 30),
	# ),
	# (
	# 	n_version = 2,
	# 	nbands = 40,
	# 	dataset_kwargs = (max_points = 30, ma_size = 45, ma_step = 30),
	# )
]

iteration_blacklist = []

################################################################################
################################################################################
################################################################################
################################################################################

models_to_study = Dict([
	(
		"fcmel",8000,false,"stump_with_memoization",("c",3,true,"KDD-norm-partitioned-v1",["NG","Normalize","RemSilence"]),30,(max_points = 50, ma_size = 30, ma_step = 20),false,"TestOp_80","IA"
	) => [
		"tree_d3377114b972e5806a9e0631d02a5b9803c1e81d6cd6633b3dab4d9e22151969"
	],
])

models_to_study = Dict(JSON.json(k) => v for (k,v) in models_to_study)

MakeOntologicalDataset(Xs, test_operators, ontology) = begin
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
						throw_n_log("Unknown test_operator type: $(test_operator), $(typeof(test_operator))")
					end
				end
			end

			featsnops = Vector{<:TestOperatorFun}[
				if any(map(t->isa(feature,t), [AttributeMinimumFeatureType, AttributeSoftMinimumFeatureType]))
					[≥]
				elseif any(map(t->isa(feature,t), [AttributeMaximumFeatureType, AttributeSoftMaximumFeatureType]))
					[≤]
				else
					throw_n_log("Unknown feature type: $(feature), $(typeof(feature))")
					[≥, ≤]
				end for feature in features
			]

			OntologicalDataset(X, ontology, features, featsnops)
		end for X in Xs])
end

################################################################################
################################################################################
################################################################################
################################################################################

mkpath(results_dir)

if "-f" in ARGS
	if isfile(iteration_progress_json_file_path)
		println("Backing up existing $(iteration_progress_json_file_path)...")
		backup_file_using_creation_date(iteration_progress_json_file_path)
	end
end

# Copy scan script into the results folder
backup_file_using_creation_date(PROGRAM_FILE; copy_or_move = :copy, out_path = results_dir)

exec_ranges_names, exec_ranges_iterators = collect(string.(keys(exec_ranges))), collect(values(exec_ranges))
history = load_or_create_history(
	iteration_progress_json_file_path, exec_ranges_names, exec_ranges_iterators
)

# Log to console AND to .out file, & send Telegram message with Errors
using Logging, LoggingExtras
using Telegram, Telegram.API
using ConfigEnv

i_log_filename,log_filename = 0,""
while i_log_filename == 0 || isfile(log_filename)
	global i_log_filename,log_filename
	i_log_filename += 1
	log_filename = 
		results_dir * "/" *
		(dry_run == :dataset_only ? "datasets-" : "") *
		"$(i_log_filename).out"
end
logfile_io = open(log_filename, "w+")
dotenv()

tg = TelegramClient()
tg_logger = TelegramLogger(tg; async = false)

new_logger = TeeLogger(
	current_logger(),
	SimpleLogger(logfile_io, log_level),
	MinLevelLogger(tg_logger, Logging.Error), # Want to ignore Telegram? Comment out this
)
global_logger(new_logger)

################################################################################
################################################################################
################################################################################
################################################################################
# TODO actually,no need to recreate the dataset when changing, say, testoperators. Make a distinction between dataset params and run params
for params_combination in IterTools.product(exec_ranges_iterators...)

	flush(logfile_io);

	# Unpack params combination
	# params_namedtuple = (zip(Symbol.(exec_ranges_names), params_combination) |> Dict |> namedtuple)
	params_namedtuple = (;zip(Symbol.(exec_ranges_names), params_combination)...)

	# FILTER ITERATIONS
	if (!is_whitelisted_test(params_namedtuple, iteration_whitelist)) || is_blacklisted_test(params_namedtuple, iteration_blacklist)
		continue
	end

	##############################################################################
	##############################################################################
	##############################################################################

	run_name = join([replace(string(values(value)), ", " => ",") for value in params_combination], ",")

	# Placed here so we can keep track of which iteration is being skipped
	print("Iteration \"$(run_name)\"")

	# Check whether this iteration was already computed or not
	if all(iteration_in_history(history, (params_namedtuple, dataseed)) for dataseed in exec_dataseed) && (!save_datasets) # !skip_training
		println(": skipping")
		continue
	else
		println("...")
	end

	if dry_run == true
		continue
	end

	##############################################################################
	##############################################################################
	##############################################################################
	
	base_freq,
	base_freq_min,
	base_freq_max,
	fbtype,
	max_sample_rate,
	# dataset_dir,
	ignore_low_sr_samples,
	use_training_form,
	(n_version, n_task, use_aug, dataset_dir, preprocess_wavs),
	nbands,
	dataset_kwargs,
	use_full_mfcc,
	test_operators,
	ontology = params_combination
	
	test_operators = test_operators_dict[test_operators]
	ontology       = ontology_dict[ontology]

	cur_audio_kwargs = merge(
		if use_full_mfcc
			audio_kwargs_full_mfcc(max_sample_rate, nbands, fbtype, base_freq, base_freq_min, base_freq_max)
		else
			audio_kwargs_partial_mfcc(max_sample_rate, nbands, fbtype, base_freq, base_freq_min, base_freq_max)
		end
		, (;))

	cur_preprocess_wavs = [ wav_preprocessors[k] for k in preprocess_wavs ]

	cur_modal_args = modal_args
	
	cur_data_modal_args = merge(data_modal_args,
		(
			test_operators = test_operators,
			ontology       = ontology,
		)
	)

	dataset_fun_sub_params = (
		n_task, use_aug,
		n_version,
		cur_audio_kwargs,
		dataset_kwargs,
		cur_preprocess_wavs,
		use_full_mfcc,
		ignore_low_sr_samples,
		max_sample_rate,
		dataset_dir,
	)


	if dry_run == :model_study
		# println(JSON.json(params_combination))
		# println(models_to_study)
		# println(keys(models_to_study))
		if JSON.json(params_combination) in keys(models_to_study)
			
			trees = models_to_study[JSON.json(params_combination)]
			
			println()
			println()
			println("Study models for $(params_combination): $(trees)")

			if length(trees) == 0
				continue
			end
			
			println("dataset_fun_sub_params: $(dataset_fun_sub_params)")

			# @assert dataset_fun_sub_params isa String
			
			# dataset_fun_sub_params = merge(dataset_fun_sub_params, (; mode = :testing))

			datasets = []
			println("TODO")
			# datasets = [
			# 	(mode,if dataset_fun_sub_params isa Tuple
			# 		dataset = dataset_function(dataset_fun_sub_params...; mode = mode)
			# 		# dataset = @cachefast "dataset" data_savedir dataset_fun_sub_params dataset_function
			# 		(X, Y), (n_pos, n_neg) = dataset
			# 		# elseif dataset_fun_sub_params isa String
			# 		# 	# load_cached_obj("dataset", data_savedir, dataset_fun_sub_params)
			# 		# 	dataset = Serialization.deserialize("$(data_savedir)/dataset_$(dataset_fun_sub_params).jld").train_n_test
			# 		# 	println(typeof(dataset))
			# 		# 	(X, Y), (n_pos, n_neg) = dataset
			# 		# 	(X, Y, nothing), (n_pos, n_neg)

			# 		# TODO should not need these at test time. Instead, extend functions so that one can use a MatricialDataset instead of an OntologicalDataset
			# 		X = MakeOntologicalDataset(X, test_operators, ontology)
			# 		# println(length(Y))
			# 		# println((n_pos, n_neg))

			# 		println(display_structure(X))
			# 		# println(Y)
			# 		dataset = (X, Y), (n_pos, n_neg)
			# 		dataset
			# 	else
			# 		throw_n_log("$(typeof(dataset_fun_sub_params))")
			# 	end) for mode in [:testing, :development]
			# ]

			for model_hash in trees

				println()
				println()
				println("Loading model: $(model_hash)...")
				
				model = load_model(model_hash, model_savedir)

				println()
				println("Original model (training):")
				if model isa DTree
					print_model(model)
				end

				for (mode,dataset) in datasets
					(X, Y), (n_pos, n_neg) = dataset

					println()

					println()
					println("Regenerated model ($(mode)):")

					if model isa DTree
						regenerated_model = print_apply_model(model, X, Y; print_relative_confidence = true)
						println()
						# print_model(regenerated_model)
					end

					preds = apply_model(model, X);
					cm = confusion_matrix(Y, preds)
					println(cm)
					
					# readline()
				end
			end
		end
	end

	# Load Dataset
	dataset = @cachefast "dataset" data_savedir dataset_fun_sub_params dataset_function

	## Dataset slices
	# obtain dataseeds that are were not done before
	todo_dataseeds = filter((dataseed)->!iteration_in_history(history, (params_namedtuple, dataseed)), exec_dataseed)

	linearized_dataset, dataset_slices = 
		if dataset isa NamedTuple{(:train_n_test,:only_training)}
			balanced_dataset_slice(dataset, todo_dataseeds, split_threshold; discourage_only_training = prefer_nonaug_data)
		else
			balanced_dataset_slice(dataset, todo_dataseeds)
		end
	dataset_slices = collect(zip(todo_dataseeds, dataset_slices))

	println("Dataseeds = $(todo_dataseeds)")

	if dry_run == :dataset_only
		continue
	end

	##############################################################################
	##############################################################################
	##############################################################################
	
	if dry_run == false
		exec_scan(
			params_namedtuple,
			linearized_dataset;
			### Training params
			train_seed                      =   train_seed,
			modal_args                      =   cur_modal_args,
			tree_args                       =   tree_args,
			tree_post_pruning_purity_thresh =   [],
			forest_args                     =   forest_args,
			forest_runs                     =   forest_runs,
			optimize_forest_computation     =   optimize_forest_computation,
			test_flattened                  =   test_flattened,
			test_averaged                   =   test_averaged,
			### Dataset params
			split_threshold                 =   split_threshold,
			data_modal_args                 =   cur_data_modal_args,
			dataset_slices                  =   dataset_slices,
			round_dataset_to_datatype       =   round_dataset_to_datatype,
			use_training_form               =   use_training_form,
			### Run params
			results_dir                     =   results_dir,
			data_savedir                    =   data_savedir,
			model_savedir                   =   model_savedir,
			legacy_gammas_check             =   legacy_gammas_check,
			# logger                          =   logger,
			timing_mode                     =   timing_mode,
			### Misc
			save_datasets                   =   save_datasets,
			skip_training                   =   skip_training,
			callback                        =   (dataseed)->begin
				# Add this step to the "history" of already computed iteration
				push_iteration_to_history!(history, (params_namedtuple, dataseed))
				save_history(iteration_progress_json_file_path, history)
			end
		);
	end

end

println("Done!")

@error "Done!"

close(logfile_io);

exit(0)
