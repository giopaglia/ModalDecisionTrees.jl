################################################################################
################################################################################
################################## Scan script #################################
################################################################################
################################################################################

include("scanner.jl")

using SoleBase
using SoleBase: dimension
using SoleViz
using StatsPlots
using Plots.PlotMeasures


train_seed = 1


################################################################################
#################################### FOLDERS ###################################
################################################################################

results_dir = "./siemens/TURBOEXPO-regression-v3-fix-leakage-n-true-reg"

iteration_progress_json_file_path = results_dir * "/progress.json"
data_savedir  = results_dir * "/data_cache"
model_savedir = results_dir * "/models_cache"

# dry_run = false
dry_run = :dataset_only
# dry_run = :model_study
# dry_run = true

skip_training = false

#save_datasets = true
save_datasets = false

perform_consistency_check = false

iteration_blacklist = []

################################################################################
##################################### TREES ####################################
################################################################################

# Optimization arguments for single-tree
tree_args = [
#	(
#		loss_function = DecisionTree.util.entropy,
#		min_samples_leaf = 1,
#		min_purity_increase = 0.01,
#		max_purity_at_leaf = 0.6,
#	)
]

for loss_function in [nothing] # DecisionTree.util.variance]
	for min_samples_leaf in [4] # [1,2]
		for min_purity_increase in [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0] # [0.01, 0.001]
		# for min_purity_increase in [0.0] # ,0.01, 0.001]
			for max_purity_at_leaf in [0.001] # [0.4, 0.6]
				push!(tree_args,
					(
						loss_function       = loss_function,
						min_samples_leaf    = min_samples_leaf,
						min_purity_increase = min_purity_increase,
						max_purity_at_leaf  = max_purity_at_leaf,
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
optimize_forest_computation = true


forest_args = []

for n_trees in [50, 100]
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
					# max_purity_at_leaf  = Inf,
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
	ontology = getIntervalOntologyOfDim(Val(1)),
	# ontology = Ontology{ModalLogic.Interval}([ModalLogic.IA_A]),
	# ontology = Ontology{ModalLogic.Interval}([ModalLogic.IA_A, ModalLogic.IA_L, ModalLogic.IA_Li, ModalLogic.IA_D]),
	test_operators = [TestOpGeq_80, TestOpLeq_80],
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

traintest_threshold = 0.8
# train_instances_per_class = 100

split_threshold = 1.0
# split_threshold = 0.8
# split_threshold = 1.0
# split_threshold = false

# use_training_form = :dimensional
# use_training_form = :fmd
# use_training_form = :stump
# use_training_form = :stump_with_memoization

test_flattened = false
test_averaged  = false

prefer_nonaug_data = true

################################################################################
##################################### SCAN #####################################
################################################################################

exec_dataseed = [1:4...,0]

#exec_datadirname = ["Siemens-Data-Features", "Siemens-Data-Measures"]
exec_datadirname = ["Siemens-Data-Measures"]

exec_use_training_form = [:stump_with_memoization]

exec_binning = [
	nothing,
	# (1.0 => "High-Risk", 3.0 => "Risky", Inf => "Low-Risk"),
	# (4.0 => "High-Risk", 8.0 => "Risky", Inf => "Low-Risk"),
]

exec_ignore_last_minutes = [10] # , 2*60]
# exec_regression_step_in_minutes = [1]
exec_regression_step_in_minutes = [10]
exec_regression_window_in_minutes = [60]

exec_moving_average = [
	(
		ma_size = 4,
		ma_step = 4,
	),
	(
		ma_size = 3,
		ma_step = 3,
	),
	(
		ma_size = 2,
		ma_step = 2,
	),
]

exec_ranges = (;
	datadirname                                  = exec_datadirname,
	use_training_form                            = exec_use_training_form,
	binning                                      = exec_binning,
	ignore_last_minutes                          = exec_ignore_last_minutes,
	moving_average                               = exec_moving_average,
	regression_window_in_minutes                 = exec_regression_window_in_minutes,
	regression_step_in_minutes                   = exec_regression_step_in_minutes,
)


dataset_function = (datadirname, binning, ignore_last_minutes, moving_average, regression_window_in_minutes, regression_step_in_minutes)->begin
	SiemensDataset_regression(datadirname;
		moving_average...,
		binning = binning,
		sortby_datasource = true,
		only_consider_trip_days = true,
		ignore_last_minutes = ignore_last_minutes,
		regression_window_in_minutes = regression_window_in_minutes,
		regression_step_in_minutes = regression_step_in_minutes,
	)
end

################################################################################
################################### SCAN FILTERS ###############################
################################################################################

iteration_whitelist = []

################################################################################
################################################################################
################################################################################
################################################################################

models_to_study = Dict([
	# (
	# 	"fcmel",8000,false,"stump_with_memoization",("c",3,true,"KDD-norm-partitioned-v1",["NG","Normalize","RemSilence"]),30,(max_points = 50, ma_size = 30, ma_step = 20),false,"TestOp_80","IA"
	# ) => [
	# 	"tree_d3377114b972e5806a9e0631d02a5b9803c1e81d6cd6633b3dab4d9e22151969"
	# ],
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
n_interations = 0
n_interations_done = 0
for params_combination in IterTools.product(exec_ranges_iterators...)

	flush(logfile_io);

	# Unpack params combination
	# params_namedtuple = (zip(Symbol.(exec_ranges_names), params_combination) |> Dict |> namedtuple)
	params_namedtuple = (;zip(Symbol.(exec_ranges_names), params_combination)...)

	# FILTER ITERATIONS
	if (!is_whitelisted_test(params_namedtuple, iteration_whitelist)) || is_blacklisted_test(params_namedtuple, iteration_blacklist)
		continue
	end

	global n_interations += 1

	##############################################################################
	##############################################################################
	##############################################################################

	run_name = join([replace(string(values(value)), ", " => ",") for value in params_combination], ",")

	# Placed here so we can keep track of which iteration is being skipped
	print("Iteration \"$(run_name)\"")

	# Check whether this iteration was already computed or not
	if all(iteration_in_history(history, (params_namedtuple, dataseed)) for dataseed in exec_dataseed) && (!save_datasets)
		println(": skipping")
		continue
	else
		println("...")
	end

	global n_interations_done += 1

	if dry_run == true
		continue
	end

	##############################################################################
	##############################################################################
	##############################################################################
	
	datadirname, use_training_form, binning, ignore_last_minutes, moving_average, regression_window_in_minutes, regression_step_in_minutes = params_combination
	dataset_fun_sub_params = (datadirname, binning, ignore_last_minutes, moving_average, regression_window_in_minutes, regression_step_in_minutes)
	
	cur_modal_args = modal_args
	cur_data_modal_args = data_modal_args

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
	dataset, datasource_counts = @cachefast "dataset" data_savedir dataset_fun_sub_params dataset_function

	# class_counts = get_class_counts(dataset)

	X, Y = dataset

	##############################################################################
	##############################################################################
	##############################################################################

	# using SoleBase
	# using StatsPlots
	# using Plots.PlotMeasures

	# X, Y = randn(19,5,20), rand(0:1, 20)

	for f in []
		# f = [1][1]

		n_attrs = size(X,2)[end]
		n_insts = size(X)[end]
		columns = []
		for i_attr in 1:n_attrs
			push!(columns, ([X[:,i_attr,i_inst] for i_inst in 1:n_insts]))
		end
		colnames = [string(i) for i in 1:n_attrs]
		df = DataFrame(columns, colnames)
		mfd = MultiFrameDataset([1:ncol(df)], df)
		# ClassificationMultiFrameDataset(Y, mfd)

		# println(mfd)

		p = SoleViz.plotdescription(mfd)[1]

		plot!(p; size = (1920, 1080))
		savefig(p, "plotdescription-$(run_name).png")

		display(p)

	# 	for t in [1] # ,2,4,8,16] # TODO eventualmente fissarlo.
	# 		# t = 4
	# 		# description = describe(mfd; desc = [:mean_m], t = [(t,0,0)])[1]
	# 		# TODO add catch22
	# 		description = describe(mfd; desc = [:mean_m, :min_m, :max_m], t = [(t,0,0)])[1]
	# 		attr_descr_cols = names(description)[2:end]
	# 		# println(description)
	# 		# println(attr_descr_cols)

	# 		# max_mean = mean([v for values in description[:,:mean_m] for v in values])
	# 		# plot(collect(1:n_attrs), [description[band,:mean_m][1] for band in 1:n_attrs], leg = false, ylims = (0, max_mean), size = (1920, 1080))
	# 		# for i_inst in 2:(nrow(new_dataset)-1)
	# 		# 	plot!(collect(1:n_attrs), [description[band,:mean_m][i_inst] for band in 1:n_attrs], leg = false, ylims = (0, max_mean), size = (1920, 1080))
	# 		# end
	# 		# all_plot = plot!(collect(1:n_attrs), [description[band,:mean_m][nrow(new_dataset)] for band in 1:n_attrs], leg = false, ylims = (0, max_mean), size = (1920, 1080))

	# 		d = SoleBase.SoleData.SoleDataset._stat_description(description; functions = Function[mean, var, std])
	# 		# p = plot(
	# 		# 	[
	# 		# 		plot(collect(1:n_attrs), [v[1] for v in d[:,feat_symb]], size = (1080, 1080), title = string(feat_symb), xticks = 1:n_attrs)
	# 		# 		for feat_symb in [
	# 		# 				:mean_m_mean, :mean_m_var, :mean_m_std,
	# 		# 				:min_m_mean, :min_m_var, :min_m_std,
	# 		# 				:max_m_mean, :max_m_var, :max_m_std]
	# 		# 	]...
	# 		# 	layout = (3, 3),
	# 		# )

	# 		# display(p)
	# 		# readline()
			
	# 		p = begin
	# 			feat_symb = :mean_m_var
	# 			ys = [v[1] for v in d[:,feat_symb]]
	# 			sp = sortperm(ys, rev = true)
	# 			println(sp)
	# 			plot(collect(1:n_attrs), ys[sp]; size = (1080, 1080), title = string(feat_symb), xticks = (1:n_attrs,sp))
	# 		end

	# 		# p = plot(
	# 		# 	[
	# 		# 		begin
	# 		# 			ys = [v[1] for v in d[:,feat_symb]]
	# 		# 			sp = sortperm(ys, rev=true)
	# 		# 			println(sp)
	# 		# 			plot(collect(1:n_attrs), ys[sp]; size = (1080, 1080), title = string(feat_symb), xticks = sp)
	# 		# 		end for feat_symb in [
	# 		# 			:mean_m_var,
	# 		# 			:min_m_var,
	# 		# 			:max_m_var,
	# 		# 		]
	# 		# 	]...;
	# 		# 	layout = (3, 1),
	# 		# 	xticks = (ys = [v[1] for v in d[:,:mean_m_var]]; sp = sortperm(ys, rev=true)),
	# 		# )

	# 		display(p)

	# 		# for (i_op, op_descr) in enumerate(eachcol(description[:,2:end]))
	# 		# 	# (i_op, op_descr) = collect(enumerate(eachcol(description[:,2:end])))[1]

	# 		# 	stackedhists = []
	# 		# 	op = attr_descr_cols[i_op]
	# 		# 	for (i_attr, matrix) in enumerate(op_descr)
	# 		# 		# (i_attr, matrix) = collect(enumerate(op_descr))[1]
	# 		# 		# matrix = op_descr[1]
	# 		# 		# println(i_op, op_descr)
	# 		# 		println(size(matrix))
	# 		# 		gr()
	# 		# 		for w in 1:t
	# 		# 			# w = (1:t)[1]
	# 		# 			kwargs = begin
	# 		# 				if w == 1
	# 		# 					(title = "Histograms",)
	# 		# 				elseif w == t
	# 		# 					(xlabel = "Response",)
	# 		# 				else
	# 		# 					(;)
	# 		# 				end
	# 		# 			end
	# 		# 			# push!(stackedhists, histogram(matrix[:,w], bins=20, legend=false, ylabel="$(op), $(w)/$(t)", xticks=nothing, kwargs...))
	# 		# 			push!(stackedhists, histogram(matrix[:,w], bins=20, legend=false, ylabel="$(w), $(i_attr)", xticks=nothing))
	# 		# 		end
	# 		# 		# h = histogram([matrix[:,w] for w in 1:t]; bins=20, legend=false, ylabel="$(op), $(t) chunks, $(i_attr) attribute", xticks=nothing, title = "Histograms", xlabel = "Response")
	# 		# 		# readline()
	# 		# 		# println(stackedhists |> typeof)
	# 		# 		# break
	# 		# 		# println(names(col))
	# 		# 		# println(matrix)
	# 		# 		# println(matrix |> typeof)
	# 		# 	end
	# 		# 	plot(stackedhists..., layout=(n_attrs,t), size = (1000,500), margin=1mm, title = "$(op)")
	# 		# end
	# 		readline()
	# 	end

	end

	# histogram2d(x, y)

	# using StatsBase, Plots
	# h = fit(Histogram, (x, y), nbins=20)
	# plot(h) # same as histogram2d
	# wireframe(midpoints(h.edges[1]), midpoints(h.edges[2]), h.weights)

	##############################################################################
	##############################################################################
	##############################################################################
	

	# println("class_distribution: ")
	# println(StatsBase.countmap(dataset[2]))
	# println("class_counts: $(class_counts)")

	## Dataset slices
	# obtain dataseeds that are were not done before
	todo_dataseeds = filter((dataseed)->!iteration_in_history(history, (params_namedtuple, dataseed)), exec_dataseed)

	dataset_slices = begin
		n_insts = length(Y)
		# @assert (n_insts % n_cv_folds == 0) "$(n_insts) % $(n_cv_folds) != 0"
		# n_insts_fold = div(n_insts, n_cv_folds)
		
		# todo_dataseeds = 1:10
		[(dataseed, begin
				if dataseed == 0
					(Vector{Integer}(collect(1:n_insts)), Vector{Integer}(collect(1:n_insts)))
				else
					# balanced_dataset_slice(class_counts, dataseed; n_samples_per_class = floor(Int, minimum(class_counts)*traintest_threshold), also_return_discarted = true)
					# balanced_dataset_slice(class_counts, dataseed; n_samples_per_class = train_instances_per_class, also_return_discarted = true)
					@assert dataseed in 1:length(datasource_counts)
					@assert datasource_counts[dataseed] > 0
					a = datasource_counts[1:dataseed-1];
					idx_base = (length(a) == 0 ? 0 : sum(a))
					test_idxs = idx_base .+ (1:datasource_counts[dataseed])
					# test_idxs = 1+(dataseed-1)*n_insts_fold:(dataseed-1)*n_insts_fold+(n_insts_fold)
					(Vector{Integer}(collect(test_idxs)), Vector{Integer}(collect(setdiff(Set(1:n_insts), Set(test_idxs)))))
				end
			end) for dataseed in todo_dataseeds]
	end

	println("Dataseeds = $(todo_dataseeds)")

	# for (dataseed,data_slice) in dataset_slices
	# 	println("class_distribution: ")
	# 	println(StatsBase.countmap(dataset[2][data_slice]))
	# 	println("...")
	# 	break # Note: Assuming this print is the same for all dataseeds
	# end
	# println()

	if dry_run == :dataset_only
		continue
	end

	##############################################################################
	##############################################################################
	##############################################################################
	
	if dry_run == false
		exec_scan(
			params_namedtuple,
			dataset;
			is_regression_problem           =   (eltype(Y) != String),
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
println("# Iterations $(n_interations_done)/$(n_interations)")

# Notify the Telegram Bot
@error "Done!"

close(logfile_io);

exit(0)
