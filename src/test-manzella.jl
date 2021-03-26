# julia -i -t4 test-manzella.jl
# julia -i test-manzella.jl

include("test-header.jl")

rng = my_rng()

args = (
	loss = DecisionTree.util.entropy,
	# loss = DecisionTree.util.gini,
	# loss = DecisionTree.util.zero_one,
	# max_depth = -1,
	# min_samples_leaf = 4,
	# min_purity_increase = 0.02, # TODO check this
	# min_loss_at_leaf = 1.0, # TODO check there's something wrong here, I think this sets min_purity_increase.
)

# TODO add parameter: allow relationAll at all levels? Maybe it must be part of the relations... I don't know
kwargs = (
	initCondition = DecisionTree.startAtCenter,
	# initCondition = DecisionTree._startAtWorld(ModalLogic.Interval2D((1,3),(3,4))),
	# initCondition = DecisionTree.startWithRelationAll,
	
	# ontology = getIntervalOntologyOfDim(Val(2)),
	# ontology = Ontology(ModalLogic.Interval2D,setdiff(Set(ModalLogic.RCC8Relations),Set([ModalLogic.Topo_PO]))),
	# ontology = Ontology(ModalLogic.Interval2D,[ModalLogic._IA2DRel(i,j) for j in [ModalLogic.IA_O,ModalLogic.IA_Oi] for i in [ModalLogic.IA_O,ModalLogic.IA_Oi]]),
	ontology = getIntervalRCC8OntologyOfDim(Val(1)),
	# ontology = getIntervalRCC8OntologyOfDim(Val(2)),
	# ontology = getIntervalRCC5OntologyOfDim(Val(2)),

	# ontology=Ontology(ModalLogic.Interval2D,ModalLogic.AbstractRelation[]),
	useRelationId = true,
	# useRelationId = false,
	# useRelationAll = true,
	useRelationAll = false,
	# test_operators = [ModalLogic.TestOpGeq],
	# test_operators = [ModalLogic.TestOpLeq],
	test_operators = [ModalLogic.TestOpGeq, ModalLogic.TestOpLeq],
	# test_operators = [ModalLogic.TestOpGeq, ModalLogic.TestOpLeq, ModalLogic.TestOpGeq_85, ModalLogic.TestOpLeq_85],
	# test_operators = [ModalLogic.TestOpGeq_75, ModalLogic.TestOpLeq_75],
	# test_operators = [ModalLogic.TestOpGeq_85, ModalLogic.TestOpLeq_85],
	# test_operators = [ModalLogic.TestOpGeq_75],
	# rng = my_rng,
	# rng = DecisionTree.mk_rng(123),
)

loss = DecisionTree.util.entropy
min_samples_leaf = 4
min_purity_increase = 0.01
min_loss_at_leaf = 0.3

selected_args = merge(args, (loss = loss,
															min_samples_leaf = min_samples_leaf,
															min_purity_increase = min_purity_increase,
															min_loss_at_leaf = min_loss_at_leaf,
															))
log_level = DecisionTree.DTOverview
# log_level = Logging.Warn

# timeit = 2
timeit = 0
scale_dataset = false
# scale_dataset = UInt8


# n_instances = 1
n_instances = 100
# n_instances = 300
# n_instances = 500

# rng_i = DecisionTree.mk_rng(124)
rng_i = DecisionTree.mk_rng(1)

dataset = SplatEduardDataset(5)
T = testDataset("Test", dataset, false, 0, args=args, kwargs=kwargs);      
