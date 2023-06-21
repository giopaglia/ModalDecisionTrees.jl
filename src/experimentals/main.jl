################################################################################
# Experimental features
################################################################################
module experimentals

using ModalDecisionTrees
using ModalDecisionTrees:
       relation, feature, test_operator, threshold,
       is_propositional_decision,
       is_global_decision

using SoleModels
using SoleModels.DimensionalDatasets
using SoleLogics


using SoleModels: nfeatures, nrelations,
                            nmodalities, eachmodality, modality,
                            displaystructure,
                            #
                            relations,
                            #
                            GenericDataset,
                            MultiLogiset,
                            AbstractLogiset,
                            SupportedLogiset

using SoleModels: scalarlogiset

using SoleModels: AbstractWorld, AbstractRelation
using SoleModels: AbstractWorldSet, WorldSet

using SoleLogics: FullDimensionalFrame
using SoleModels.DimensionalDatasets
using SoleModels: MultiLogiset
using SoleModels: WorldSet, GenericDataset


using SoleModels: worldtype

using SoleModels: OneWorld

using SoleModels: Interval, Interval2D

using SoleModels: IARelations

MDT = ModalDecisionTrees
SL  = SoleLogics

include("parse.jl")
include("decisionpath.jl")

end
