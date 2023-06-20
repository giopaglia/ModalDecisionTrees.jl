################################################################################
# Experimental features
################################################################################
module experimentals

using ..ModalDecisionTrees
using SoleModels
using SoleModels.DimensionalDatasets
using SoleLogics

using SoleModels.DimensionalDatasets
using SoleModels: MultiLogiset
using SoleModels: WorldSet, GenericDataset


using SoleModels: nfeatures, nrelations,
                            nmodalities, frames, frame,
                            displaystructure,
                            #
                            relations,
                            #
                            GenericDataset,
                            MultiLogiset,
                            AbstractLogiset,
                            DimensionalLogiset,
                            Logiset,
                            SupportedScalarLogiset

using SoleModels: AbstractWorld, AbstractRelation
using SoleModels: AbstractWorldSet, WorldSet
using SoleModels: FullDimensionalFrame

using SoleModels: Ontology, worldtype, worldtypes

using SoleModels: get_ontology,
                            get_interval_ontology

using SoleModels: OneWorld, OneWorldOntology

using SoleModels: Interval, Interval2D

using SoleModels: IARelations

MDT = ModalDecisionTrees
SL  = SoleLogics

include("parse.jl")
include("decisionpath.jl")

end
