################################################################################
# Experimental features
################################################################################
module experimentals

using ..ModalDecisionTrees
using SoleModels
using SoleModels.DimensionalDatasets
using SoleLogics

using SoleModels.DimensionalDatasets
using SoleModels: MultiFrameLogiset
using SoleModels: WorldSet, GenericModalDataset


using SoleModels: nfeatures, nrelations,
                            nframes, frames, frame,
                            displaystructure,
                            #
                            relations,
                            #
                            GenericModalDataset,
                            ActiveMultiFrameLogiset,
                            MultiFrameLogiset,
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
