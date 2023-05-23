################################################################################
# Experimental features
################################################################################
module experimentals

using ..ModalDecisionTrees
using SoleModels
using SoleModels.DimensionalDatasets
using SoleLogics

using SoleModels.DimensionalDatasets
using SoleModels.DimensionalDatasets: MultiFrameConditionalDataset
using SoleModels.DimensionalDatasets: ActiveMultiFrameConditionalDataset
using SoleModels: WorldSet, GenericModalDataset


using SoleModels: nfeatures, nrelations,
                            nframes, frames, getframe,
                            display_structure,
                            #
                            relations,
                            #
                            GenericModalDataset,
                            ActiveMultiFrameConditionalDataset,
                            MultiFrameConditionalDataset,
                            AbstractActiveFeaturedDataset,
                            DimensionalFeaturedDataset,
                            FeaturedDataset,
                            SupportedFeaturedDataset

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
