################################################################################
# Experimental features
################################################################################
module experimentals

using ..ModalDecisionTrees
using ..ModalDecisionTrees.ModalLogic
using SoleLogics

MDT = ModalDecisionTrees
ML  = ModalLogic
SL  = SoleLogics

include("parse.jl")
include("decisionpath.jl")

end
