# The code in this file for decision tree learning is inspired from:
# - Ben Sadeghi's DecisionTree.jl (released under the MIT license);
# - scikit-learn's and numpy's (released under the 3-Clause BSD license);

# Also thanks to Poom Chiarawongse <eight1911@gmail.com>

##############################################################################
##############################################################################
##############################################################################
##############################################################################

mutable struct NodeMeta{P,L}
    region             :: UnitRange{Int}                   # a slice of the samples used to decide the split of the node
    depth              :: Int
    modal_depth        :: Int
    # worlds      :: AbstractVector{WorldSet{W}}           # current set of worlds for each training instance
    purity             :: P                                # purity grade attained at training time
    prediction         :: L                                # most likely label
    is_leaf            :: Bool                             # whether this is a leaf node, or a split one
    # split node-only properties
    split_at           :: Int                              # index of samples
    l                  :: NodeMeta{P,L}                    # left child
    r                  :: NodeMeta{P,L}                    # right child

    i_frame            :: Integer                          # Id of frame
    decision           :: Decision{T} where {T}                    
    
    onlyallowRelationGlob:: Vector{Bool}

    function NodeMeta{P,L}(
            region      :: UnitRange{Int},
            depth       :: Int,
            modal_depth :: Int,
            oura        :: Vector{Bool},
            ) where {P,L}
        node = new{P,L}()
        node.region = region
        node.depth = depth
        node.modal_depth = modal_depth
        node.purity = P(NaN)
        node.is_leaf = false
        node.onlyallowRelationGlob = oura
        node
    end
end

# Split node at a previously-set node.split_at value.
# The children inherits some of the data
@inline function fork!(node::NodeMeta{P, L}) where {P, L}
    ind = node.split_at
    region = node.region
    depth = node.depth+1
    mdepth = node.modal_depth+Int(!node.is_leaf && !is_propositional_decision(node.decision))
    @logmsg DTDetail "fork!(...): " node ind region mdepth

    # onlyallowRelationGlob changes:
    # on the left node, the frame where the decision was taken
    l_oura = copy(node.onlyallowRelationGlob)
    l_oura[node.i_frame] = false
    r_oura = node.onlyallowRelationGlob

    # no need to copy because we will copy at the end
    node.l = NodeMeta{P,L}(region[    1:ind], depth, mdepth, l_oura)
    node.r = NodeMeta{P,L}(region[ind+1:end], depth, mdepth, r_oura)
end

# Conversion: NodeMeta (node + training info) -> DTNode (bare decision tree model)
function _convert(
        node          :: NodeMeta,
        labels        :: AbstractVector{L},
        class_names   :: AbstractVector{L},
        threshold_backmap :: Vector{<:Function}
    ) where {L<:CLabel}
    this_leaf = DTLeaf(class_names[node.prediction], labels[node.region])
    if node.is_leaf
        this_leaf
    else
        left  = _convert(node.l, labels, class_names, threshold_backmap)
        right = _convert(node.r, labels, class_names, threshold_backmap)
        DTInternal(node.i_frame, Decision(node.decision, threshold_backmap[node.i_frame]), this_leaf, left, right)
    end
end

# Conversion: NodeMeta (node + training info) -> DTNode (bare decision tree model)
function _convert(
        node   :: NodeMeta,
        labels :: AbstractVector{L},
        threshold_backmap :: Vector{<:Function}
    ) where {L<:RLabel}
    this_leaf = DTLeaf(node.prediction, labels[node.region])
    if node.is_leaf
        this_leaf
    else
        left  = _convert(node.l, labels, threshold_backmap)
        right = _convert(node.r, labels, threshold_backmap)
        DTInternal(node.i_frame, Decision(node.decision, threshold_backmap[node.i_frame]), this_leaf, left, right)
    end
end

##############################################################################
##############################################################################
##############################################################################
##############################################################################

# function optimize_tree_parameters!(
#       X               :: InterpretedModalDataset{T, N},
#       iC   :: InitCondition,
#       allow_global_splits :: Bool,
#       test_operators  :: AbstractVector{<:TestOperator}
#   ) where {T, N}

#   # A dimensional ontological datasets:
#   #  flatten to adimensional case + strip of all relations from the ontology
#   if prod(max_channel_size(X)) == 1
#       if (length(ontology(X).relations) > 0)
#           warn("The InterpretedModalDataset provided has degenerate max_channel_size $(max_channel_size(X)), and more than 0 relations: $(ontology(X).relations).")
#       end
#       # X = InterpretedModalDataset{T, 0}(ModalLogic.strip_ontology(ontology(X)), @views ModalLogic.strip_domain(domain(X)))
#   end

#   ontology_relations = deepcopy(ontology(X).relations)

#   # Fix test_operators order
#   test_operators = unique(test_operators)
#   ModalLogic.sort_test_operators!(test_operators)
    
#   # Adimensional operators:
#   #  in the adimensional case, some pairs of operators (e.g. <= and >)
#   #  are complementary, and thus it is redundant to check both at the same node.
#   #  We avoid this by only keeping one of the two operators.
#   if prod(max_channel_size(X)) == 1
#       # No ontological relation
#       ontology_relations = []
#       if test_operators ⊆ ModalLogic.all_lowlevel_test_operators
#           test_operators = [canonical_geq]
#           # test_operators = filter(e->e ≠ canonical_geq,test_operators)
#       else
#           warn("Test operators set includes non-lowlevel test operators. Update this part of the code accordingly.")
#       end
#   end

#   # Softened operators:
#   #  when the largest world only has a few values, softened operators fallback
#   #  to being hard operators
#   # max_world_wratio = 1/prod(max_channel_size(X))
#   # if canonical_geq in test_operators
#   #   test_operators = filter((e)->(typeof(e) != CanonicalFeatureGeqSoft || e.alpha < 1-max_world_wratio), test_operators)
#   # end
#   # if canonical_leq in test_operators
#   #   test_operators = filter((e)->(typeof(e) != CanonicalFeatureLeqSoft || e.alpha < 1-max_world_wratio), test_operators)
#   # end


#   # Binary relations (= unary modal operators)
#   # Note: the identity relation is the first, and it is the one representing
#   #  propositional splits.
    
#   if RelationId in ontology_relations
#       throw_n_log("Found RelationId in ontology provided. No need.")
#       # ontology_relations = filter(e->e ≠ RelationId, ontology_relations)
#   end

#   if RelationGlob in ontology_relations
#       throw_n_log("Found RelationGlob in ontology provided. Use allow_global_splits = true instead.")
#       # ontology_relations = filter(e->e ≠ RelationGlob, ontology_relations)
#       # allow_global_splits = true
#   end

#   relations = [RelationId, RelationGlob, ontology_relations...]
#   relationId_id = 1
#   relationGlob_id = 2
#   ontology_relation_ids = map((x)->x+2, 1:length(ontology_relations))

#   compute_relation_glob = (allow_global_splits || (iC == ModalDecisionTrees.start_without_world))

#   # Modal relations to compute gammas for
#   inUseRelation_ids = if compute_relation_glob
#       [relationGlob_id, ontology_relation_ids...]
#   else
#       ontology_relation_ids
#   end

#   # Relations to use at each split
#   availableRelation_ids = []

#   push!(availableRelation_ids, relationId_id)
#   if allow_global_splits
#       push!(availableRelation_ids, relationGlob_id)
#   end

#   availableRelation_ids = [availableRelation_ids..., ontology_relation_ids...]

#   (
#       test_operators, relations,
#       relationId_id, relationGlob_id,
#       inUseRelation_ids, availableRelation_ids
#   )
# end


# In the modal case, dataset instances are Kripke models.
# In this implementation, we don't accept a generic Kripke model in the explicit form of
#  a graph; instead, an instance is a dimensional domain (e.g. a matrix or a 3D matrix) onto which
#  worlds and relations are determined by a given Ontology.

# DEBUGprintln = println


############################################################################################
############################################################################################
############################################################################################
################################################################################
# Split a node
# Find an optimal local split satisfying the given constraints
#  (e.g. max_depth, min_samples_leaf, etc.)
Base.@propagate_inbounds @inline function split_node!(
    node                      :: NodeMeta{P, L},                     # node to splitt
    Xs                        :: ActiveMultiFrameModalDataset,       # modal dataset
    Ss                        :: AbstractVector{
        <:AbstractVector{WST} where {WorldType,WST<:WorldSet{WorldType}}
    }, # vector of current worlds for each instance and frame
    Y                         :: AbstractVector{L},                  # label vector
    init_conditions           :: AbstractVector{<:InitCondition},   # world starting conditions
    W                         :: AbstractVector{U}                   # weight vector
    ;
    ##########################################################################
    # Logic-agnostic training parameters
    loss_function             :: Union{Nothing,LossFunction},
    max_depth                 :: Int,                        # maximum depth of the resultant tree
    min_samples_leaf          :: Int,                        # minimum number of samples each leaf needs to have
    min_purity_increase       :: AbstractFloat,              # maximum purity allowed on a leaf
    max_purity_at_leaf        :: AbstractFloat,              # minimum purity increase needed for a split
    ##########################################################################
    # Modal parameters
    n_subrelations            :: AbstractVector{NSubRelationsFunction}, # relations used for the decisions
    n_subfeatures             :: AbstractVector{Int},        # number of features for the decisions
    allow_global_splits       :: AbstractVector{Bool},       # allow/disallow using RelationGlob at any decisional node
    ##########################################################################
    # Other
    idxs                      :: AbstractVector{Int},
    n_classes                 :: Int,
    _is_classification        :: Union{Val{true},Val{false}},
    _perform_consistency_check:: Union{Val{true},Val{false}},
    rng                       :: Random.AbstractRNG,
) where{P, L<:_Label, U, LossFunction<:Function, NSubRelationsFunction<:Function}

    # Region of idxs to use to perform the split
    region = node.region
    _n_samples = length(region)
    r_start = region.start - 1

    # DEBUGprintln("split_node!"); readline()

    # Gather all values needed for the current set of instances
    # TODO also slice the dataset?

    @inbounds Yf = Y[idxs[region]]
    @inbounds Wf = W[idxs[region]]

    # Yf = Vector{L}(undef, _n_samples)
    # Wf = Vector{U}(undef, _n_samples)
    # @inbounds @simd for i in 1:_n_samples
    #   Yf[i] = Y[idxs[i + r_start]]
    #   Wf[i] = W[idxs[i + r_start]]
    # end

    ############################################################################
    # Prepare counts
    ############################################################################
    if _is_classification isa Val{true}
        (nc, nt),
        (node.purity, node.prediction) = begin
            nc = fill(zero(U), n_classes)
            @inbounds @simd for i in 1:_n_samples
                nc[Yf[i]] += Wf[i]
            end
            nt = sum(nc)
            # TODO use _compute_purity
            purity = loss_function(loss_function(nc, nt)::Float64)::Float64
            # Assign the most likely label before the split
            prediction = argmax(nc)
            # prediction = average_label(Yf)
            (nc, nt), (purity, prediction)
        end
    else
        sums, (tsum, nt),
        (node.purity, node.prediction) = begin
            # sums = [Wf[i]*Yf[i]       for i in 1:_n_samples]
            sums = Yf
            # ssqs = [Wf[i]*Yf[i]*Yf[i] for i in 1:_n_samples]
            
            # tssq = zero(U)
            # tssq = sum(ssqs)
            # tsum = zero(U)
            tsum = sum(sums)
            # nt = zero(U)
            nt = sum(Wf)
            # @inbounds @simd for i in 1:_n_samples
            #   # tssq += Wf[i]*Yf[i]*Yf[i]
            #   # tsum += Wf[i]*Yf[i]   
            #   nt += Wf[i]
            # end

            # purity = (tsum * prediction) # TODO use loss function
            # purity = tsum * tsum # TODO use loss function
            # tmean = tsum/nt
            # purity = -((tssq - 2*tmean*tsum + (tmean^2*nt)) / (nt-1)) # TODO use loss function
            # TODO use _compute_purity
            purity = begin
                if W isa UniformVector{Int}
                    loss_function(loss_function(sums, tsum, length(sums))::Float64)
                else
                    loss_function(loss_function(sums, Wf, nt)::Float64)
                end
            end
            # Assign the most likely label before the split
            prediction =  tsum / nt
            # prediction = average_label(Yf)
            sums, (tsum, nt), (purity, prediction)
        end
    end

    ############################################################################
    ############################################################################
    ############################################################################

    @logmsg DTDebug "_split!(...) " _n_samples region nt

    ############################################################################
    # Preemptive leaf conditions
    ############################################################################
    if _is_classification isa Val{true}
        if (
            # If all instances belong to the same class, make this a leaf
                (nc[node.prediction]       == nt)
            # No binary split can honor min_samples_leaf if there aren't as many as
            #  min_samples_leaf*2 instances in the first place
             || (min_samples_leaf * 2 >  _n_samples)
            # If the node is pure enough, avoid splitting # TODO rename purity to loss
             || (node.purity          > max_purity_at_leaf)
            # Honor maximum depth constraint
             || (max_depth            < node.depth))
            # DEBUGprintln("BEFORE LEAF!")
            # DEBUGprintln(nc[node.prediction])
            # DEBUGprintln(nt)
            # DEBUGprintln(min_samples_leaf)
            # DEBUGprintln(_n_samples)
            # DEBUGprintln(node.purity)
            # DEBUGprintln(max_purity_at_leaf)
            # DEBUGprintln(max_depth)
            # DEBUGprintln(node.depth)
            # readline()
            node.is_leaf = true
            @logmsg DTDetail "leaf created: " (min_samples_leaf * 2 >  _n_samples) (nc[node.prediction] == nt) (node.purity  > max_purity_at_leaf) (max_depth <= node.depth)
            return
        end
    else
        if (
            # No binary split can honor min_samples_leaf if there aren't as many as
            #  min_samples_leaf*2 instances in the first place
                (min_samples_leaf * 2 >  _n_samples)
          # equivalent to old_purity > -1e-7
             || (node.purity          > max_purity_at_leaf) # TODO
             # || (tsum * node.prediction    > -1e-7 * nt + tssq)
            # Honor maximum depth constraint
             || (max_depth            < node.depth))
            node.is_leaf = true
            @logmsg DTDetail "leaf created: " (min_samples_leaf * 2 >  _n_samples) (tsum * node.prediction    > -1e-7 * nt + tssq) (tsum * node.prediction) (-1e-7 * nt + tssq) (max_depth <= node.depth)
            return
        end
    end
    ############################################################################
    ############################################################################
    ############################################################################

    # TODO try this solution for rsums and lsums (regression case)
    # rsums = Vector{U}(undef, _n_samples)
    # lsums = Vector{U}(undef, _n_samples)
    # @simd for i in 1:_n_samples
    #   rsums[i] = zero(U)
    #   lsums[i] = zero(U)
    # end

    Sfs = Vector{Vector{WST} where {WorldType,WST<:WorldSet{WorldType}}}(undef, n_frames(Xs))
    for (i_frame,WT) in enumerate(get_world_types(Xs))
        Sfs[i_frame] = Vector{Vector{WT}}(undef, _n_samples)
        @simd for i in 1:_n_samples
            Sfs[i_frame][i] = Ss[i_frame][idxs[i + r_start]]
        end
    end

    # Optimization-tracking variables
    best_i_frame = -1
    best_purity_times_nt = typemin(P)
    best_decision = Decision{Float64}()
    if isa(_perform_consistency_check,Val{true})
        consistency_sat_check = Vector{Bool}(undef, _n_samples)
    end
    best_consistency = nothing
    
    #####################
    ## Find best split ##
    #####################
    ## Test all decisions
    # For each frame (modal dataset)
    @inbounds for (i_frame,
                (X,
                frame_Sf,
                frame_n_subrelations::Function,
                frame_n_subfeatures,
                frame_allow_global_splits,
                frame_onlyallowRelationGlob)) in enumerate(zip(frames(Xs), Sfs, n_subrelations, n_subfeatures, allow_global_splits, node.onlyallowRelationGlob))

        @logmsg DTDetail "  Frame $(best_i_frame)/$(length(frames(Xs)))"

        allow_propositional_decisions, allow_modal_decisions, allow_global_decisions, modal_relations_inds, features_inds = begin
            
            # Derive subset of features to consider
            # Note: using "sample" function instead of "randperm" allows to insert weights for features which may be wanted in the future 
            features_inds = StatsBase.sample(rng, 1:n_features(X), frame_n_subfeatures, replace = false)
            sort!(features_inds)

            # Derive all available relations
            allow_propositional_decisions, allow_modal_decisions, allow_global_decisions = begin
                if world_type(X) == OneWorld
                    true, false, false
                elseif frame_onlyallowRelationGlob
                    false, false, true
                else
                    true, true, frame_allow_global_splits
                end
            end

            tot_relations = 0
            if allow_modal_decisions
                tot_relations += length(relations(X))
            end
            if allow_global_decisions
                tot_relations += 1
            end
            
            # Derive subset of relations to consider
            n_subrel = Int(frame_n_subrelations(tot_relations))
            modal_relations_inds = StatsBase.sample(rng, 1:tot_relations, n_subrel, replace = false)
            sort!(modal_relations_inds)

            # Check whether the global relation survived
            if allow_global_decisions
                allow_global_decisions = (tot_relations in modal_relations_inds)
                modal_relations_inds = filter!(r->r≠tot_relations, modal_relations_inds)
                tot_relations = length(modal_relations_inds)
            end
            allow_propositional_decisions, allow_modal_decisions, allow_global_decisions, modal_relations_inds, features_inds
        end
        
        # println(modal_relations_inds)
        # println(features_inds)
        # readline()

        ########################################################################
        ########################################################################
        ########################################################################
        
        @inbounds for (decision, aggr_thresholds) in generate_feasible_decisions(X, idxs[region], frame_Sf, allow_propositional_decisions, allow_modal_decisions, allow_global_decisions, modal_relations_inds, features_inds)
            
            # println(display_decision(i_frame, decision))

            # TODO avoid ugly unpacking and figure out a different way of achieving this
            (test_operator, threshold) = (decision.test_operator, decision.threshold)
            ########################################################################
            # Apply decision to all instances
            ########################################################################
            # Note: consistency_sat_check is also changed
            if _is_classification isa Val{true}
                (ncr, nr, ncl, nl) = begin
                    # Re-initialize right counts
                    nr = zero(U)
                    ncr = fill(zero(U), n_classes)
                    if isa(_perform_consistency_check,Val{true})
                        consistency_sat_check .= 1
                    end
                    for i_sample in 1:_n_samples
                        gamma = aggr_thresholds[i_sample]
                        satisfied = evaluate_thresh_decision(test_operator, gamma, threshold)
                        @logmsg DTDetail " instance $i_sample/$_n_samples: (f=$(gamma)) -> satisfied = $(satisfied)"
                        
                        # Note: in a fuzzy generalization, `satisfied` becomes a [0-1] value
                        if !satisfied
                            nr += Wf[i_sample]
                            ncr[Yf[i_sample]] += Wf[i_sample]
                        else
                            if isa(_perform_consistency_check,Val{true})
                                consistency_sat_check[i_sample] = 0
                            end
                        end
                    end
                    # Calculate left counts
                    ncl = Vector{U}(undef, n_classes)
                    ncl .= nc .- ncr
                    nl = nt - nr
                    
                    (ncr, nr, ncl, nl)
                end
            else
                (rsums, nr, lsums, nl, rsum, lsum) = begin
                    # Initialize right counts
                    # rssq = zero(U)
                    rsum = zero(U)
                    nr   = zero(U)
                    # TODO experiment with running mean instead, because this may cause a lot of memory inefficiency
                    # https://it.wikipedia.org/wiki/Algoritmi_per_il_calcolo_della_varianza
                    rsums = Float64[] # Vector{U}(undef, _n_samples)
                    lsums = Float64[] # Vector{U}(undef, _n_samples)

                    if isa(_perform_consistency_check,Val{true})
                        consistency_sat_check .= 1
                    end
                    for i_sample in 1:_n_samples
                        gamma = aggr_thresholds[i_sample]
                        satisfied = evaluate_thresh_decision(test_operator, gamma, threshold)
                        @logmsg DTDetail " instance $i_sample/$_n_samples: (f=$(gamma)) -> satisfied = $(satisfied)"
                        
                        # TODO make this satisfied a fuzzy value
                        if !satisfied
                            push!(rsums, sums[i_sample])
                            # rsums[i_sample] = sums[i_sample]
                            nr   += Wf[i_sample]
                            rsum += sums[i_sample]
                            # rssq += ssqs[i_sample]
                        else
                            push!(lsums, sums[i_sample])
                            # lsums[i_sample] = sums[i_sample]
                            if isa(_perform_consistency_check,Val{true})
                                consistency_sat_check[i_sample] = 0
                            end
                        end
                    end

                    # Calculate left counts
                    lsum = tsum - rsum
                    # lssq = tssq - rssq
                    nl   = nt - nr
                    
                    (rsums, nr, lsums, nl, rsum, lsum)
                end
            end
            
            ########################################################################
            ########################################################################
            ########################################################################
            
            @logmsg DTDebug "  (n_left,n_right) = ($nl,$nr)"

            # Honor min_samples_leaf
            if nl >= min_samples_leaf && (_n_samples - nl) >= min_samples_leaf
                purity_times_nt = begin
                    if _is_classification isa Val{true}
                        loss_function(ncl, nl, ncr, nr)
                    else
                        purity = begin
                            if W isa UniformVector{Int}
                                loss_function(lsums, lsum, nl, rsums, rsum, nr)
                            else
                                error("TODO expand code to weigthed version!")
                                loss_function(lsums, ws_l, nl, rsums, ws_r, nr)
                            end
                        end
                        
                        # TODO use loss_function instead
                        # ORIGINAL: TODO understand how it works
                        # purity_times_nt = (rsum * rsum / nr) + (lsum * lsum / nl)
                        # Variance with ssqs
                        # purity_times_nt = (rmean, lmean = rsum/nr, lsum/nl; - (nr * (rssq - 2*rmean*rsum + (rmean^2*nr)) / (nr-1) + (nl * (lssq - 2*lmean*lsum + (lmean^2*nl)) / (nl-1))))
                        # Variance
                        # var = (x)->sum((x.-StatsBase.mean(x)).^2) / (length(x)-1)
                        # purity_times_nt = - (nr * var(rsums)) + nl * var(lsums))
                        # Simil-variance that is easier to compute but it doesn't work with few samples on the leaves
                        # var = (x)->sum((x.-StatsBase.mean(x)).^2)
                        # purity_times_nt = - (var(rsums) + var(lsums))
                        # println("purity_times_nt: $(purity_times_nt)")
                    end
                end::P

                if purity_times_nt > best_purity_times_nt # && !isapprox(purity_times_nt, best_purity_times_nt)
                    # DEBUGprintln((ncl,nl,ncr,nr), purity_times_nt)
                    #################################
                    best_i_frame             = i_frame
                    #################################
                    best_purity_times_nt     = purity_times_nt
                    #################################
                    best_decision            = decision
                    #################################
                    # print(decision)
                    # println(" NEW BEST $best_i_frame, $best_purity_times_nt/nt")
                    @logmsg DTDetail "  Found new optimum in frame $(best_i_frame): " (best_purity_times_nt/nt) best_decision
                    #################################
                    best_consistency = begin
                        if isa(_perform_consistency_check,Val{true})
                            consistency_sat_check[1:_n_samples]
                        else
                            nr
                        end
                    end
                    #################################
                end
            end
        end # END decisions
    end # END frame

    # TODO, actually, when using Shannon entropy, we must correct the purity:
    corrected_best_purity_times_nt = loss_function(best_purity_times_nt)::Float64

    # DEBUGprintln("corrected_best_purity_times_nt: $(corrected_best_purity_times_nt)")
    # DEBUGprintln(min_purity_increase)
    # DEBUGprintln(node.purity)
    # DEBUGprintln(corrected_best_purity_times_nt)
    # DEBUGprintln(nt)
    # DEBUGprintln(purity - best_purity_times_nt/nt)
    # DEBUGprintln("dishonor: $(dishonor_min_purity_increase(L, min_purity_increase, node.purity, corrected_best_purity_times_nt, nt))")
    # readline()

    # println("corrected_best_purity_times_nt = $(corrected_best_purity_times_nt)")
    # println("nt =  $(nt)")
    # println("node.purity =  $(node.purity)")
    # println("corrected_best_purity_times_nt / nt - node.purity = $(corrected_best_purity_times_nt / nt - node.purity)")
    # println("min_purity_increase * nt =  $(min_purity_increase) * $(nt) = $(min_purity_increase * nt)")

    # @logmsg DTOverview "purity_times_nt increase" corrected_best_purity_times_nt/nt node.purity (corrected_best_purity_times_nt/nt + node.purity) (best_purity_times_nt/nt - node.purity)
    # If the best split is good, partition and split accordingly
    @inbounds if ((
            corrected_best_purity_times_nt == typemin(P)) ||
            dishonor_min_purity_increase(L, min_purity_increase, node.purity, corrected_best_purity_times_nt, nt)
        )
        
        if _is_classification isa Val{true}
            @logmsg DTDebug " Leaf" corrected_best_purity_times_nt min_purity_increase (corrected_best_purity_times_nt/nt) node.purity ((corrected_best_purity_times_nt/nt) - node.purity)
        else
            @logmsg DTDebug " Leaf" corrected_best_purity_times_nt tsum node.prediction min_purity_increase nt (corrected_best_purity_times_nt / nt - tsum * node.prediction) (min_purity_increase * nt)
        end
        ##########################################################################
        ##########################################################################
        ##########################################################################
        # DEBUGprintln("AFTER LEAF!")
        # readline()
        node.is_leaf = true
    else
        best_purity = corrected_best_purity_times_nt/nt

        # Compute new world sets (= take a modal step)

        # println(decision_str)
        decision_str = display_decision(best_i_frame, best_decision)
        
        # TODO instead of using memory, here, just use two opposite indices and perform substitutions. indj = _n_samples
        unsatisfied_flags = fill(1, _n_samples)
        if isa(_perform_consistency_check,Val{true})
            world_refs = []
        end
        for i_sample in 1:_n_samples
            # TODO perform step with an OntologicalModalDataset

            # instance = ModalLogic.get_instance(X, best_i_frame, idxs[i_sample + r_start])
            X = get_frame(Xs, best_i_frame)
            Sf = Sfs[best_i_frame]
            # instance = ModalLogic.get_instance(X, idxs[i_sample + r_start])

            # println(instance)
            # println(Sf[i_sample])
            _sat, _ss = ModalLogic.modal_step(X, idxs[i_sample + r_start], Sf[i_sample], best_decision)
            (satisfied,Ss[best_i_frame][idxs[i_sample + r_start]]) = _sat, _ss
            @logmsg DTDetail " [$satisfied] Instance $(i_sample)/$(_n_samples)" Sf[i_sample] (if satisfied Ss[best_i_frame][idxs[i_sample + r_start]] end)
            # println(satisfied)
            # println(Ss[best_i_frame][idxs[i_sample + r_start]])
            # readline()

            # I'm using unsatisfied because sorting puts YES instances first, but TODO use the inverse sorting and use satisfied flag instead
            unsatisfied_flags[i_sample] = !satisfied
            if isa(_perform_consistency_check,Val{true})
                push!(world_refs, _ss)
            end
        end

        @logmsg DTDetail " unsatisfied_flags" unsatisfied_flags

        if length(unique(unsatisfied_flags)) == 1
            throw_n_log("An uninformative split was reached. Something's off\nPurity: $(node.purity)\nSplit: $(decision_str)\nUnsatisfied flags: $(unsatisfied_flags)")
        end
        @logmsg DTDetail " Branch ($(sum(unsatisfied_flags))+$(_n_samples-sum(unsatisfied_flags))=$(_n_samples) samples) on frame $(best_i_frame) with decision: $(decision_str), purity $(best_purity)"

        # if sum(unsatisfied_flags) >= min_samples_leaf && (_n_samples - sum(unsatisfied_flags)) >= min_samples_leaf
            # DEBUGprintln("LEAF!")
        #     node.is_leaf = true
        #     return
        # end

        # Check consistency
        consistency = if isa(_perform_consistency_check,Val{true})
                unsatisfied_flags
            else
                sum(Wf[BitVector(unsatisfied_flags)])
        end

        if !isapprox(best_consistency,consistency; atol=eps(Float32), rtol=eps(Float32))
            errStr = ""
            errStr *= "A low-level error occurred. Please open a pull request with the following info."
            errStr *= "Decision $(best_decision).\n"
            errStr *= "Possible causes:\n"
            errStr *= "- feature returning NaNs\n"
            errStr *= "- erroneous accessibles_aggr for relation $(best_decision.relation), aggregator $(existential_aggregator(best_decision.test_operator)) and feature $(best_decision.feature)\n"
            errStr *= "\n"
            errStr *= "Branch ($(sum(unsatisfied_flags))+$(_n_samples-sum(unsatisfied_flags))=$(_n_samples) samples) on frame $(best_i_frame) with decision: $(decision_str), purity $(best_purity)\n"
            errStr *= "$(length(idxs[region])) Instances: $(idxs[region])\n"
            errStr *= "Different partition was expected:\n"
            if isa(_perform_consistency_check,Val{true})
                errStr *= "Actual: $(consistency) ($(sum(consistency)))\n"
                errStr *= "Expected: $(best_consistency) ($(sum(best_consistency)))\n"
                diff = best_consistency.-consistency
                errStr *= "Difference: $(diff) ($(sum(abs.(diff))))\n"
            else
                errStr *= "Actual: $(consistency)\n"
                errStr *= "Expected: $(best_consistency)\n"
                diff = best_consistency-consistency
                errStr *= "Difference: $(diff)\n"
            end
            errStr *= "unsatisfied_flags = $(unsatisfied_flags)\n"

            if isa(_perform_consistency_check,Val{true})
                errStr *= "world_refs = $(world_refs)\n"
                errStr *= "new world_refs = $([Ss[best_i_frame][idxs[i_sample + r_start]] for i_sample in 1:_n_samples])\n"
            end
            
            # for i in 1:_n_samples
                # errStr *= "$(ModalLogic.get_channel(Xs, idxs[i + r_start], best_decision.feature))\t$(Sf[i])\t$(!(unsatisfied_flags[i]==1))\t$(Ss[best_i_frame][idxs[i + r_start]])\n";
            # end

            # throw_n_log("ERROR! " * errStr)
            println("ERROR! " * errStr) # TODO fix
        end

        @logmsg DTDetail " unsatisfied_flags" unsatisfied_flags

        # TODO this should be satisfied, since min_samples_leaf is always > 0 and nl,nr>min_samples_leaf
        if length(unique(unsatisfied_flags)) == 1
            errStr = "An uninformative split was reached. Something's off\n"
            errStr *= "Purity: $(best_purity)\n"
            errStr *= "Split: $(decision_str)\n"
            errStr *= "Unsatisfied flags: $(unsatisfied_flags)"

            println("ERROR! " * errStr) # TODO fix
            # throw_n_log(errStr)
            node.is_leaf = true
        else
            # split the samples into two parts:
            #  ones for which the is satisfied and those for whom it's not
            node.purity         = best_purity
            node.i_frame        = best_i_frame
            node.decision       = best_decision

            # DEBUGprintln("unsatisfied_flags")
            # DEBUGprintln(unsatisfied_flags)

            @logmsg DTDetail "pre-partition" region idxs[region] unsatisfied_flags
            node.split_at = util.partition!(idxs, unsatisfied_flags, 0, region)
            @logmsg DTDetail "post-partition" idxs[region] node.split_at

            # For debug:
            # idxs = rand(1:10, 10)
            # unsatisfied_flags = rand([1,0], 10)
            # partition!(idxs, unsatisfied_flags, 0, 1:10)
            
            # Sort [Xf, Yf, Wf, Sf and idxs] by Xf
            # util.q_bi_sort!(unsatisfied_flags, idxs, 1, _n_samples, r_start)
            # node.split_at = searchsortedfirst(unsatisfied_flags, true)
        end
    end

    # println("END split!")
    # readline()
    # node
end


############################################################################################
############################################################################################
############################################################################################

@inline function _fit_tree(
        Xs                        :: ActiveMultiFrameModalDataset,       # modal dataset
        Y                         :: AbstractVector{L},                  # label vector
        init_conditions           :: AbstractVector{<:InitCondition},   # world starting conditions
        W                         :: AbstractVector{U}                   # weight vector
        ;
        ##########################################################################
        _is_classification        :: Union{Val{true},Val{false}},
        _perform_consistency_check:: Union{Val{true},Val{false}},
        rng = Random.GLOBAL_RNG   :: Random.AbstractRNG,
        print_progress            :: Bool = true,
        kwargs...,
    ) where{L<:_Label, U}

    _n_samples = n_samples(Xs)
    
    # Initialize world sets for each instance
    Ss = init_world_sets(Xs, init_conditions)

    # Distribution of the instances indices throughout the tree.
    #  It will be recursively permuted, and regions of it assigned to the tree nodes (idxs[node.region])
    idxs = collect(1:_n_samples)
    
    # Create root node
    NodeMetaT = NodeMeta{Float64,(_is_classification isa Val{true} ? Int64 : Float64)}
    onlyallowRelationGlob = [(iC == ModalDecisionTrees.start_without_world) for iC in init_conditions]
    root = NodeMetaT(1:_n_samples, 0, 0, onlyallowRelationGlob)
    
    # if print_progress TODO
    #     p = ProgressThresh(Inf, 1, "Computing DTree...")
    # end

    # Process nodes recursively, using multi-threading
    function process_node!(node, rng)
        # Note: better to spawn rng's beforehand, to preserve reproducibility independently from split_node!
        rng_l = util.spawn_rng(rng)
        rng_r = util.spawn_rng(rng)
        @inbounds split_node!(node, Xs, Ss, Y, init_conditions, W;
            _is_classification         = _is_classification,
            _perform_consistency_check = _perform_consistency_check,
            idxs                       = idxs,
            rng                        = rng,
            kwargs...,
        )
        # TODO: !print_progress || ProgressMeter.update!(p, ...idxs[region]...)
        if !node.is_leaf
            fork!(node)
            l = Threads.@spawn process_node!(node.l, rng_l)
            r = Threads.@spawn process_node!(node.r, rng_r)
            wait(l), wait(r)
        end
    end
    @sync Threads.@spawn process_node!(root, rng)

    return (root, idxs)
end

##############################################################################
##############################################################################
##############################################################################
##############################################################################

@inline function check_input(
        Xs                      :: ActiveMultiFrameModalDataset,
        Y                       :: AbstractVector{S},
        init_conditions         :: Vector{<:InitCondition},
        W                       :: AbstractVector{U}
        ;
        ##########################################################################
        loss_function           :: Function,
        max_depth               :: Int,
        min_samples_leaf        :: Int,
        min_purity_increase     :: AbstractFloat,
        max_purity_at_leaf      :: AbstractFloat,
        ##########################################################################
        n_subrelations          :: Vector{<:Function},
        n_subfeatures           :: Vector{<:Integer},
        allow_global_splits     :: Vector{Bool},
        ##########################################################################
        kwargs...,
    ) where {S, U}
    _n_samples = n_samples(Xs)

    if length(Y) != _n_samples
        throw_n_log("Dimension mismatch between dataset and label vector Y: ($(_n_samples)) vs $(size(Y))")
    elseif length(W) != _n_samples
        throw_n_log("Dimension mismatch between dataset and weights vector W: ($(_n_samples)) vs $(size(W))")
    ############################################################################
    elseif length(n_subrelations) != n_frames(Xs)
        throw_n_log("Mismatching number of n_subrelations with number of frames: $(length(n_subrelations)) vs $(n_frames(Xs))")
    elseif length(n_subfeatures)  != n_frames(Xs)
        throw_n_log("Mismatching number of n_subfeatures with number of frames: $(length(n_subfeatures)) vs $(n_frames(Xs))")
    elseif length(init_conditions) != n_frames(Xs)
        throw_n_log("Mismatching number of init_conditions with number of frames: $(length(init_conditions)) vs $(n_frames(Xs))")
    elseif length(allow_global_splits) != n_frames(Xs)
        throw_n_log("Mismatching number of allow_global_splits with number of frames: $(length(allow_global_splits)) vs $(n_frames(Xs))")
    ############################################################################
    # elseif any(n_relations(Xs) .< n_subrelations)
    #   throw_n_log("In at least one frame the total number of relations is less than the number "
    #       * "of relations required at each split\n"
    #       * "# relations:    " * string(n_relations(Xs)) * "\n\tvs\n"
    #       * "# subrelations: " * string(n_subrelations |> collect))
    # elseif length(findall(n_subrelations .< 0)) > 0
    #   throw_n_log("Total number of relations $(n_subrelations) must be >= zero ")
    elseif any(n_features(Xs) .< n_subfeatures)
        throw_n_log("In at least one frame the total number of features is less than the number "
            * "of features required at each split\n"
            * "# features:    " * string(n_features(Xs)) * "\n\tvs\n"
            * "# subfeatures: " * string(n_subfeatures |> collect))
    elseif length(findall(n_subfeatures .< 0)) > 0
        throw_n_log("Total number of features $(n_subfeatures) must be >= zero ")
    elseif min_samples_leaf < 1
        throw_n_log("Min_samples_leaf must be a positive integer "
            * "(given $(min_samples_leaf))")
    # if loss_function in [util.entropy]
    #   max_purity_at_leaf_thresh = 0.75 # min_purity_increase 0.01
    #   min_purity_increase_thresh = 0.5
    #   if (max_purity_at_leaf >= max_purity_at_leaf_thresh)
    #       println("Warning! It is advised to use max_purity_at_leaf<$(max_purity_at_leaf_thresh) with loss $(loss_function)"
    #           * "(given $(max_purity_at_leaf))")
    #   elseif (min_purity_increase >= min_purity_increase_thresh)
    #       println("Warning! It is advised to use max_purity_at_leaf<$(min_purity_increase_thresh) with loss $(loss_function)"
    #           * "(given $(min_purity_increase))")
    # end
    # elseif loss_function in [util.gini, util.zero_one] && (max_purity_at_leaf > 1.0 || max_purity_at_leaf <= 0.0)
    #     throw_n_log("Max_purity_at_leaf for loss $(loss_function) must be in (0,1]"
    #         * "(given $(max_purity_at_leaf))")
    elseif max_depth < -1
        throw_n_log("Unexpected value for max_depth: $(max_depth) (expected:"
            * " max_depth >= 0, or max_depth = -1 for infinite depth)")
    end

    if ModalLogic.hasnans(Xs)
        # println(Xs)
        # println(ModalLogic.display_structure(Xs))
        # println(ModalLogic.hasnans(Xs))
        # println(ModalLogic.hasnans.([X.emd for X in frames(Xs)]))
        # println(ModalLogic.hasnans.([X.emd.fwd for X in frames(Xs)]))
        # println(frames(Xs)[1].emd.fwd)
        throw_n_log("This algorithm doesn't allow NaN values")
    end

    if nothing in Y
        throw_n_log("This algorithm doesn't allow nothing values in Y")
    # elseif any(isnan.(Y))
    #   throw_n_log("This algorithm doesn't allow NaN values in Y")
    elseif nothing in W
        throw_n_log("This algorithm doesn't allow nothing values in W")
    elseif any(isnan.(W))
        throw_n_log("This algorithm doesn't allow NaN values in W")
    end

end


############################################################################################
############################################################################################
############################################################################################
################################################################################

function fit_tree(
        # modal dataset
        Xs                        :: ActiveMultiFrameModalDataset,
        # label vector
        Y                         :: AbstractVector{L},
        # world starting conditions
        init_conditions           :: Vector{<:InitCondition},
        # Weights (unary weigths are used if no weight is supplied)
        W                         :: AbstractVector{U} = default_weights(n_samples(Xs))
        # W                       :: AbstractVector{U} = Ones{Int}(n_samples(Xs)), # TODO check whether this is faster
        ;
        # Perform minification: transform dataset so that learning happens faster
        perform_minification      :: Bool,
        # Debug-only: checks the consistency of the dataset during training
        perform_consistency_check :: Bool,
        kwargs...,
    ) where {L<:Union{CLabel,RLabel}, U}
    # Check validity of the input
    check_input(Xs, Y, init_conditions, W; kwargs...)

    # Classification-only: transform labels to categorical form (indexed by integers)
    n_classes = begin
        if L<:CLabel
            class_names, Y = util.get_categorical_form(Y)
            length(class_names)
        else
            0 # dummy value for the case of regression
        end
    end

    Xs, threshold_backmaps = begin
        if perform_minification
            minify(Xs)
        else
            Xs, fill(identity, n_frames(Xs))
        end
    end
    # println(threshold_backmaps)
    # Call core learning function
    root, idxs = _fit_tree(Xs, Y, init_conditions, W;
        n_classes = n_classes,
        _is_classification = Val(L<:CLabel),
        _perform_consistency_check = Val(perform_consistency_check),
        kwargs...
    )
    
    # Finally create Tree
    root = begin
        if L<:CLabel
            _convert(root, map((y)->class_names[y], Y[idxs]), class_names, threshold_backmaps)
        else
            _convert(root, Y[idxs], threshold_backmaps)
        end
    end
    DTree{L}(root, get_world_types(Xs), init_conditions)
end
