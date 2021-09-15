
rel_dict = Dict{String, DecisionTree.ModalLogic.AbstractRelation}([ DecisionTree.ModalLogic.display_rel_short(rel) => rel for rel in DecisionTree.ModalLogic.IARelations ])
rel_dict["G"] = DecisionTree.ModalLogic._RelationGlob()
rel_dict[""] = DecisionTree.ModalLogic._RelationGlob()
rel_dict["Id"] = DecisionTree.ModalLogic._RelationId()

pos_label = "YES"
neg_label = "NO_CLEAN_HISTORY_AND_LOW_PROBABILITY"

pos_n_neg(pos::Integer, neg::Integer)::Vector{String} = [ [ pos_label for i in 1:pos ]..., [ neg_label for i in 1:neg  ]... ]

leafYES() = DTLeaf{String}(pos_label, Vector{String}())
leafYES(correct::Integer, total::Integer) = DTLeaf{String}(pos_label, pos_n_neg(correct, total-correct))

leafNO() = DTLeaf{String}(neg_label, Vector{String}())
leafNO(correct::Integer, total::Integer) = DTLeaf{String}(neg_label, pos_n_neg(total-correct, correct))

internal(rel, attr, op, t, l, r) = begin
    DTInternal{Float64, String}(1, rel_dict[rel], 
        isequal(op, >=) ? DecisionTree.ModalLogic.AttributeSoftMinimumFeatureType(attr, 0.80) : DecisionTree.ModalLogic.AttributeSoftMaximumFeatureType(attr, 0.80),
        op, t, l, r
    )
end

"""
    Paper tree 'τ1`
"""
τ1_path = """./results-audio-scan/trees/tree_2762459adb09a9855451eac3779700f313ca7fd7f52260de7edf31db2c2b62ab.jld"""
τ1_str = """cough, 60, Normalized
⟨⟩ (V48 ⪴₈₀ 0.05543164795735502)
✔ V19 ⪴₈₀ 0.5800882546885471
│✔ ⟨B⟩ (V39 ⪴₈₀ 4.281481679778438)
││✔ YES : 12/12 (conf: 1.0)
││✘ V27 ⪳₈₀ 0.4761242998381368
││ ✔ ⟨L⟩ (V41 ⪴₈₀ 0.8572583827971237)
││ │✔ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 6/6 (conf: 1.0)
││ │✘ YES : 22/27 (conf: 0.8148148148148148)
││ ✘ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 40/45 (conf: 0.8888888888888888)
│✘ YES : 75/87 (conf: 0.8620689655172413)
✘ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 78/105 (conf: 0.7428571428571429)"""
τ1 = DTree(
    internal("G", 48, >=, 0.05543164795735502,
        internal("Id", 19, >=, 0.5800882546885471,
            internal("B", 39, >=, 4.281481679778438,
                leafYES(12, 12),
                internal("Id", 27, <=, 0.4761242998381368,
                    internal("L", 41, >=, 0.8572583827971237,
                        leafNO(6, 6),
                        leafYES(22, 27)
                    ),
                    leafNO(40, 45)
                )
            ),
            leafYES(75, 87)
        ),
        leafNO(78, 105)
    ), [DecisionTree.ModalLogic.Interval], [DecisionTree._startWithRelationGlob()]
)

"""
    Paper tree 'τ2`
"""
τ2_path = """./results-audio-scan/trees/tree_c5cd0b8a6705f0f04d081502c09cda82b7bf07f00b7497e4aff54263ebd3e828.jld"""
τ2_str = """cough, 40
⟨⟩ (V37 ⪴₈₀ 0.023027655130685647)
✔ ⟨B⟩ (V12 ⪴₈₀ 0.5608189970253608)
│✔ ⟨L⟩ (V27 ⪴₈₀ 3.931789023007937)
││✔ YES : 11/11 (conf: 1.0)
││✘ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 40/48 (conf: 0.8333333333333334)
│✘ YES : 91/109 (conf: 0.8348623853211009)
✘ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 83/114 (conf: 0.7280701754385965)"""
τ2 = DTree(
    internal("G", 37, >=, 0.023027655130685647,
        internal("B", 12, >=, 0.5608189970253608,
            internal("L", 27, >=, 3.931789023007937,
                leafYES(11, 11),
                leafNO(40, 48)
            ),
            leafYES(91, 109)
        ),
        leafNO(83, 114)
    ), [DecisionTree.ModalLogic.Interval], [DecisionTree._startWithRelationGlob()]
)

"""
    Paper tree 'τ3`
"""
τ3_path = """./results-audio-scan/trees/tree_d66e50308f2cf643cfb3822e18babd979226ed21118cc421008195e95bfa0da0.jld"""
τ3_str = """breathe, 40
⟨⟩ (V32 ⪴₈₀ 7.781613442411969e-5)
✔ V38 ⪳₈₀ 0.00016212943898189937
│✔ YES : 109/141 (conf: 0.7730496453900709)
│✘ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 31/40 (conf: 0.775)
✘ NO_CLEAN_HISTORY_AND_LOW_PROBABILITY : 78/101 (conf: 0.7722772277227723)"""
τ3 = DTree(
    internal("G", 32, >=, 7.781613442411969e-5,
        internal("Id", 38, <=, 0.00016212943898189937,
            leafYES(109, 141),
            leafNO(31, 40)
        ),
        leafNO(78, 101)
    ), [DecisionTree.ModalLogic.Interval], [DecisionTree._startWithRelationGlob()]
)
