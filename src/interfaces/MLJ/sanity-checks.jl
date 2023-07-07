
# if model.check_conditions == true
#     check_conditions(model.conditions)
# end
# function check_conditions(conditions)
#     if isnothing(conditions)
#         return
#     end
#     # Check that feature extraction functions are scalar
#     wrong_conditions = filter((f)->begin
#             !all(
#                 (ch)->!(f isa Base.Callable) ||
#                     (ret = f(ch); isa(ret, Real) && typeof(ret) == eltype(ch)),
#                 [collect(1:10), collect(1.:10.)]
#             )
#         end, conditions)
#     @assert length(wrong_conditions) == 0 "When specifying feature extraction functions " *
#         "for inferring `conditions`, please specify " *
#         "scalar functions accepting an object of type `AbstractArray{T}` " *
#         "and returning an object of type `T`, with `T<:Real`. " *
#         "Instead, got wrong feature functions: $(wrong_conditions)."
# end
