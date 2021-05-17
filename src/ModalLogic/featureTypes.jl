export FeatureType

const FeatureType = Integer
# abstract type FeatureType end

SimpleFeatureType(a, feature) = feature

display_feature(feature) = "V$(feature)"
################################################################################
################################################################################

# struct _FeatureTypeNone  <: FeatureType end; const FeatureTypeNone  = _FeatureTypeNone();

################################################################################
################################################################################



# f(x) = getindex(x,1,:) |> maximum
