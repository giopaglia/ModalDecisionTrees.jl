using DataFrames
using Pingouin
using CSV
using Plots
using Distributions
using StatsPlots
using HypothesisTests

input_file = "./siemens/TURBOEXPO-regression-v8-analysis-first/plotdescription-Siemens-Data-Measures,stump_with_memoization,(4.0 => High-Risk,Inf => Low-Risk),10,(1,1),60,10,Any[]-sub-values.csv"
class_names = ["High-Risk", "Low-Risk"] # TODO read :class_names column instead

@assert length(class_names) == 2 "TODO expand code to multi-class setting"

@assert all([!endswith(c1, c2) for (i,c1) in enumerate(class_names) for c2 in class_names[i+1:end]]) "a class endswith another class (class_names = $(class_names))"

df = CSV.read(input_file, DataFrame)

# assert nrow(df) == length(class_names)

canonical_features = names(df)

# columns = unique(map((n)->(begin
# 		if endswith(n, "-x") n[1:end-length("-x")]
# 		elseif endswith(n, "-y") n[1:end-length("-y")]
# 		else error(n)
# 		end
# 	end), columns))

# canonical_features = unique(map((n)->(begin
# 		_class_names = ["-"*c for c in class_names]
# 		f = [endswith(n, c) for c in _class_names]
# 		if any(f)
# 			c = _class_names[findfirst(x->x==true, f)]
# 			n[1:end-length(c)]
# 		else error(n)
# 		end
# 	end), columns))

for feature in canonical_features
	println("TEST " * feature)
	some_test_failed = false
	class_serie_is_normal = []
	series = []
	for i_c in 1:nrow(df)
		serie = eval(Meta.parse(df[i_c,feature]))
		push!(series, serie)
		# println(serie)
		try
			stats = Pingouin.normality(serie)

			pval = stats[1,:pval]
			W = stats[1,:W]
			normal = stats[1,:normal]

			# println()
			# println(serie)
			println("\t$(class_names[i_c])")
			if normal
				println("NORMALITY")
				println("\t\t$(pval), $(W), $(normal)")
				push!(class_serie_is_normal,true)
			else
				println("NON-NORMAL ($(pval))")
				push!(class_serie_is_normal,false)
			end
		catch e
			println("Failure on serie:")
			# println(serie)
			println(e)
			readline()
			some_test_failed = true
		end
	end

	if some_test_failed
		println("NOTE: some tests failed")
	end
	# length(unique(class_serie_is_normal)) == 1
	if some_test_failed || false in class_serie_is_normal
		s1 = series[1]
		s2 = series[2]
		density(s1)
		display(density!(s2))

		# stats = Pingouin.wilcoxon(s1, s2)
		# stats = Pingouin.mwu(s1, s2)
		# pval = stats[1,:p_val]
		# stats = HypothesisTests.ApproximateMannWhitneyUTest(s1, s2)
		stats = HypothesisTests.ExactMannWhitneyUTest(s1, s2)
		# stats = HypothesisTests.MannWhitneyUTest(s1, s2)
		pval = pvalue(stats)
		# println(stats)
		# println(stats.pvalue)
		# pval = stats[1,:p_val]
		println("pval = $(pval)")
		readline()
	else
		println("T-TEST:")
		error("TODO expand code. which version of the t-test?")
	end
end
