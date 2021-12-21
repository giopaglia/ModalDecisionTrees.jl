import scipy.stats
import pandas

input_file = "./siemens/TURBOEXPO-regression-v8-analysis-first/plotdescription-Siemens-Data-Measures,stump_with_memoization,(4.0 => High-Risk,Inf => Low-Risk),10,(1,1),60,10,Any[]-sub-values.csv"
class_names = ["High-Risk", "Low-Risk"] # TODO read :class_names column instead

assert len(class_names) == 2

# TODO assert all([!c1.endswith(c2) for (i,c1) in enumerate(class_names) for c2 in class_names[i+1:end]])

df = pandas.read_csv(input_file)

# assert nrow(df) == len(class_names)

canonical_features = list(df.columns)

alpha = 0.05

for feature in canonical_features:
	print("TEST ", end='')
	print(feature)
	df_feature = df[feature]
	some_test_failed = False
	class_serie_is_normal = []
	series = []
	for i_c in range(0,len(class_names)):
		serie = eval(df_feature[i_c])
		series.append(serie)
		# print(serie)
		try:
			stats = scipy.stats.shapiro(serie)

			pval = stats.pvalue
			normal = pval > alpha

			# print()
			# print(serie)
			print("\t", end='')
			print(class_names[i_c])
			if normal:
				print("NORMALITY")
				print("\t\t", end='')
				print(pval, end='')
				print(", ", end='')
				print(normal)
				class_serie_is_normal.append(True)
			else:
				print("NON-NORMAL ()", end='')
				print(pval, end='')
				print(")")
				class_serie_is_normal.append(False)
		except Exception as e:
			print("Failure on serie:")
			# print(serie)
			print(e)
			input()
			some_test_failed = True

	if some_test_failed:
		print("NOTE: some tests failed")

	# len(unique(class_serie_is_normal)) == 1
	if some_test_failed or False in class_serie_is_normal:
		s1 = series[0]
		s2 = series[1]
		# density(s1)
		# display(density!(s2))

		# stats = scipy.stats.wilcoxon(s1, s2)
		stats = scipy.stats.mannwhitneyu(s1, s2)
		# stats = scipy.stats.mannwhitneyu(s1, s2, use_continuity=False)
		# stats = scipy.stats.mannwhitneyu(s1, s2, use_continuity=True, alternative='two-sided', axis=0, method = "exact")
		pval = stats.pvalue
		print("pval =", end='')
		print(pval)
		# input()
	else:
		print("T-TEST:")
		println("TODO expand code. which version of the t-test?")
