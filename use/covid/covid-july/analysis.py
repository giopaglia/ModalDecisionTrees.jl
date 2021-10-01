# import seaborn library
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express
import pandas.plotting
from scipy.stats import t
from pandas_profiling import ProfileReport

models = [
	"T(entropy,2,0.01,0.2)",
	"T(entropy,2,0.01,0.6)",
	"T(entropy,4,0.01,0.2)",
	"T(entropy,4,0.01,0.6)",
	"T(,2,0.01,0.2)",
	# "T(,2,0.01,0.6)",
	"T(,4,0.01,0.2)",
	# "T(,4,0.01,0.6)",
	"RF(50,half,all)",
	"RF(100,half,all)",
]

metrics = [
	# "K",
	"accuracy",
	# "safe_macro_sensitivity",
	# "safe_macro_specificity",
	# "safe_macro_PPV",
	# "safe_macro_NPV",
	# "safe_macro_F1",
	# "t",
]

# load the dataset
# data = sns.load_dataset('tips')
data = pd.read_csv('full_columns.tsv', sep='\t')

print("Original columns:", data.columns)

selected_head_columns = [
	"Dataseed",
	"use_training_form",
	"n_task_use_aug",
	"n_version",
	"nbands",
	"dataset_kwargs",
	"use_full_mfcc",
	"preprocess_wavs",
	"test_operators",
]
selected_tail_columns = [model+metric for model in models for metric in metrics]
selected_columns = selected_head_columns + selected_tail_columns
print("Selecting columns: ", selected_columns, "...")
for sel_col in selected_columns:
	if not sel_col in data.columns:
		raise ValueError("Column 'sel_col' not found")
data = data[selected_columns]
# print("Current columns:", list(data.columns))

print("Filtering constant columns...")
data_head = data[selected_head_columns]
data_head = data_head.loc[:, (data_head != data_head.iloc[0]).any()]
data_tail = data[selected_tail_columns]
data = data_head.join(data_tail)
print("Current columns:", list(data.columns))

print("Other fixes...")
col_n_task = data["n_task_use_aug"].apply(lambda x: int(eval(x[0:3] + x[3].upper() + x[4:])[0]))
col_aug    = data["n_task_use_aug"].apply(lambda x: int(eval(x[0:3] + x[3].upper() + x[4:])[1]))
loc_n_task_use_aug = data.columns.get_loc("n_task_use_aug")
data.insert(loc_n_task_use_aug, "n_task", col_n_task)
data.insert(loc_n_task_use_aug+1, "aug", col_aug)
data = data.drop(columns="n_task_use_aug")

dataset_kwargs_name = {
	"(120,100)"  : 100,
	"(100,75)"   : 75,
	"(75,50)"    : 50,
}
data["dataset_kwargs"] = data["dataset_kwargs"].apply(lambda x: dataset_kwargs_name[x])

# for model in models:
# 	for metric in metrics:
# 		col_n_task = data["n_task_use_aug"].apply(lambda x: int(eval(x[0:3] + x[3].upper() + x[4:])[0]))
# 		col_aug    = data["n_task_use_aug"].apply(lambda x: int(eval(x[0:3] + x[3].upper() + x[4:])[1]))
# 		loc_n_task_use_aug = data.columns.get_loc("n_task_use_aug")
# 		data.insert(loc_n_task_use_aug, "n_task", col_n_task)
# 		data.insert(loc_n_task_use_aug+1, "aug", col_aug)
# 		data = data.drop(columns="n_task_use_aug")
print("Current columns:", list(data.columns))

# print(data.index)

# data = pd.MultiIndex
# index = pd.MultiIndex.from_tuples([], names=selected_head_columns)

# Source: https://stackoverflow.com/questions/6486450/compute-list-difference
def diff(first, second):
	second = set(second)
	return [item for item in first if item not in second]


data = data.set_index(diff(data.columns, selected_tail_columns))

new_indices = [(metric, model) for model in models for metric in metrics]
print(new_indices)
data.columns = pd.MultiIndex.from_tuples(new_indices, names=["Metric", "Model"])

print(data)

data = data.rename(columns={
	"n_task"     : "Task",
	"n_version"  : "Version",
})

# view the dataset
print(data.head(5))
# print(data['Params-combination'])
# print(data.describe())
print(data.info())

print(data.columns)


# format_dict = {} #Simplified format dictionary with values that do make sense for our data
# print(data.head().style.format(format_dict).highlight_max(color='darkgreen').highlight_min(color='#ff0000'))

# data.head().style.format(format_dict).highlight_max(color='darkgreen').highlight_min(color='#ff0000').render()

# prof = ProfileReport(data, explorative=True)
# prof = ProfileReport(data)
# prof.to_file(output_file="report.html")


# https://stackoverflow.com/questions/51078388/how-to-melt-first-level-column-in-multiindex-with-pandas
data = data.stack(-1)

print(data)
print(data.info())

data = data.reset_index()

print(data)
print(data.info())


# data = data.groupby(diff(data.columns, selected_tail_columns+["Dataseed"]), as_index=False).agg({metric:'mean' for metric in metrics})
data = data.groupby(diff(data.columns, selected_tail_columns+["Dataseed"]+metrics), as_index=False).mean()
# data = data[diff(data.columns, selected_tail_columns+["Dataseed"]+metrics)]
data = data.drop(columns="Dataseed")
data.to_csv('full_columns-agg.tsv', sep='\t')

print(data)
print(data.info())

data.insert(data.columns.get_loc("Model"), "Model_id", data["Model"].apply(lambda x: models.index(x)))

print(data)
print(data.info())

# fig = plotly.express.parallel_coordinates(data[diff(data.columns, selected_tail_columns+metrics)],
fig = plotly.express.parallel_coordinates(data,
					dimensions = diff(data.columns, selected_tail_columns),
					# color="Model_id",
					color="accuracy",
          range_color = [50, 100],
					labels={"Model_id" : "Model"},
          # labels=dict(zip(list(data.columns), 
          # list(['_'.join(i.split('_')[1:]) for i in data.columns]))),
          # color_continuous_scale=plotly.express.colors.diverging.Tealrose,
          # color_continuous_midpoint=27
          )
fig.show()


# fig = pandas.plotting.parallel_coordinates(data, "Model",
# 					color=["#849483","#4e937a","#b4656f","#a47481","#9c7c8a","#98808e","#948392","#c7f2a7"], 
#           # cols = 
#           # labels=dict(zip(list(data.columns), 
#           # list(['_'.join(i.split('_')[1:]) for i in data.columns]))),
#           )

# plt.yscale('log')
# plt.show()

exit()

data = data.pivot(index=['Task', 'Version', 'Dataseed'], columns=['Params-combination'])

# print(data.columns.tolist())
# print(data.head(5))
# input()
data = data.append(data.groupby(['Dataset']).mean())

# data = data[cols]
# print(data.head(5))
# input()
cols = [
('T(entropy,4,0.01,0.3)K', '((1,7),false,"o_None","TestOpGeq")'),
('T(entropy,4,0.01,0.3)accuracy', '((1,7),false,"o_None","TestOpGeq")'),
('T(entropy,4,0.01,0.3)safe_macro_PPV', '((1,7),false,"o_None","TestOpGeq")'),
('T(entropy,4,0.01,0.3)safe_macro_F1', '((1,7),false,"o_None","TestOpGeq")'),
# ('T(entropy,4,0.01,0.3)t', '((1,7),false,"o_None","TestOpGeq")'),
('T(entropy,4,0.01,0.3)K', '((3,7),:flattened,"o_None","TestOpGeq")'),
('T(entropy,4,0.01,0.3)accuracy', '((3,7),:flattened,"o_None","TestOpGeq")'),
('T(entropy,4,0.01,0.3)safe_macro_PPV', '((3,7),:flattened,"o_None","TestOpGeq")'),
('T(entropy,4,0.01,0.3)safe_macro_F1', '((3,7),:flattened,"o_None","TestOpGeq")'),
# ('T(entropy,4,0.01,0.3)t', '((3,7),:flattened,"o_None","TestOpGeq")'),
# ('T(entropy,4,0.01,0.3)K', '(3,:averaged,"o_None","TestOpGeq")'),
# ('T(entropy,4,0.01,0.3)accuracy', '(3,:averaged,"o_None","TestOpGeq")'),
# ('T(entropy,4,0.01,0.3)safe_macro_PPV', '(3,:averaged,"o_None","TestOpGeq")'),
# ('T(entropy,4,0.01,0.3)safe_macro_F1', '(3,:averaged,"o_None","TestOpGeq")'),
# ('T(entropy,4,0.01,0.3)t', '(3,:averaged,"o_None","TestOpGeq")'),
('T(entropy,4,0.01,0.3)K', '((3,7),false,"o_RCC8","TestOp")'),
('T(entropy,4,0.01,0.3)accuracy', '((3,7),false,"o_RCC8","TestOp")'),
('T(entropy,4,0.01,0.3)safe_macro_PPV', '((3,7),false,"o_RCC8","TestOp")'),
('T(entropy,4,0.01,0.3)safe_macro_F1', '((3,7),false,"o_RCC8","TestOp")'),
# ('T(entropy,4,0.01,0.3)t', '((3,7),false,"o_RCC8","TestOp")'),
('T(entropy,4,0.01,0.3)K', '((3,7),false,"o_RCC5","TestOp")'),
('T(entropy,4,0.01,0.3)accuracy', '((3,7),false,"o_RCC5","TestOp")'),
('T(entropy,4,0.01,0.3)safe_macro_PPV', '((3,7),false,"o_RCC5","TestOp")'),
('T(entropy,4,0.01,0.3)safe_macro_F1', '((3,7),false,"o_RCC5","TestOp")'),
# ('T(entropy,4,0.01,0.3)t', '((3,7),false,"o_RCC5","TestOp")'),
('T(entropy,4,0.01,0.3)K', '((3,7),false,"o_RCC8","TestOpAll")'),
('T(entropy,4,0.01,0.3)accuracy', '((3,7),false,"o_RCC8","TestOpAll")'),
('T(entropy,4,0.01,0.3)safe_macro_PPV', '((3,7),false,"o_RCC8","TestOpAll")'),
('T(entropy,4,0.01,0.3)safe_macro_F1', '((3,7),false,"o_RCC8","TestOpAll")'),
# ('T(entropy,4,0.01,0.3)t', '((3,7),false,"o_RCC8","TestOpAll")'),
('T(entropy,4,0.01,0.3)K', '((3,7),false,"o_RCC5","TestOpAll")'),
('T(entropy,4,0.01,0.3)accuracy', '((3,7),false,"o_RCC5","TestOpAll")'),
('T(entropy,4,0.01,0.3)safe_macro_PPV', '((3,7),false,"o_RCC5","TestOpAll")'),
('T(entropy,4,0.01,0.3)safe_macro_F1', '((3,7),false,"o_RCC5","TestOpAll")'),
# ('T(entropy,4,0.01,0.3)t', '((3,7),false,"o_RCC5","TestOpAll")'),
]

data = data.reindex(cols, axis=1)

# print(data.columns)
# print(data)
# print(data.swaplevel())
# data.style
# input()
data.to_csv('full_columns-pivoted.tsv', sep='\t')



# load the dataset
# data = sns.load_dataset('tips')
data = pd.read_csv('full_columns.tsv', sep='\t')
data = data.rename(columns={"dataset_name": "Dataset", "windowsize_flattened_ontology_test_operators" : "Params-combination"})

# view the dataset
print(data.head(5))
# print(data['Params-combination'])

data = data.pivot(index=['Dataseed'], columns=['Dataset','Params-combination'])

# print(data.columns)
# print(data)
# data.style
# input()
# data.to_csv('full_columns-pivoted.tsv', sep='\t')

n_classes = {
	"IndianPines"             : 12,
	"PaviaUniversity"         : 9,
	"PaviaCentre"             : 9,
	"Salinas"                  : 16,
	"Salinas-A"                : 6,
}
inst_per_class = 100
split_theshold = 0.8

datasets = [
		"IndianPines",
		"PaviaUniversity"      ,
		"PaviaCentre",
		"Salinas"    ,
		"Salinas-A"  ,
	]
# pairs_to_t_test = [
# 		('((1,7),false,"o_None","TestOpGeq")', '((3,7),false,"o_RCC8","TestOp")'),
# 		('((3,7),:flattened,"o_None","TestOpGeq")', '((3,7),false,"o_RCC8","TestOp")'),
# 		('(3,:averaged,"o_None","TestOpGeq")', '((3,7),false,"o_RCC8","TestOp")'),
# 		('((1,7),false,"o_None","TestOpGeq")', '((3,7),false,"o_RCC5","TestOp")'),
# 		('((3,7),:flattened,"o_None","TestOpGeq")', '((3,7),false,"o_RCC5","TestOp")'),
# 		('(3,:averaged,"o_None","TestOpGeq")', '((3,7),false,"o_RCC5","TestOp")'),
# 		('((3,7),false,"o_RCC5","TestOp")', '((3,7),false,"o_RCC8","TestOp")'),
# 	]

groups_of_pairs_to_t_test = [
		# RCC5+
		[
			('((1,7),false,"o_None","TestOpGeq")', '((3,7),false,"o_RCC5","TestOpAll")'),
			('((3,7),:flattened,"o_None","TestOpGeq")', '((3,7),false,"o_RCC5","TestOpAll")'),
			('((3,7),false,"o_RCC8","TestOp")', '((3,7),false,"o_RCC5","TestOpAll")'),
			('((3,7),false,"o_RCC5","TestOp")', '((3,7),false,"o_RCC5","TestOpAll")'),
			('((3,7),false,"o_RCC8","TestOpAll")', '((3,7),false,"o_RCC5","TestOpAll")'),
		],[
		# RCC8+
			('((1,7),false,"o_None","TestOpGeq")', '((3,7),false,"o_RCC8","TestOpAll")'),
			('((3,7),:flattened,"o_None","TestOpGeq")', '((3,7),false,"o_RCC8","TestOpAll")'),
			('((3,7),false,"o_RCC8","TestOp")', '((3,7),false,"o_RCC8","TestOpAll")'),
			('((3,7),false,"o_RCC5","TestOp")', '((3,7),false,"o_RCC8","TestOpAll")'),
		],[
		# RCC5
			('((1,7),false,"o_None","TestOpGeq")', '((3,7),false,"o_RCC5","TestOp")'),
			('((3,7),:flattened,"o_None","TestOpGeq")', '((3,7),false,"o_RCC5","TestOp")'),
			('((3,7),false,"o_RCC8","TestOp")', '((3,7),false,"o_RCC5","TestOp")'),
		],[
		# RCC8
			('((1,7),false,"o_None","TestOpGeq")', '((3,7),false,"o_RCC8","TestOp")'),
			('((3,7),:flattened,"o_None","TestOpGeq")', '((3,7),false,"o_RCC8","TestOp")'),
		],[
		# flattened
			('((1,7),false,"o_None","TestOpGeq")', '((3,7),:flattened,"o_None","TestOpGeq")'),
		]
		#
	]

# print(data['T(entropy,4,0.01,0.3)accuracy'])
# print(data['T(entropy,4,0.01,0.3)accuracy','((1,7),false,"o_None","TestOpGeq")'])
# print(data['T(entropy,4,0.01,0.3)accuracy','((3,7),false,"o_RCC8","TestOp")'])

for group_of_pairs_to_t_test in groups_of_pairs_to_t_test:
	for dataset in datasets:
		for pair_to_t_test in group_of_pairs_to_t_test:
			
			len_train = inst_per_class*n_classes[dataset]*split_theshold
			len_test  = inst_per_class*n_classes[dataset]*(1-split_theshold)
			len_total = inst_per_class*n_classes[dataset]

			col1, col2 = pair_to_t_test

			cur_data = data['T(entropy,4,0.01,0.3)accuracy']
			cur_data = cur_data.get(dataset, None)
			if cur_data is None:
				continue
			scores1, scores2 = cur_data.get(col1, default=None), cur_data.get(col2, default=None)
			if scores1 is None or scores2 is None:
				continue
			
			# Using the formula from
			#  Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms
			#  Section 3.2
			
			#Compute the difference between the results
			diff = [y - x for y, x in zip(scores1, scores2)]
			#Compute the mean of differences
			d_bar = np.mean(diff)
			#compute the variance of differences
			sigma2 = np.var(diff, ddof=1)
			#compute the number of data points used for training 
			n1 = len_train
			#compute the number of data points used for testing 
			n2 = len_test
			#compute the total number of data points
			n = len(diff)
			#compute the modified variance
			sigma2_mod = sigma2 * (1/n + n2/n1)
			#compute the t_static
			t_static =  d_bar / np.sqrt(sigma2_mod)
			
			t_static = np.abs(t_static)

			#Compute p-value and plot the results 
			# Pvalue = ((1 - t.cdf(t_static, n-1))*2.0)
			Pvalue_perc = ((1 - t.cdf(t_static, n-1))*200)
			print(dataset,pair_to_t_test,Pvalue_perc,"%")


m = {
	'((1,7),false,"o_None","TestOpGeq")'       : "propositional",
	'((3,7),:flattened,"o_None","TestOpGeq")'  : "flattened",
	# '(3,:averaged,"o_None","TestOpGeq")'       : "averaged",
	'((3,7),false,"o_RCC8","TestOp")'          : "RCC8",
	'((3,7),false,"o_RCC5","TestOp")'          : "RCC5",
	'((3,7),false,"o_RCC8","TestOpAll")'       : "RCC8+",
	'((3,7),false,"o_RCC5","TestOpAll")'       : "RCC5+",
}

d = {
	"IndianPines" : "Indian Pines",
	"Pavia"       : "Pavia University"      ,
	"PaviaCentre" : "Pavia Centre",
	"Salinas"     : "Salinas"    ,
	"Salinas-A"   : "Salinas-A"  ,
}

# load the dataset
# data = sns.load_dataset('tips')
data = pd.read_csv('full_columns.tsv', sep='\t')
data = data.rename(columns={"dataset_name": "Dataset", "windowsize_flattened_ontology_test_operators" : "Params-combination"})
data['Params-combination'] = data['Params-combination'].apply(lambda x: m[x])
data['Dataset'] = data['Dataset'].apply(lambda x: d[x])

# d = data.Dataset.isin(["Pavia Centre", "Pavia University", "Indian Pines"])
# data = data[d]

# view the dataset
print(data.head(5))
# print(data['Params-combination'])

data = data.rename(columns={'Params-combination': 'Method'})

# create grouped boxplot
ax = sns.boxplot(x = 'Dataset',
			y = 'T(entropy,4,0.01,0.3)accuracy',
			hue = 'Method',
			data = data,
			order=["Indian Pines", "Pavia University", "Pavia Centre", "Salinas", "Salinas-A", ],
			palette="pastel",
			# palette=["#F1A130", "#F19330", "#F18730", "#F16430", "#F14630"],
			# palette="flare",
			# palette="Set2",
			# palette=sns.color_palette("flare", as_cmap=True),
			# palette=["r", "g", "b", "black", "white"],
			)

ax.set(ylabel="Accuracy (%)")
# ax = sns.swarmplot(x="Dataset", y="T(entropy,4,0.01,0.3)accuracy", data=data, color=".25")
# plt.get_legend().remove()

# legend = plt.legend()
# legend.remove()

fig1 = plt.gcf()
plt.show()
fig1.savefig("img/results.png")
plt.draw()
