# Bias & Fairness in NLP
author: samantha dobesh
date:   feb 7th. 2022

# Measuring Fairness with Bert Embeddings

Based on the paper **Unmasking Contextual Stereotypes**: https://arxiv.org/pdf/2010.14534.pdf

## 1. Measure Bias

**Evaluation Dataset**
Bias Evaluation Corpus with Professions (BEC-pro)

- "**\[MASK\]**, the **\[MASK\]**, had a good day at work."
- "**\[T\]**, the **\[A\]**, had a good day at work."
- P(T|A)

### sentence examples
1. **\<person>** is a **\<profession>**.
2. **\<person>** works as a **\<profession>**.
3. **\<person>** applied for the position of **\<profession>**.
4. **\<person>**, the **\<profession>**, had a good day at work.
5. **\<person>** wants to become a **\<profession>**.

### masking examples
- **original:** My son is a medical records technician.
- **T masked:** My **\[MASK\]** is a medical records technician.
- **A masked:** My son is **\[MASK\] \[MASK\] \[MASK\]**.
- **T+A Masked:** My **\[MASK\]** is a **\[MASK\] \[MASK\] \[MASK\]**.

### calculating probability
1. Take a sentence with a target and attribute word	
	- "He is a kindergarten teacher."
2. Mask the target word
	-  "**\[MASK\]** is a kindergarten teacher."
3. Obtain the probability of the target word appearing in the sentence.
	- pT = P(he = **\[MASK\]** | sent)
4. Mask both target and attribute word. In compounds, mask coponents seperately.
	- "**\[MASK\]** is a **\[MASK\] \[MASK\]**"
5. Obtain the prior probability, the probability of the target word when the attribute is masked.
	- pPrior = P(he = **\[MASK\]** | masked_sent)
6. Caclulate the association by dividing the target probability by the prior and take the natural log.
	- ln(pT / pPrior)

A negative association between a target and an attribute means that the probability of the target is lower than the prior probability. The probability of the target word *decreases* due to the attribute. A positive association indicates the attribute makes the target word likelyhood *increase*.


## 2. Fine-Tuning
### Fine-Tuning Dataset: GAP corpus
**example:** The historical Octavia Minor’s first husband was Gaius Claudius Marcellus Minor, and she bore him three children, Marcellus, Claudia Marcella Major and **\[Claudia Marcella Minor\]**; the **\[Octavia\]** in Rome is married to a nobleman named Glabius, with whom **\[she\]** has no children.

## 3. Analysis
### example results
| hypothesis | expected observation |
| :-- | :-- |
| There is a strong association of female (male) person-denoting noun phrases (NPs) with statistically female (male) professions, which is reduced through fine-tuning. | Positive association scores between female (male) NPSs and statistically female (male) professions, which decrease after fine tuning. |_
| There is a weak association of female (male) NPs with statistically male (female) professions, which is strengthened through fine-tuning. | Negative association scores between female (male) NPs and statistically male (female) professions, which increase after fine-tuning. |
| There is no difference between the associations of female and male person-denoting NPs with statistically gender-balanced professions. Associations do not change much after fine-tuning. | Both association scores of female and male NPs have approx. the same value, which is likely located around zero. After fine-tuning, the association score |

| | | pre | post | diff. | Wilcoxon test | | 
| :-- | :-- | :-- | :-- | :-- | :-- |:-- |
| jobs | person | mean | mean | mean | W | r |
| B | f | -0.35 | 0.20 | 0.55 | 359188 | -0.47 |
| B | m | 0.05 | 0.07 | 0.01 | 359188 | -0.47 |
| F | f | 0.50 | 0.36 | -0.14 | 96428 | -0.32 |
| F | m | -0.68 | -0.14 | 0.55 | 96428 | -0.32 |
| M | f | -0.83 | 0.13 | 0.96 | 395974 | -0.58 |
| M | m | 0.16 | 0.21 | 0.05 | 395974 | -0.58 |
# Fairness.JL
- gender fairness on compas data set
- intersectional fairness

**julia :** https://julialang.org/downloads/

##  simple synthetic dataset
```julia
julia> using Fairness

julia> ŷ = categorical([1, 0, 1, 1, 0]);

julia> y = categorical([0, 0, 1, 1, 1]);

julia> grp = categorical(["Asian", "African", "Asian", "American", "African"]);

julia> ft = fair_tensor(ŷ, y, grp);
```
## evaluate disparity
```julia

julia> M = [true_positive_rate, positive_predictive_value];

julia> df = disparity(M, ft; refGrp="Asian");
┌──────────┬────────────────────────────┬─────────────────────┐
│ labels   │ TruePositiveRate_disparity │ Precision_disparity │
│ String   │ Float64                    │ Float64             │
│ Textual  │ Continuous                 │ Continuous          │
├──────────┼────────────────────────────┼─────────────────────┤
│ African  │ 1.0e-15                    │ 0.0                 │
│ American │ 1.0                        │ 0.0                 │
│ Asian    │ 1.0                        │ 1.0                 │
└──────────┴────────────────────────────┴─────────────────────┘

julia> f(x, y) = x - y
f (generic function with 1 method)


julia> df_1 = disparity(M, ft; refGrp="Asian", func=f);
┌──────────┬────────────────────────────┬─────────────────────┐
│ labels   │ TruePositiveRate_disparity │ Precision_disparity │
│ String   │ Float64                    │ Float64             │
│ Textual  │ Continuous                 │ Continuous          │
├──────────┼────────────────────────────┼─────────────────────┤
│ African  │ -1.0                       │ -0.5                │
│ American │ 0.0                        │ -0.5                │
│ Asian    │ 0.0                        │ 0.0                 │
└──────────┴────────────────────────────┴─────────────────────┘
```
## real datasets
### COMPAS - @load_compas
Macro to load [COMPAS dataset](https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb) It is a reduced version of COMPAS Datset with 8 features and 6907 rows. The protected attributes are sex and race. The available features are used to predict whether a criminal defendant will recidivate(reoffend).

### ADULT - @load_adult
Macro to Load the [Adult dataset](https://archive.ics.uci.edu/ml/datasets/adult) It has 14 features and 32561 rows. The protected attributes are race and sex. This dataset is used to predict whether income exceeds 50K dollars per year.
 

```julia
julia> using Fairness

julia> X, y = @load_compas;

julia> X, y = @load_adult;
```

## evaluate COMPAS
link : https://nextjournal.com/ashryaagr/fairness

### package dependencies
```julia
julia> using Pkg
julia> Pkg.activate("my_environment", shared=true)
julia> Pkg.add("Fairness")
julia> Pkg.add("MLJ") # Toolkit for Machine Learning
julia> Pkg.add("PrettyPrinting") # For readibility of outputs
julia> using Fairness
julia> using MLJ
julia> using PrettyPrinting
```

### load COMPAS
```julia
julia> data = @load_compas
julia> X, y = data
julia> data |> pprint
```

### load classifier
```julia
julia> Pkg.add("MJLFlux")
julia> @load NeuralNetworkClassifier
julia> model = @pipeline ContinuousEncoder NeuralNetworkClassifier
```

### fairness algorithm wrapper
```julia
julia> wrappedModel = ReweighingSamplingWrapper(classifier=model, grp=:race)

julia> wrappedModel2 = LinProgWrapper(classifier=wrappedModel, grp=:race, measure=false_positive_rate)
```

### evaluate
```julia
julia> evaluate(
	wrappedModel2, 
	X, y, 
	measures=[
		Disparity(false_positive_rate, refGrp="Caucasian", grp=:race),
		MetricWrapper(accuracy, grp=:race)
	]) |> pprint
)
```

### fine control
```julia
julia> train, test = partition(eachindex(y), 0.7, shuffle=true)
julia> mach = machine(wrappedModel2, X, y)
julia> fit!(mach, rows=train)
julia> ŷ = predict(mach, rows=test)
julia> ŷ |> pprint
```
#### fairness tensor
[fairness tensor](https://ashryaagr.github.io/Fairness.jl/dev/fairtensor/)

```julia
julia> ft = fair_tensor(ŷ, y[test], X[test, :race])
```

### disparity
```julia
julia> df_disparity = disparity(
  [accuracy, false_positive_rate], 
  ft, refGrp="Caucasian",
  func=(x, y)->(x-y)/y
)
```
| labels | Accuracy_disparity | FalsePositiveRate_disparity |
| :-- | :-- | :-- |
| African-American | 0.182352728194359 | -0.10329691532209435 |
| Asian | 0.22193658954584394 | -0.09541381128097005 |
| Caucasian | 0.0 | 0.0 |
| Hispanic | 0.05468828539012794 | -0.05958861568813718 |
| Native American | 0.832904884318766 | -1.0 |
| Other| 0.09233725429098169 | -0.16956022019236605 |

### parity calculations
``` julia
julia> parity(df_disparity,
  func= (x) -> abs(x)<0.4
)
```

### visualizing improvements
-   valuate metric values using MLJ.evaluate for both: The Wrapped Model and the original model
-   Collect metric values from the result of evaluate function
-   Create a DataFrame using the collected values that will later be used with VegaLite to plot the graphs

```julia 
julia> Pkg.add("Ipopt")
julia> Pkg.add("VegaLite")
julia> Pkg.add("DataFrames")
julia> using VegaLite
julia> using DataFrames

julia> result = evaluate(wrappedModel2,
    X, y,
    measures=[
    Disparity(false_positive_rate, refGrp="Caucasian", grp=:race),
    MetricWrapper(accuracy, grp=:race)])

fulia> result_1 = evaluate(model,
    X, y,
    measures=[
    Disparity(false_positive_rate, refGrp="Caucasian", grp=:race),
    MetricWrapper(accuracy, grp=:race)])

julia> n_grps = length(levels(X[!, :race]))
julia> dispVals = collect(values(result.measurement[1]))
julia> dispVals_1 = collect(values(result_1.measurement[1]))
julia> accVals = collect(values(result.measurement[2]))
julia> accVals_1 = collect(values(result_1.measurement[2]))

julia> df = DataFrame(
  disparity=vcat(dispVals, dispVals_1),
  accuracy=vcat(accVals, accVals_1),
  algo=vcat(repeat(["Wrapped Model"],n_grps+1), repeat(["ML Model"],n_grps+1)),
  grp=repeat(collect(keys(result.measurement[1])), 2));
```

### add fairness metric
```julia
julia> pred_compas = X.decile_score .>= 5
julia> dispVals_compas = collect(values(Disparity(false_positive_rate, 
                          refGrp="Caucasian", grp=:race)(pred_compas, X, y)))
julia> accVals_compas = collect(values(MetricWrapper(accuracy, grp=:race)(pred_compas, X, y)))
julia> df_compas = DataFrame(
  disparity=dispVals_compas,
  accuracy=accVals_compas,
  algo=repeat(["COMPAS Scores"],n_grps+1),
  grp=repeat(collect(keys(result.measurement[1])), 1));
julia> df = vcat(df, df_compas);
```
### plot
#### improvement in false positive rate disparity
```julia
julia> df |> @vlplot(
  :bar,
  column="grp:o",
  y={"disparity:q",axis={title="False Positive Rate Disparity"}},
  x={"algo:o", axis={title=""}},
  color={"algo:o"},
  spacing=20,
  config={
  view={stroke=:transparent},
  axis={domainWidth=1}
  }
)
```
#### accuracy comparison
```julia
julia> df |> @vlplot(
  :bar,
  column="grp:o",
  y={"accuracy:q",axis={title="Accuracy"}},
  x={"algo:o", axis={title=""}},
  color={"algo:o"},
  spacing=20,
  config={
  view={stroke=:transparent},
  axis={domainWidth=1}
  }
)
```
#### comparison across algorithms
```julia
julia> Pkg.add(Pkg.PackageSpec(;name="FFMPEG", version="0.2.4"))
julia> Pkg.add(Pkg.PackageSpec(;name="Plots", version="1.5.0"))
julia> Pkg.add("PyPlot")
julia> using Plots

julia> function algorithm_comparison(algorithms, algo_names, X, y;
  refGrp, grp::Symbol=:class)
	grps = X[!, grp]
	categories = levels(grps)
	train, test = partition(eachindex(y), 0.7, shuffle=true)
	plot(title="Fairness vs Accuracy Comparison", seriestype=:scatter, 
        xlabel="accuracy", 
        ylabel="False Positive Rate Disparity refGrp="*refGrp,
        legend=:topleft, framestyle=:zerolines)
	for i in 1:length(algorithms)
		mach = machine(algorithms[i], X, y)
		fit!(mach, rows=train)
		ŷ = predict(mach, rows=test)
		if typeof(ŷ) <: MLJ.UnivariateFiniteArray
			ŷ = mode.(ŷ)
		end
		ft = fair_tensor(ŷ, y[test], X[test, grp])
		plot!([accuracy(ft)], [fpr(ft)/fpr(ft, grp=refGrp)], 
      seriestype=:scatter, label=algo_names[i])
	end
	display(plot!())
end

julia> algorithm_comparison([model, wrappedModel, wrappedModel2], 
  ["NeuralNetworkClassifier", "Reweighing(Model)",
    "LinProg+Reweighing(Model)"], X, y, 
    refGrp="Caucasian", grp=:race)
```
#### boxplots
```julia
julia> n_grps = length(levels(X[!, :race]))
julia> dispVals = vcat(collect.(values.(result.per_fold[1]))...)
julia> dispVals_1 = vcat(collect.(values.(result_1.per_fold[1]))...)
julia> accVals = vcat(collect.(values.(result.per_fold[2]))...)
julia> accVals_1 = vcat(collect.(values.(result_1.per_fold[2]))...)

julia> df_folds = DataFrame(
  disparity=vcat(dispVals, dispVals_1),
  accuracy=vcat(accVals, accVals_1),
  algo=vcat(repeat(["Wrapped Model"],6*(n_grps+1)), 
            repeat(["ML Model"],6*(n_grps+1))),
  grp=repeat(collect(keys(result.measurement[1])), 6*2));

julia> df_folds |> @vlplot(
  :boxplot,
  column="grp:o",
  y={"disparity:q",axis={title="False Positive Rate Disparity"}},
  x={"algo:o", axis={title=""}},
  color={"algo:o"},
  spacing=20,
  config={
  view={stroke=:transparent},
  axis={domainWidth=1}
  }
)

julia> df_folds |> @vlplot(
  :boxplot,
  column="grp:o",
  y={"accuracy:q",axis={title="Accuracy"}},
  x={"algo:o", axis={title=""}},
  color={"algo:o"},
  spacing=20,
  config={
  view={stroke=:transparent},
  axis={domainWidth=1}
  }
)
```
