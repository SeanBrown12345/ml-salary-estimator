.PHONY: all clean

all: report/income_level_predictor_report.html report/income_level_predictor_report.pdf

# download and extract data
data/raw/adult.zip:
	python scripts/data_download.py \
		--url "https://archive.ics.uci.edu/static/public/2/adult.zip" \
		--destination_path data/raw

# clean and split data into train and test sets
data/processed/train.data data/processed/test.data: data/raw/adult.zip
	python scripts/clean_split_data.py \
		--input-path data/raw/adult.data \
		--output-train-path data/processed/train.data \
		--output-test-path data/processed/test.data \
		--test-size 0.6 \
		--random-state 123 \
		--target-col income

# perform eda and save plots
results/figures/categorical_feature_bars.png \
results/figures/correlation_bubble.png \
results/figures/numeric_feature_histograms.png: \
	data/processed/train.data
	python scripts/eda.py \
		--input_path data/processed/train.data \
		--output_path results/figures


# train model
results/figures/precision_recall_curve.png \
results/figures/confusion_matrix.png \
results/tables/classification_report.csv \
results/tables/tuning_results.csv \
results/tables/baseline_comparison.csv: \
	data/processed/train.data \
	data/processed/test.data
	python scripts/model_training.py \
		--train data/processed/train.data \
		--test data/processed/test.data \
		-o results/

# build HTML and pdf report 
# build HTML and pdf report
report/income_level_predictor_report.html report/income_level_predictor_report.pdf : report/income_level_predictor_report.qmd \
	report/references.bib \
	results/figures/categorical_feature_bars.png \
	results/figures/correlation_bubble.png \
	results/figures/numeric_feature_histograms.png \
	results/figures/precision_recall_curve.png \
	results/figures/confusion_matrix.png \
	results/tables/classification_report.csv \
	results/tables/tuning_results.csv \
	results/tables/baseline_comparison.csv
	quarto render report/income_level_predictor_report.qmd --to html
	quarto render report/income_level_predictor_report.qmd --to pdf



# clean generated files
clean:
	rm -rf \
		data/raw/* \
		data/processed/train.data \
		data/processed/test.data \
		results/figures/*.png \
		results/tables/*.csv \
		report/income_level_predictor_report.html \
		report/income_level_predictor_report.pdf