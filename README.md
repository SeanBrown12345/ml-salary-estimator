# Predicting Income Levels Using the Adult Census Dataset

## Contributors
- Sean Brown
- Tiantong Yin
- Yanxin Liang
- Siting Wang

Report URL: https://seanbrown12345.github.io/ml-salary-estimator/


## About
This project aims to build a predictive machine learning model that determines whether an individual’s annual income exceeds \$50K based on demographic and employment-related features. Using the Adult Census Income Dataset from the UCI Machine Learning Repository, we explore socioeconomic patterns and evaluate classification models to understand which features are most strongly associated with higher income levels.

This repository is created as part of the DSCI 522 group project (Milestone 1), focusing on developing a reproducible, well-structured, and collaborative data analysis workflow.


## Data

This project uses the Adult Census Income dataset from the 
UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/2/adult).

The following raw files are included for full reproducibility:
- `adult.data` – training data  
- `adult.test` – test data  
- `adult.names` – feature documentation  
- `old.adult.names` – legacy attribute file  
- `Index` – dataset index file  

Each row represents information recorded from a 1994 U.S. census population survey.  
The target variable is **income**, categorized as `<=50K` or `>50K`.


## Usage

Follow the steps below to reproduce the analysis.

### Setup

1. Make sure Docker Desktop is running, then clone this repo.

### Usage

2. Navigate to the root of this project, then run this command:
```bash
docker compose up
```


3. In the terminal, look for the URL that looks something like `http://127.0.0.1:8888/lab?token=` , copy and paste that URL into your browser.



4. In the jupyter lab session that just launched, open `src/income_level_predictor_report.ipynb`. In the top menu under "Kernel", click "Restart Kernel and Run All Cells..."


### Clean up

1. Go back to the terminal where docker compose is running,  then
type `Cntrl` + `C` in the terminal where you launched the container, and then type `docker compose rm` to remove the stopped container.

## Dependencies

All dependencies required to run this project are listed in the `environment.yml` file, which ensures a fully reproducible computational environment. Key packages include:

- Python 3.12  
- pandas  
- scikit-learn  
- ucimlrepo  
- matplotlib  
- altair & vegafusion  
- pyarrow  
- jupyter  
- conda-lock  

These dependencies will be installed automatically when running:

```bash
conda env create --file environment.yml
```
## Adding Dependencies

1. In a new branch, add the dependency to the environment.yml file.

2. Run the command `conda-lock -k explicit --file environment.yml` --update to update the `conda-lock.yml` file.

3. Re-build the Docker image.

4. Push the changes to github, the Docker image will be updated automatically. This image will be tagged with the SHA for the commit which changed the file.

5. Update `docker.compose.yml` on your branch to use the newly created container image.

6. Create a PR to merge your branch with main.

## License

The written documentation and report in this repository are licensed under the  
**Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** license.

The source code in this repository is released under the **MIT License**.

For full details, see the `LICENSE` file included in this repository.


## References

- Dua, Dheeru, and Casey Graff. 2019. *UCI Machine Learning Repository: Adult Data Set*. University of California, Irvine, School of Information and Computer Sciences.  
  https://archive.ics.uci.edu/dataset/2/adult

- Kohavi, Ron. 1996. “Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid.” *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD)*.  
  https://dl.acm.org/doi/10.5555/3001460.3001507

- Lichman, Moshe. 2013. *Adult Data Set Documentation*. UCI Machine Learning Repository.  
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names

- UCI Machine Learning Repository. “Adult Data Set – Original Sources and Description.”  
  https://archive.ics.uci.edu/ml/datasets/Adult
