# RankEF
## Installation
The code requires some dependencies as specified in `requirements.txt`. Please follow the relevant libraries to install or run: 

```
pip install -r requirements.txt
```
We are using transformers version 4.33.1, make sure you install the same version as us!

## 1. Data Construction
### Dataset
You can download the APPS dataset [here](https://github.com/hendrycks/apps) to finetune base models.
### Finetune Base Models
First fine tune the base models on the APPS dataset by running the following code:
```
python train_base_model.py
```
### Sample Code Candidates
The fine-tuned model was then used to sample code candidates for subsequent construction of RankEF's dataset. Run the following code:
```
python generate.py
```
### Obtain Execution Feedback
Obtain Execution Feedback to build RankEF dataset by runing:
```
cd metric
bash test_one_solution.sh
```
## 2. Train RankEF
We take three multi-task training approaches to train RankEF by runing:

Hard Parameter Sharing
```
python train_hard.py
```
Soft Parameter Sharing
```
python train_soft.py
```
Intermediate Fine-tuning(INF)
```
python train_gen.py
python train_encoder.py
```
## 3. Rank Code Candidates
Finally, using RankEF to rank the code candidates
```
python test_RankEF.py
```
