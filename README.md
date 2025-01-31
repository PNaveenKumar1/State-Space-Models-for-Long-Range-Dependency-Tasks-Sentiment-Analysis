# Learning to Forecast under External Influence Using Temporal Knowledge Graphs: Datasets, Methods and Benchmarks

## Overview
This project addresses the challenge of forecasting stock market trends by constructing a **Temporally-Aware Heterogeneous Knowledge Graph (SmKG)**. SmKG captures inter-stock relationships, financial statements, analyst sentiments, corporate events, and macroeconomic factors from stock assets listed on NASDAQ and NSE markets. By leveraging time-aware embeddings, this project ranks stock assets based on their potential return on investment over various holding periods.

## Data Sources

The datasets used in this project include :
- NASDAQ100, S&P500 (from Yahoo Finance)
- NIFTY500 (from NSE)

## Requirements

To run this project, you’ll need to install the required dependencies. These can be found in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
The `requirements.txt` contains the necessary Python libraries, such as:
- `torch` for PyTorch
- `numpy` for numerical operations
- `pandas` for data manipulation
- Any other libraries relevant to your project


## Running the Project

There are two ways to run the project, depending on your needs.

### Option 1: Without Temporal Point Process
1. Running without Temporal Point Process, Navigate to Phase-Stock-KG folder. which include main.py file along with model_imports.py and all neccesary model files in the models folder.
2. Main parameters are ENCODER_LAYER which is transf and lstm for getting the sequential embeddings
3. For the Graph Convolutions USE_GRAPH parameter is hgat with includes the HEAT Convolution
4. Change the INDEX according to the dataset want to run "nasdaq100, sp500, nifty500"
4. Ensure the dataset is prepared and placed in the required directory.
5. Run the command:
   ```bash
   python3 main.py
   ```
6. Remaining all the parameters are placed in the model_imports.py


### Option 2: With Temporal Point Process
1. To run the model with Temporal Point Process with the specified parameters, Navigate to Stock_KG_TPP folder. which includes the main_TPP.py along the neccessary model files in the models folder. use the following command: 
2. Prepare the dataset as per Type 2 requirements.
3. Run the command:
   ```bash
   python3 main_TPP.py --tpp_batch_size 4096 --INDEX nasdaq100 --D_MODEL 20 --REL_EMB 16 --TPP_EMB 128 --gpu 1 --ENCODER_LAYER transf
   ```
- --tpp_batch_size 4096: Specifies the batch size for training the Temporal Point Process (TPP) model in phased manner.
- --INDEX nasdaq100: This indicates that the model will use data from the nasdaq100 index. You can replace nasdaq100 with any other index or dataset name as per your use case. 
- --D_MODEL 20: This sets the model dimension (D_MODEL) to 20. It controls the size of the sequential embedding vectors used in the model.
- --REL_EMB 16: Specifies the size of the relative embedding. A value of 16 means that the model will use relation embeddings of size 16 for the temporal relations.
- --TPP_EMB 128: Sets the Temporal Point Process each node and relation embedding size to 128. This embedding will be used for learning the temporal structure in the data.
- --ENCODER_LAYER transf: Specifies the type of encoder layer used in the model. In this case, it's set to transf, which means a Transformer-based encoder layer will be used.

## Data Preparation

Make sure the data is formatted as described below:
- Data files used for the Encoder model for sequential embedding are avialable at Stock_Market_Hawkes/Phase-Stock-KG/data/pickle folder for each dataset
- Data files used for the convolution are available at the Stock_Market_Hawkes/Phase-Stock-KG/kg/tkg_create with temporal_kg_nifty.pkl for NIFTY and temporal_kg.pkl for the NASDAQ and S&P

## Conclusion
