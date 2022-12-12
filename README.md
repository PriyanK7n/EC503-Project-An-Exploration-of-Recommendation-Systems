# EC503 Project: An Exploration of Recommendation Systems


## Datasets Used in Project: 
1. Books Recommendation Dataset
2. Artificial Dataset
                        
          
          
Programming Language Used: Python
## Tools Used: 
- Python Programming language to perform experiments.
- Used Sklearn for Modelling SVD.
- Used Gridsearch for performing hyperparameter tuning. 
- Used matplotlib library to build various plots to analyze and showcase the results of experiments. 
- Pandas and Numpy to perform data-preprocessing


## Hyperparameters: 
- Training and Testing Split: 80%, 20%
- num_of_components or top 'r' singular values =250
- iterations=20


## Data Pre-Processing for Creation of Books-Users Ratings Sparse Matrix?

* Ensured the Users have at least rated more than 300 books
* Ensured the Books had at least 50 ratings from users
* Fill Empty/NAN(Not a Number) values with zeros.
* Remove duplicates(ensures unique users and books)


## Project Structure


    ├── LICENSE
    ├── README.md          <- The top-level README for developers/collaborators using this project. 
    │ 
    │   
    │ 
    │   
    ├── src                <- Source code folder for this project
        │
        ├── data           <- Datasets used and collected for this project
        │   
        ├── docs           <- Folder for Task documentations, Meeting Presentations and task Workflow Documents and Diagrams.
        │
        ├── references     <- Data dictionaries, manuals, and all other explanatory references used 
        │
        ├── visualizations <- Folder to store and Visualization generated for the project
        │
        └── results        <- Folder to store final results and code. 



     
## Folder Overview

- Reports           - Folder to store all Final Reports of this project
- Data              - Folder to Store all the data collected and used for this project 
- References        - Folder to store any referenced code/research papers and other useful documents used for this project
- Visualizations    - Folder to store plots
- Results           - Folder to store Final results and code.

