# TabularDataScience22

This is the repo of the Tabular Data Science Final project 2022
By: Sharon & Elad

Project Report is attached `TabularDataFinalProject22.pdf`

Code and project Demonstration can be viewed in the jupyter notebook (python3) under `Notebook\TDS - Project demonstration.ipynb`


# Running options:
## Preferred option: Docker
The only prerequisite is docker.
After that, you will have the notebook running on your localhost And accessible via the link
`http://localhost:8899/` 


Building the docker image:
`sudo docker build -t tds/finalproject .`

Running the docker with:
`sudo docker run -p 8899:8899 tds/finalproject` (you can change the port forwarding as you wish)

(sudo is necessary whenever `docker` is not in the sudoers)



## Another option: Raw python3 and requierments

Do not forget installing the necessary requirments
`pip3 install -r requirments.txt`

To view and run the jupyter notebook use the following:
`pip3 install jupyter`

Use `jupyter notebook` command to launch and view the `TDS - Project demonstration.ipynb` file