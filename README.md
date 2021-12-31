#Medical Sessions- Lets Analyse Patient Report data

Docker application to serve NLP insights of programme feedback for medical patients of an APP, including NER, Keyphrase extraction, theme detection and sentiment analysis
Initially appeared on
[gist](https://github.com/cm2435)

## Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes. This project has been 
tested on a localhost

First things first, clone the project to localhost and navigate to your commandline 
    git clone https://github.com/cm2435/Medical-Session-Inferance-App/edit/main/README.md

Check that docker desktop daemon is running, navigate to command line and run
    docker-comopse up -d --build 

This should start the program. Wait for a while, the backend transformers download API is slow. This needs a refactor

Go to 
    http://localhost:8501/
And enjoy running!
   

### Prerequisites

Docker, for windows WSL and other Daemon requiremnts.
Python >=3.6
Check backend and frontend requirements.txt for their respective dependancies. 

### Installing

TODO- Deploy to pypi so that App can be downloaded

    python -m pip install Medical-Session-Inferance


## Running the tests

TODO- Write the automated tests for check cases using numpy arrays other than examples
TODO- Impliment pandas dataframes to fit and predict models

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct, and the process for submitting pull requests to us.

TODO- Impliment contributing.md 

## Authors

  - **Charlie Masters ** - *Currently Sole contributer* -
    [LinkedIn](https://www.linkedin.com/in/charlie-masters-a55269166/)

## Acknowledgments

Please feel free to fork and contribute to appear down Here! 
