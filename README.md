# personalize-infrequent-interactions
[This notebook](https://github.com/bbonik/personalize-infrequent-interactions/blob/main/personalize-infrequent-interactions.ipynb) shows how to use **Amazon Personalize** with datasets of infrequent interactions. It particularly shows how to use a combination of **SIMS+Personalize Ranking recipes**, in order to recommend new items to users, which can give better results when the user interactions with items are infrequent. 

![example](static/example.png "example")


The [notebook](https://github.com/bbonik/personalize-infrequent-interactions/blob/main/personalize-infrequent-interactions.ipynb) includes the following sections:
1. Generating a synthetic infrequent interactions dataset
2. Setting up Amazon Personalize
3. Training 4 solutions (User Personalization, Personalized Ranking, SIMS, Popular Items) and deploying 4 campaigns. 
4. Creating a **function that combines SIMS + Personalized Ranking recipes**, in order to recommend items to users.
5. Comparing and evaluating SIMS + Personalized Ranking vs the User Personalization recipe.
6. Cleaning up resources. 


The [notebook](https://github.com/bbonik/personalize-infrequent-interactions/blob/main/personalize-infrequent-interactions.ipynb) makes use of 2 helper Python (3.8) scripts, located in the ```utils``` folder.
- ```dataset_generator.py```: This script generates a random synthetic dataset of infrequent interactions.
- ```delete_resources.py```: This script deletes all the Amazon Personalize resources (e.g. campaigns, solutions, dataset groups etc.) that was generated.
