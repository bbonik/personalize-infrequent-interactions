from scipy.stats import invgamma
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

def generate_ii_dataset(
    number_of_users = 1000,
    number_of_items = 100,
    date_begin = (2020,1,1,0,0,0),
    date_end = (2022,1,1,0,0,0),
    number_of_popular_items = 5,
    max_popularity_factor = 5.0,
    random_seed = None,
    verbose = True,
    ):
    
    '''
    Function to generate a random synthetic dataset of infrequent interactions
    
    INPUTS:
    ------
    number_of_users: int
        The total number of users (user_id is an integer in the interval [0,number_of_users])
    number_of_items: int
        The total number of products to interact with (item_id is an integer in the interval [0,number_of_items])
    date_begin: tuple of int
        Oldest possible timestamp (YYYY,DD,MM,HH,MM,SS)
    date_end: tuple of int
        Most recent possible timestamp (YYYY,DD,MM,HH,MM,SS)
    number_of_popular_items: int
        Number of items which will be considered more popular
    max_popularity_factor: float
        Up to how many times each popular item will be more popular compared to the regular ones [1,max_popularity_factor]
    random_seed: None or integer
        If None, then results will be random (based on the current clock). If integer, the random seed will be fixed to this value,
        generating reproducible results. 
    verbose: bool
        Whether to display graphs and a report for the generated dataset.
        
    OUTPUTS:
    -------
    df_dataset: dataframe of the generated dataset
    df_item_popularity: dataframe of the interactions per item
    '''
    
    
    # fixing random seeds
    if type(random_seed) is int:
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    max_item_digits = len(str(number_of_items))
    max_user_digits = len(str(number_of_users)) - 1
    ALPHA = 1  # this parameter defines the skewness of the sampling distribution
    
    # starting and ending range timestamps
    timestamp_range_begin = int(
        datetime(
            date_begin[0],
            date_begin[1],
            date_begin[2],
            date_begin[3],
            date_begin[4],
            date_begin[5]
        ).timestamp()
    )
    timestamp_range_end = int(
        datetime(
            date_end[0],
            date_end[1],
            date_end[2],
            date_end[3],
            date_end[4],
            date_end[5]
        ).timestamp()
    )
        
    # define sampling distribution for interactions
    alpha = ALPHA  # this parameter defines the skewness of the sampling distribution
    x = np.linspace(invgamma.ppf(0.01, alpha), invgamma.ppf(0.99, alpha), 100)
    
    
    # defining the popularity distribution of items (popular vs other items)
    population = [i for i in range(number_of_items+1)]
    popular_items = random.sample(  # which items are popular
        population, 
        k=number_of_popular_items
    )
    popularity_factor = np.random.uniform(  # how much more popular the popular items are, compared to the rest
        low=1.0, 
        high=max_popularity_factor, 
        size=number_of_popular_items
    )
    popularity_weights = np.ones(number_of_items+1, dtype=float)
    popularity_weights[popular_items] *= popularity_factor
    
    
    # generating interactions dataset
    ls_user_ids = []
    ls_timestamps = []
    ls_item_ids = []
    for user in range(number_of_users):
        # randomly sampling how many interactions the user had (from the gamma distribution)
        number_of_interactions = np.round(
            invgamma.rvs(
                a=alpha, 
                size=1)
        ).astype(int)[0]

        if number_of_interactions > 0:
            # randomly sampling the items the user has interacted with 
            # (popular items have higher chance of being selected)
            interacted_items = random.choices(
                population, 
                popularity_weights,
                k=number_of_interactions
            )
            # randomly sampling when the interactions happened (uniform distribution)
            interaction_timestamps = np.random.randint(
                low=timestamp_range_begin, 
                high=timestamp_range_end+1, 
                size=number_of_interactions, 
                dtype='int'
            )
            ls_timestamps.extend(list(interaction_timestamps))
            ls_item_ids.extend('item_' + ('%0*d' % (max_item_digits, item)) for item in list(interacted_items))
            ls_user_ids.extend(['user_' + ('%0*d' % (max_user_digits, user)) for i in range(number_of_interactions)])
    
    
    # package dataset into a dataframe
    df_dataset = pd.DataFrame(
        list(
            zip(
                ls_timestamps, 
                ls_user_ids, 
                ls_item_ids
            )
        ), 
        columns=[
            'TIMESTAMP', 
            'USER_ID', 
            'ITEM_ID'
        ]
    )
    
    # create a dataframe of the popularity of each item
    df_item_popularity = df_dataset.groupby(['ITEM_ID']).count()
    df_item_popularity.drop(columns=['TIMESTAMP'], inplace=True)
    df_item_popularity.sort_values(by=['USER_ID'], ascending=False, inplace=True)
    df_item_popularity.rename(columns={'ITEM_ID': 'ITEM_ID', 'USER_ID': 'Interactions'}, inplace=True)
    
    
    if verbose is True:
        
        plt.style.use('seaborn')
        grid = plt.GridSpec(3, 2)
        plt.figure(figsize=(17, 15))
        plt.suptitle('Random synthetic dataset of ' + str(len(df_dataset)) + ' infrequent interactions')
        
        # plot the sampling distribution 
        plt.subplot(grid[0,0])
        plt.plot(x, invgamma.pdf(x, alpha), label='invgamma pdf')
        plt.legend()
        plt.xlabel('Number of interactions')
        plt.title('Distribution used to sample user interactions')
        
        # plot interactions per user
        plt.subplot(grid[0,1])
        df_counts = df_dataset.groupby(['USER_ID']).count()
        df_counts.drop(columns=['TIMESTAMP'], inplace=True)
        interactions = df_counts['ITEM_ID'].values
        hist = np.histogram(interactions, bins=[0,1,2,3,4,5,6,7,8,9,10,20,50, 100000])  # does not contain those with zero interactions
        hist[0][0] = 100 - (hist[0].sum()/number_of_users)*100  # adding those with zero interactions
        hist[0][1:] = hist[0][1:]*100/number_of_users  # normalizing the rest of the bins
        bars = plt.bar(range(len(hist[0])), hist[0])
        plt.xticks(ticks=range(len(hist[0])), labels=['0','1', '2','3','4','5','6','7','8','9','10-20','20-50','>50'])   
        plt.ylabel('% of users')
        plt.xlabel('Number of interactions')
        plt.title('% of users per number of interactions')
        for rect in bars:
            height = round(rect.get_height(),1)
            plt.text(rect.get_x() + rect.get_width()/2.0, height, str(height)+'%', ha='center', va='bottom')
        
        # plot number of interactions per item
        plt.subplot(grid[1,:])
        df_counts2 = df_dataset.groupby(['ITEM_ID']).count()
        df_counts2.drop(columns=['TIMESTAMP'], inplace=True)
        interactions2 = df_counts2['USER_ID'].values
        bars = plt.bar(range(len(interactions2)), interactions2)
        plt.ylabel('Interactions')
        plt.xlabel('Item id')
        plt.title('# of interactions per item')
        
        # plot number of interactions per timestamp bin
        plt.subplot(grid[2,:])
        plt.hist(df_dataset['TIMESTAMP'].values, bins=100)
        plt.ylabel('Interactions')
        plt.xlabel('Timestamp bin')
        plt.title('# of interactions per timestamp bin')
        plt.show()

        # print some report
        print('Number of unique items: ', number_of_items)
        print('Number of unique users: ', number_of_users)
        print('Number of interactions: ', len(df_dataset))
        print('Timestamp range start:', timestamp_range_begin)
        print('Timestamp range end:', timestamp_range_end)
        print(hist[0][0]+hist[0][1], '% of users have less than 2 interactions!' )
        print('Most popular items:', popular_items)
        print('By a multiplication factor of:',popularity_factor)
        
    return df_dataset, df_item_popularity
