import pickle
import pandas as pd
import re
import os

direct_messages_location = 'iron_march_201911/core_message_posts.csv'
forum_posts_location = 'iron_march_201911/core_search_index.csv'
hate_words_pickle_location = 'side_information/hate_words/Hate_term_Data.pickle'

pickle_file_objects = []
with open(hate_words_pickle_location, 'rb') as pickled_info:
    while True:
        try:
            pickle_file_objects.append(pickle.load(pickled_info))
        except EOFError:
            break

def get_hate_data(text):
    data = [0, 0, 0, 0, 0, 0, 0]
    text = re.split('\W+', text.lower())
    for word in words:
        if word['term'].lower() in text:
            if word['is_about_nationality']:
                data[0] = 1
            if word['is_about_ethnicity']:
                data[1] = 1
            if word['is_about_religion']:
                data[2] = 1
            if word['is_about_gender']:
                data[3] = 1
            if word['is_about_sexual_orientation']:
                data[4] = 1
            if word['is_about_disability']:
                data[5] = 1
            if word['is_about_class']:
                data[6] = 1
        if sum(data) == 7:
            break
    return data

words = pickle_file_objects[0][0]

direct_messages_dataframe = pd.read_csv(direct_messages_location)
forum_posts_dataframe = pd.read_csv(forum_posts_location)

hate_words = []

for idx, post in direct_messages_dataframe.iterrows():
    id = 'direct_messages_' + str(post['msg_id'])
    data = get_hate_data(str(post['msg_post']))
    hate_words.append(
        {
            'msg_id' : id,
            'data' : data
        }
    )

for idx, post in forum_posts_dataframe.iterrows():
    id = 'forum_posts_' + str(post['index_id'])
    data = get_hate_data(str(post['index_content']))
    hate_words.append(
        {
            'msg_id' : id,
            'data' : data
        }
    )

pd.DataFrame(hate_words).to_csv('side_information/hate_words/processed_side_information.csv', index = False)




