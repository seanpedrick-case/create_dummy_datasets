import pandas as pd
from faker import Faker
import datetime
import random
import numpy as np
import typo

# Defaults
faker_country = 'en_UK'
add_group = True # whether to add group key to dfs

def create_fake_df(num_people, frames, overlap_percentage, random_seed, not_common_appear_once, duplicates_percentage, noise_prob, missing_prob, entry_list, faker_country=faker_country, add_group=add_group):

    # Set random seed to predefined value
    Faker.seed(random_seed)
    random.seed(random_seed)

    # Use weighting = False for truly random names rather than commonality weighted - faster
    
    fake = Faker(faker_country, use_weighting=False) # Localise this for UK
    ### Main
    
    people_master = create_people_master(num_people, fake, add_group, overlap_percentage, duplicates_percentage, random_seed, frames, entry_list)
    people_master_df = pd.DataFrame(people_master)


    # Sample from the master range and generate
    data_frame_list = []
    data_frame_name_save_list = []

    sample_list = people_master
    #print(people_master)

    for frame_no in range(1,frames+1):
        frame_name = 'f' + str(frame_no)

        frame_number_of_people = int(num_people.iloc[0,frame_no-1])
        print(frame_number_of_people)

        frame_common = sample_from_people_master_list(sample_list, frame_number_of_people, frame_name, duplicates_percentage, is_common=True, random_seed=random_seed, noise_prob=noise_prob, missing_prob=missing_prob, entry_list=entry_list)
        frame_unique = sample_from_people_master_list(sample_list, frame_number_of_people, frame_name, duplicates_percentage, is_common=False, random_seed=random_seed, noise_prob=noise_prob, missing_prob=missing_prob, entry_list=entry_list)

        # Remove items from sample list that appeared in frame_unique
        if not_common_appear_once == "Yes":
            if frame_unique:
                groups_sampled = pd.DataFrame(frame_unique)['group'].unique()
                sample_list = [item for item in sample_list if item['group'] not in groups_sampled]

        data_frame = pd.concat([pd.DataFrame(frame_common), pd.DataFrame(frame_unique)])

        # Add duplicates to end of dataframe
        frame_duplicates = pd.DataFrame()
        if duplicates_percentage > 0:
            number_rows_duplicates = round(int(num_people.iloc[0,frame_no-1])*duplicates_percentage)
            # Sample duplicates with replacement (so same people can be duplicated multiple times)
            frame_duplicates = data_frame.sample(n=number_rows_duplicates, replace=True)
            frame_duplicates['in_duplicate_set'] = True

        data_frame_with_dups = pd.concat([data_frame, frame_duplicates]).reset_index().drop('index', axis = 1)

        data_frame_with_dups['ID'] = frame_name + "-" + (data_frame_with_dups.index+1).astype(str)

        # Update people_master_df to label records that are present in new frame
        data_frame_with_dups_j = data_frame_with_dups[['group']]
        data_frame_with_dups_j['exists_in_' + frame_name] = True
        data_frame_with_dups_j_group = data_frame_with_dups_j.groupby('group')['exists_in_' + frame_name].sum().reset_index()
        people_master_df = people_master_df.merge(data_frame_with_dups_j_group, on = 'group', how = 'left')

        # Save to csv
        file_name = frame_name + '_' + str(int(num_people.iloc[0,frame_no-1])) + '_seed_' + str(int(random_seed)) + '_overlap_' + str(overlap_percentage) + '.csv'
        data_frame_with_dups.to_csv(file_name, index = None)

        data_frame_name_save_list.append(file_name)
        data_frame_list.append(data_frame_with_dups)

    # For each master record, sum up how many dataframes they exist in
    selected_columns = [col for col in people_master_df.columns if col.startswith('exists_')]
    #people_master_df['number_dataframes_exists'] = people_master_df[selected_columns].where(people_master_df[selected_columns] > 0).sum(axis=1)
    people_master_df['number_dataframes_exists'] = (people_master_df[selected_columns] > 0).sum(axis=1)

    # Write people master to csv
    people_master_df_length = len(people_master_df)
    people_master_file_name = 'people_master_' + str(int(people_master_df_length)) + '_seed_' + str(int(random_seed)) + '_overlap_' + str(overlap_percentage) + '.csv'
    people_master_df.to_csv(people_master_file_name, index=None)

    # Add people master to list of output dfs
    data_frame_name_save_list.append(people_master_file_name)

    out_message = "Finished"

    return data_frame_name_save_list, out_message


def sample_from_people_master_list(people_master, num_people, frame_name, duplicates_percentage, is_common, random_seed, noise_prob, missing_prob, entry_list):
    people_master_df = pd.DataFrame(people_master)

    common_people = people_master_df[people_master_df['common'] == True]
    len_common_people = len(common_people)

    not_common_people = people_master_df[people_master_df['common'] == False]
    len_not_common_people = len(not_common_people)
                
    person_sample_df = people_master_df[people_master_df['common'] == is_common]
    person_sample_df_length = len(person_sample_df)

    len_duplicate_people = round(duplicates_percentage*num_people)

    if is_common == True: 
         sample_range = range(len_common_people)
         sample_size = len(sample_range)

    if is_common == False:
         sample_range =  range(len_common_people, len(people_master_df))

         sample_size = num_people - len_common_people - len_duplicate_people

    print("is_common is ",is_common, " sample size is ", sample_size, " sample range is ", sample_range)

    frame = []
    frame_idxs = random.sample(sample_range, sample_size)
    for n, idx in enumerate(frame_idxs):
        person = {'Within common subset ID': f"{frame_name}-{str(is_common)}-{n+1}"}
        person.update(people_master[idx])
        frame.append(add_noise(person, random_seed, noise_prob, missing_prob, entry_list))

    return frame

def create_people_master(num_people, fake, add_group, overlap_percentage, duplicates_percentage, random_seed, frames, entry_list):
    
    # Generate a list of fake cities that seem realistic
    cities, city_weights = create_cities_list(fake) 

    # Create all people
    
    people_master = []

    duplicate_people = 0
    total_people_across_all_frames = 0
    common_people = 0

    num_people = num_people.astype(int)
    #print(num_people)

    shortest_dataframe = num_people.min().min()
    common_people = round(overlap_percentage*shortest_dataframe)

    for i in num_people.columns:

        #print(i)
        #print(num_people[i][0])

        duplicate_people_frame = round(duplicates_percentage*num_people[i][0])
        total_people_across_all_frames_frame = (num_people[i][0] - duplicate_people)

        duplicate_people = duplicate_people + duplicate_people_frame
        total_people_across_all_frames = total_people_across_all_frames + total_people_across_all_frames_frame
    

    total_unique_people_across_all_frames = total_people_across_all_frames-common_people

    print(total_people_across_all_frames)

    for person in range(0, common_people): # Generate people using Faker repo, should be self-explanatory
        entry = create_fake_person(person, fake, cities, city_weights, entry_list, is_common = True)
        
        people_master.append(entry)

    for person in range(common_people, total_unique_people_across_all_frames): # Generate people using Faker repo, should be self-explanatory
        entry = create_fake_person(person, fake, cities, city_weights, entry_list, is_common = False)
        
        people_master.append(entry)
    
    return people_master

def add_noise(person, random_seed, noise_prob, missing_prob, entry_list):
        """Adds noise to a person entry."""
        #random.seed(random_seed)

        person_out = person.copy()

        keys = entry_list

        for key in keys: # Fields to noise
            random_no = random.random()

            if random_no < noise_prob: # x% chance of adding noise to each key. 50% chance of a completely missing value
                    
                entry = person_out[key]
                noise = typo.StrErrer(entry)

                # See noise types from the typo repo for this
                noises = [noise.char_swap,
                    noise.missing_char,
                    noise.extra_char,
                    noise.nearby_char,
                    noise.similar_char,
                    noise.skipped_space,
                    noise.random_space,
                    noise.repeated_char,
                    noise.unichar
                ]  

                noise_choice = random.choice(noises)
                noise_choice = noise_choice().result
                person_out[key] = noise_choice

            random_missing_no = random.random()
            if (random_missing_no < missing_prob):
                person_out[key] = ""

        return person_out

def create_cities_list(fake):
        
        # Create list of cities to use in people creation
        cities = []
        additional_cities = ['Cardiff', 'Birmingham', 'Manchester', 'Durham', 'London']
        additional_cities_weights = [2, 3, 3, 1, 20] # Relative weights for the cities to be added, ie. London will be 20x as likely as Durham

        for _ in range(20):
            cities.append(fake.city()) # Generate 20 (fake) cities

        # Set default weights to 1, append on custom weights for new cities
        weights = np.ones_like(cities)

        cities.extend(additional_cities) # Add some real cities for variety  
        city_weights = np.append(weights, additional_cities_weights) 
        city_weights = np.array(city_weights, dtype=int)
        
        return cities, city_weights

def create_fake_person(person, fake, cities, city_weights, entry_list, is_common):
        add_group = True

        entry_dict = {}

        if random.random() < 0.5: # Male

            if "First name" in entry_list:
                first_name = fake.first_name_male()
                entry_dict['First name'] =  first_name
            
            if "Title" in entry_list:
                title = fake.prefix_male()
                entry_dict['Title'] =  title

        else: # Female
            if "First name" in entry_list:
                first_name = fake.first_name_female()
                entry_dict['First name'] =  first_name

            if "Title" in entry_list:
                title = fake.prefix_female()
                entry_dict['Title'] =  title

        if "Last name" in entry_list:
            last_name = fake.last_name()
            entry_dict['Last name'] =  last_name

        if "Full name" in entry_list:
            if "Title" in entry_list:
                full_name = title + " " + first_name + " " + last_name

            else:
                full_name = first_name + " " + last_name
            
            entry_dict['Full name'] =  full_name

        if "Date of birth" in entry_list:
            birth_date = fake.date_between(start_date=datetime.date(1950, 1, 1)).strftime(r"%d/%m/%Y")
            entry_dict['Date of birth'] =  birth_date

        if "Address" in entry_list:
            address = fake.street_address()
            city = random.choices(cities, weights=city_weights, k=1)[0]
            address += ", " + city

            pc = fake.postcode()

            entry_dict['Address'] =  address
            entry_dict['Postcode'] =  pc

        if "Email" in entry_list:
            email = fake.ascii_free_email()
            entry_dict['Email'] =  email
            
        if "Phone number" in entry_list:
            phone = fake.phone_number()
            entry_dict['Phone number'] =  phone
                

        entry = entry_dict
        if add_group:
                entry.update({'group': person})
                entry.update({'common':is_common})
        
        return entry