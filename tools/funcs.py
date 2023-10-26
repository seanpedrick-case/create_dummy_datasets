import pandas as pd
from faker import Faker
import datetime
import random
import numpy as np
import typo

faker_country = 'en_UK'
add_group = True # whether to add group for ground truth
overwrite_ground_truth = True # overwrite existing ground Truth file

def create_fake_df(num_people, frames, overlap_percentage, random_seed, faker_country=faker_country, add_group=add_group, overwrite_ground_truth=overwrite_ground_truth):

    # Set random seed to predefined value
    Faker.seed(random_seed)
    random.seed(random_seed)

    # Use weighting = False for truly random names rather than commonality weighted - faster
    
    fake = Faker(faker_country, use_weighting=False) # Localise this for UK
    ### Main
    
    people = create_people(num_people, fake, add_group, overwrite_ground_truth, overlap_percentage, random_seed, frames)

    # Otherwise sample from the master range and generate

    data_frame_list = []
    data_frame_name_save_list = []

    for frame_no in range(1,frames+1):
        frame_name = 'f' + str(frame_no)

        frame_common = sample_from_people_list(people, num_people, frame_name, is_common=True, random_seed=random_seed)
        frame_unique = sample_from_people_list(people, num_people, frame_name, is_common=False, random_seed=random_seed)

        data_frame = pd.concat([pd.DataFrame(frame_common), pd.DataFrame(frame_unique)])

        file_name = frame_name + '_' + str(int(num_people)) + '_seed_' + str(int(random_seed)) + '_overlap_' + str(overlap_percentage) + '.csv'

        data_frame.to_csv(file_name, index = None)

        data_frame_name_save_list.append(file_name)

        data_frame_list.append(data_frame)

    out_message = "Finished"

    return data_frame_name_save_list, out_message

def create_people(num_people, fake, add_group, overwrite_ground_truth, overlap_percentage, random_seed, frames):
    
    # Generate a list of fake cities that seem realistic
    cities, city_weights = create_cities_list(fake) 

    # Create all people
    
    people = []

    common_people = round(overlap_percentage*num_people)

    for person in range(0, common_people): # Generate people using Faker repo, should be self-explanatory
        entry = create_fake_person(person, fake, cities, city_weights, is_common = True)
        
        people.append(entry)

    for person in range(common_people, num_people*frames): # Generate people using Faker repo, should be self-explanatory
        entry = create_fake_person(person, fake, cities, city_weights, is_common = False)
        
        people.append(entry)

    # Generate ground Truth
    people_df = pd.DataFrame(people)
    people_df_length = len(people_df)
    if overwrite_ground_truth:
        file_name = 'people_' + str(int(people_df_length)) + '_seed_' + str(int(random_seed)) + '_overlap_' + str(overlap_percentage) + '.csv'
        people_df.to_csv(file_name, index=None)

    return people

def add_noise(person, random_seed):
        """Adds noise to a person entry
        Note that this is not super realistic. We also don't model missing data anywhere near as much as we should. But it's a start."""
        random.seed(random_seed)

        person = person.copy()

        for key in ['First Name',
                    'Surname',
                    'Dob',
                    'Address',
                    'Postcode']: # Fields to noise
            if random.random() < 0.2: # 20% chance of adding noise to each key

                entry = person[key]
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

                if random.random() < 0.2: # Additional 20% chance of an additional noise
                    noise = typo.StrErrer(noise_choice)

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

                person[key] = noise_choice
        return person

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

def create_fake_person(person, fake, cities, city_weights, is_common):
        add_group = True

        if random.random() < 0.5: # Male
            first_name = fake.first_name_male()
            title = fake.prefix_male()
        else: # Female
            first_name = fake.first_name_female()
            title = fake.prefix_female()

        last_name = fake.last_name()

        if random.random() < 0.1:
            first_name = title + " " + first_name

        birth_date = fake.date_between(start_date=datetime.date(1950, 1, 1)).strftime(r"%d/%m/%Y")
        address = fake.street_address()
        pc = fake.postcode()
        city = random.choices(cities, weights=city_weights, k=1)[0]

        address += "\n" + city

        entry = {'First Name': first_name,
                'Surname': last_name,
                'Dob': birth_date,
                'Address': address,
                'Postcode': pc}
        if add_group:
                entry.update({'group': person})
                entry.update({'common':is_common})
        
        return entry

def sample_from_people_list(people, num_people, frame_name, is_common, random_seed):
            
    people_df = pd.DataFrame(people)

    common_people = people_df[people_df['common'] == True]
    len_common_people = len(common_people)

    not_common_people = people_df[people_df['common'] == False]
    len_not_common_people = len(not_common_people)
                
    person_sample_df = people_df[people_df['common'] == is_common]
    person_sample_df_length = len(person_sample_df)

    if is_common == True: 
         sample_range = range(len_common_people)
         sample_size = len(sample_range)
    if is_common == False:
         sample_range =  range(len_common_people, len(people_df))
         sample_size = len(sample_range)

    print("is_common is ",is_common, " sample size is ", sample_size, " sample range is ", sample_range)

    frame = []
    frame_idxs = random.sample(sample_range, sample_size)
    for n, idx in enumerate(frame_idxs):
        person = {'ID': f"{frame_name}-{n}"}
        person.update(people[idx])
        frame.append(add_noise(person, random_seed))

    return frame