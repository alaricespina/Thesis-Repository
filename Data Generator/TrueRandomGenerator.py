import random
import pandas as pd 
import os 
import json 


MILITARY_ALPHABET = "Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliett Kilo Lima Mike November Oscar Papa Quebec Romeo Sierra Tango Uniform Victor Whiskey X-ray Yankee Zulu".split()
NORMAL_ALPHABET = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split()
JSON_FILE_NAME = "config.json"
DATASET_FILE_NAME = "generated_dataset.csv"
SAVE_AFTER_CONFIG = True
FILE_EXIST = os.path.exists(JSON_FILE_NAME)

# Function - for checking inputs to be integer
def AskInt(question, error_message="[Invalid Input] - Expecting an Integer Input"):
    while True:
        temp_value = input(question)
        try:
            return int(temp_value)
        except ValueError:
            print(error_message)

# Function - for generating the dataset itself
def GenerateDataset(in_names, out_names, i_min, i_max, o_min, o_max, n_rows, g_type):
    td = {}
    for x in in_names: td[x] = [random.randint(i_min, i_max) for _ in range(n_rows)]
    for x in out_names: 
        if g_type == "r":
            td[x] = [random.randint(o_min, o_max) for _ in range(n_rows)]
        elif g_type == "c":
            td[x] = [random.choice(o_min) for _ in range(n_rows)]

    return td

# Check first if there is already a configuration file that exists
if FILE_EXIST:
    load_from_previous = input("Configuration File exists - Load from existing configuration file? [Y] ")

    if load_from_previous.lower() == "y":
        try:
            file = open(JSON_FILE_NAME, "r")
            data = json.load(file)
            file.close()

            td = GenerateDataset(
                data["input_class_names"],
                data["output_class_names"],
                data["min_input_val"],
                data["max_input_val"],
                data["min_output_val"],
                data["max_output_val"],
                data["num_rows"],
                data["g_type"]
                )
            td_df = pd.DataFrame(td)
            print(td_df.head())
            td_df.to_csv(DATASET_FILE_NAME)

            exit()
        except Exception:
            print("Error in loading configuration - proceeding to Manual Configuration")
    else:
        print("Proceeding to Manual Configuration")

else:
    print("No Configuration File exists (make sure it is named config.json)")
    print("Proceeding to Manual Configuration")

PROBLEM_TYPE = input("[C]lassification or [R]egression?")
NUM_OUTPUT = AskInt("How many types of Output? ")
CLASS_NAMES = []

if PROBLEM_TYPE.lower() == "c":
    NUM_CLASSES = AskInt("Classification - How many classes? ")
    GENERATE_RANDOM_CLASSES = input("Specify the Classes? [Y]")
    if GENERATE_RANDOM_CLASSES.lower() == "y":
        for i in range(NUM_CLASSES):
            CLASS_NAMES.append(input(f"[{i}] - Class Name: "))
    
    else:
        for i in range(NUM_CLASSES):
            random_name = random.choice(MILITARY_ALPHABET)
            while random_name in CLASS_NAMES:
                random_name = random.choice(MILITARY_ALPHABET)

            CLASS_NAMES.append(random_name)
        
        print("Available Class Types:", end=" ")
        print(CLASS_NAMES)

if PROBLEM_TYPE.lower() == "r":
    MAX_OUTPUT = AskInt("Maximum Value for the Output? ")
    MIN_OUTPUT = AskInt("Minimum Value for the Output? ")

    if MAX_OUTPUT < MIN_OUTPUT:
        A = MAX_OUTPUT
        MAX_OUTPUT = MIN_OUTPUT
        MIN_OUTPUT = A 

elif PROBLEM_TYPE.lower() == "c":
    MIN_OUTPUT = CLASS_NAMES
    MAX_OUTPUT = None


NUM_INPUT = AskInt("How many types of Input? ")
MIN_INPUT = AskInt("Minimum Number for Input: ")
MAX_INPUT = AskInt("Maximum Number for Input: ")
NUM_ROWS = AskInt("Number of Rows: ")


INPUT_CLASS_NAMES = [NORMAL_ALPHABET[i] for i in range(NUM_INPUT)]
OUTPUT_CLASS_NAMES = ["Output-" + NORMAL_ALPHABET[i] for i in range(NUM_OUTPUT)]

if MAX_INPUT < MIN_INPUT:
    B = MAX_INPUT
    MAX_INPUT = MIN_INPUT
    MIN_INPUT = B

temporary_dataset = GenerateDataset(INPUT_CLASS_NAMES, OUTPUT_CLASS_NAMES, MIN_INPUT, MAX_INPUT, MIN_OUTPUT, MAX_OUTPUT, NUM_ROWS, PROBLEM_TYPE)
print(temporary_dataset)

td_df = pd.DataFrame(temporary_dataset)
print(td_df.head())
td_df.to_csv(DATASET_FILE_NAME)

if SAVE_AFTER_CONFIG:
    config = {
        "num_rows"  : NUM_ROWS,
        "input_class_names" : INPUT_CLASS_NAMES,
        "output_class_names" : OUTPUT_CLASS_NAMES,
        "min_input_val" : MIN_INPUT,
        "max_input_val" : MAX_INPUT,
        "min_output_val" : MIN_OUTPUT,
        "max_output_val" : MAX_OUTPUT,
        "g_type" : PROBLEM_TYPE
    }

    out_file = open(JSON_FILE_NAME, "w")
    json.dump(config, out_file, indent=7)
    out_file.close()



