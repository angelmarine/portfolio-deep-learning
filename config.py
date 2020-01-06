import os
import pprint

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print("---------------------------------------------------------------------------------------------------------------")
print("% Current Working Directory: ", CURRENT_DIR)

DIR_CONFIG = {
    "CURRENT_DIR": CURRENT_DIR,
    "PROJECT_DIR": os.path.dirname(CURRENT_DIR),
    "DATA_DIR": os.path.join(CURRENT_DIR, 'data'),
    "OLD_DATA_DIR": os.path.join(CURRENT_DIR, 'old_data'),
    "CHECKPOINT_DIR": os.path.join(CURRENT_DIR, 'checkpoints'),
    "RESULT_DIR": os.path.join(CURRENT_DIR, 'results')
}

FILE_CONFIG = {
    "COMPANY_LIST": os.path.join(DIR_CONFIG["CURRENT_DIR"], "company_list.txt"),
}

print("% Directory Configurations")
pprint.pprint(DIR_CONFIG)


print("% File Configurations")
pprint.pprint(FILE_CONFIG)