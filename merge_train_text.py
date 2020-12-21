from utils import *


train_dict1 = load_json("all_train_text/last_version_data","train_pub.json")
train_dict2 = load_json("all_train_text/last_version_data","sna_valid_pub.json")

train_dict3 = load_json("train","train_pub.json")
train_dict4 = load_json("sna_data","sna_valid_pub.json")

# train_dict5 = load_json("all_train_text/last_version_data","sna_test_pub.json")

train_dict1.update(train_dict3)
train_dict2.update(train_dict4)
# train_dict2.update(train_dict5)

dump_json(train_dict1, 'all_train_text', 'all_train_pub.json', indent=4)
dump_json(train_dict2, 'all_train_text', 'all_valid_pub.json', indent=4)