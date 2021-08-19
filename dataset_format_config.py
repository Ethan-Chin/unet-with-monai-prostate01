# %%
import numpy as np

def dataset_format_config():
    config = {
        "training_num" : 40,
        "validation_num" : 6,
        "test_num" : 9,
        "shuffle_id" : False,
        "directly_mode" : True
    }
    np.save("./dataset_format_config.npy", config)
    print("Configure Saved!")

if __name__ == "__main__":
    dataset_format_config()
# %%
