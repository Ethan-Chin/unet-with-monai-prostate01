# %%
import numpy as np

# %%
training = np.arange(1, 40+1)
validation = np.arange(41, 46+1)
test = np.arange(47, 55+1)



# %%
def dataset_format_config():
    config = {
        "training" : training,
        "validation" : validation,
        "test" : test,
        "shuffle_id" : False,
    }
    np.save("./dataset_format_config.npy", config)
    print("Configure Saved!")

if __name__ == "__main__":
    dataset_format_config()
# %%
