#!/usr/bin/env python

import numpy as np

###
### Variables to modify
###
save_file_name = "/sps/t2k/cehrhardt/Caverns/index_lists/UnifVtx_electron_HK_20MeV_train_val_1.5M.npz"
nb_events = 20
indexs = [0, 15, 17,  nb_events] # [train_first_index, val_first_index, test_first_index, num_events=test_last_index-1]


###
### No modifications needed below
###
rng = np.random.default_rng()
a = rng.permutation(range(nb_events))

res = [a[indexs[i]:indexs[i+1]] for i in range(len(indexs) - 1)]

if len(res) == 1:
    np.savez(save_file_name, train_idxs=res[0])
elif len(res) == 2:
    np.savez(save_file_name, train_idxs=res[0], val_idxs=res[1])
else:
    np.savez(save_file_name, train_idxs=res[0], val_idxs=res[1], test_idxs=res[2])

print("Done")
print(f"Index list saved at : {save_file_name}")