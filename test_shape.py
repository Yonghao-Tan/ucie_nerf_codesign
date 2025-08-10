import torch

def reshape(x_in, block_length=5):  # block_size*cols, n_samples, n_views, n_feat
    # print(x_in.shape)
    x = x_in.reshape(block_length, -1, *x_in.shape[1:]) # block_size, cols, n_samples, n_views, n_feat
    x = x.reshape(block_length, -1, block_length, *x_in.shape[1:]) # block_size, n_col_blocks, block_size, n_samples, n_views, n_feat
    x = x.permute(1, 0, 2, 3, 4, 5) # n_col_blocks, block_size, block_size, n_samples, n_views, n_feat
    x = x.reshape(-1, block_length*block_length, *x_in.shape[1:]) # n_col_blocks, block_size*block_size, n_samples, n_views, n_feat
    return x

test = torch.tensor([[1,2,3,4,5,101,102,103,104,105,201,202,203,204,205],
                     [6,7,8,9,10,106,107,108,109,110,206,207,208,209,210],
                     [11,12,13,14,15,111,112,113,114,115,211,212,213,214,215],
                     [16,17,18,19,20,116,117,118,119,120,216,217,218,219,220],
                     [21,22,23,24,25,121,122,123,124,125,221,222,223,224,225]], dtype=torch.float32)
test = test.reshape(-1)
print(test.shape)
print(test)
test = test.reshape(-1, 1, 1, 1)  # Reshape to (n_samples, n_views, n_feat)
out = reshape(test, block_length=5)
print(out)