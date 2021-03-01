def data_reshape(data_sample):
    print("--------------data_reshape------------",data_sample.shape)
    #out = []
    reshaped = data_sample.reshape(data_sample.shape[0],
                                   data_sample.shape[1]*data_sample.shape[2]) # ,order=A)
    #print(reshaped.shape)
    #out.append(reshaped)
    return(reshaped)

def reshape_numpy(data_list):
    print("--------------reshapes------------",len(data_list))
    out = []
    for d in data_list:
        out.append(data_reshape(d))

    return(out)
