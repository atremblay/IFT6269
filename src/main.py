from dataload import dataload


if __name__ == '__main__':

    # Data Path
    data_sets = dataload.DatasetLib('/home/data/')
    print('Available Datasets:' + str(list(data_sets.datasets.keys())))

    d = data_sets['make3d'] #nyuv2, camvid, or make3d
    d.mode = 'Test'  # Train or Test
    # d.task = 'regression' # regression or classification (for nyuv2 only)
    d.load()
    d.unload()




