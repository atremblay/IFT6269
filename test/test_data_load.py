from unittest import TestCase

from src.dataload import dataload


class TestDataLoad(TestCase):

    def test_loading(self):
        data_sets = dataload.DatasetLib('/home/data/')
        print('Available Datasets:' + str(list(data_sets.datasets.keys())))
        for k, d in data_sets.datasets.items():

            for mode in ('Train', 'Test'):
                d.mode = mode

                if k == "nyuv2":
                    for task in ('regression', 'classification'):
                        d.task = task
                        print("dataset: %s, mode: %s, task: %s" % (k, mode, task))
                        self.assertRaises(Exception, d.load())
                        d.unload()
                else:
                    print("dataset: %s, mode: %s" % (k, mode))
                    self.assertRaises(Exception, d.load())
                    d.unload()






