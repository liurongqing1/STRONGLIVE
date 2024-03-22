from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=d))

            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        
        self.loader_test = []
        for d in args.data_test:
            if d in [
                    'LOL_000','patch_rd000', 'Set5', 'Set14', 'B100', 'Urban100','Joceky_edge','Jockey','Cal','City','Ready','Flower','Game','Man', 'Bosp', 'Bigbosp', 
                    'Bear_000', 'Bear_001', 'Bear_002', 'Bear_003', 'Bear_004', 'Bear_005', 'Bear_006', 'Bear_007', 'Bear_008', 'Bee_000', 'Bee_001', 'Bee_002', 
                    'Bee_003', 'Bee_004', 'Bee_005', 'Bee_006', 'Bee_007', 'Bee_008', 'Bee_009', 'Bee_010', 'Bee_011', 'Bee_012', 'Bee_013', 'Bee_014', 'Bee_015', 
                    'Bee_016', 'Bee_017', 'Bee_018', 'Bee_019', 'Bee_020', 'Bigbosp_000', 'Bigbosp_001', 'Bigbosp_002', 'Bigbosp_003', 'Bigbosp_004', 'Bigbosp_005', 
                    'Bigbosp_006', 'Bigbosp_007', 'Bigbosp_008', 'Bigbosp_009', 'Bigbosp_010', 'Bigbosp_011', 'Bigbosp_012', 'Bigbosp_013', 'Bigbosp_014', 'Bigbosp_015', 'Bigbosp_016', 
                    'Bigbosp_017', 'Bigbosp_018', 'Bigbosp_019', 'Bigbosp_020', 'Bigmouse_000', 'Bigmouse_001', 'Bigmouse_002', 'Bigmouse_003', 'Bigmouse_004', 'Bigmouse_005', 'Bigmouse_006', 'Bigmouse_007', 
                    'Bigmouse_008', 'Book_000', 'Book_001', 'Book_002', 'Book_003', 'Book_004', 'Book_005', 'Book_006', 'Book_007', 'Book_008', 'Book_009', 'Book_010', 'Book_011', 'Book_012', 'Book_013', 'Book_014',
                     'Book_015', 'Book_016', 'Book_017', 'Book_018', 'Bookshelf_000', 'Bookshelf_001', 'Bookshelf_002', 'Bookshelf_003', 'Bookshelf_004', 'Bookshelf_005', 'Bookshelf_006', 'Bookshelf_007', 'Bookshelf_008', 
                    'Bookshelf_009', 'Bookshelf_010', 'Bosp_000', 'Bosp_001', 'Bosp_002', 'Bosp_003', 'Bosp_004', 'Bosp_005', 'Bosp_006', 'Bosp_007', 'Bosp_008', 'Bosp_009', 'Bosp_010', 'Bosp_011', 'Bosp_012', 'Bosp_013', 
                    'Bosp_014', 'Bosp_015', 'Bosp_016', 'Bosp_017', 'Bosp_018', 'Bosp_019', 'Bosp_020', 'City_000', 'City_001', 'City_002', 'City_003', 'City_004', 'City_005', 'City_006', 'City_007', 'City_008', 'City_009', 
                    'City_010', 'City_011', 'City_012', 'City_013', 'City_014', 'dog_000', 'dog_001', 'dog_002', 'dog_003', 'dog_004', 'dog_005', 'dog_006', 'dog_007', 'dog_008', 'dog_009', 'dog_010', 'Fish_000', 'Fish_001', 
                    'Fish_002', 'Fish_003', 'Fish_004', 'Fish_005', 'Fish_006', 'Fish_007', 'Fish_008', 'Fish_009', 'Fish_010', 'Fish_011', 'Flower_000', 'Flower_001', 'Flower_002', 'Flower_003', 'Flower_004', 'Flower_005',
                     'Flower_006', 'Flower_007', 'Flower_008', 'Flower_009', 'Flower_010', 'Flower_011', 'Flower_012', 'Flower_013', 'Flower_014', 'Flower_015', 'Flower_016', 'Flower_017', 'Flower_018', 'Flower_019',
                     'Game_000', 'Game_001', 'Game_002', 'Game_003', 'Game_004', 'Game_005', 'Hand_000', 'Hand_001', 'Hand_002', 'Hand_003', 'Hand_004', 'Hand_005', 'Hand_006', 'Hand_007', 'Hand_008', 'Hand_009',
                     'Hand_010', 'Hand_011', 'Hand_012', 'Hand_013', 'Hand_014', 'Hand_015', 'Hand_016', 'Hand_017', 'Hand_018', 'Hand_019', 'Hand_020', 'Hand_021', 'Hand_022', 'Hand_023', 'Jockey_000', 'Jockey_001', 
                    'Jockey_002', 'Jockey_003', 'Jockey_004', 'Jockey_005', 'Jockey_006', 'Jockey_007', 'Jockey_008', 'Jockey_009', 'Jockey_010', 'Jockey_011', 'Jockey_012', 'Jockey_013', 'Jockey_014', 'Jockey_015', 'Jockey_016', 
                    'Jockey_017', 'Jockey_018', 'Jockey_019', 'Jockey_020', 'Man_000', 'Man_001', 'Man_002', 'Man_003', 'Man_004', 'Man_005', 'Man_006', 'Man_007', 'Man_008', 'Man_009', 'Man_010', 'Man_011', 
                    'Mouse_000', 'Mouse_001', 'Mouse_002', 'Mouse_003', 'Mouse_004', 'Mouse_005', 'Mouse_006', 'Mouse_007', 'mulan_000', 'mulan_001', 'mulan_002', 'mulan_003', 'mulan_004', 'mulan_005', 'mulan_006', 
                    'mulan_007', 'mulan_008', 'mulan_009', 'mulan_010', 'mulan_011', 'mulan_012', 'mulan_013', 'mulan_014', 'mulan_015', 'mulan_016', 'mulan_017', 'nv_000', 'nv_001', 'Phouse_000', 'Phouse_001', 
                    'Phouse_002', 'Phouse_003', 'Phouse_004', 'Phouse_005', 'Phouse_006', 'Phouse_007', 'Phouse_008', 'Phouse_009', 'Phouse_010', 'Phouse_011', 'Phouse_012', 'Phouse_013', 'Phouse_014', 'Phouse_015', 
                    'Phouse_016', 'Phouse_017', 'Phouse_018', 'Phouse_019', 'Phouse_020', 'Phouse_021', 'Phouse_022', 'Phouse_023', 'Phouse_024', 'Phouse_025', 'Phouse_026', 'Phouse_027', 'Phouse_028', 'Phouse_029', 
                    'Phouse_030', 'Phouse_031', 'Phouse_032', 'Phouse_033', 'Phouse_034', 'Phouse_035', 'Phouse_036', 'Phouse_037', 'Phouse_038', 'Phouse_039', 'Phouse_040', 'Phouse_041', 'Phouse_042', 'Phouse_043', 
                    'Phouse_044', 'Phouse_045', 'Phouse_046', 'Phouse_047', 'Phouse_048', 'Phouse_049', 'Phouse_050', 'Phouse_051', 'Phouse_052', 'Phouse_053', 'Phouse_054', 'Phouse_055', 'Phouse_056', 'Phouse_057', 
                    'Phouse_058', 'Ready_000', 'Ready_001', 'Ready_002', 'Ready_003', 'Ready_004', 'Ready_005', 'Ready_006', 'Ready_007', 'Ready_008', 'Ready_009', 'Ready_010', 'Ready_011', 'Ready_012', 'Ready_013', 'Ready_014',
                     'Ready_015', 'Ready_016', 'Ready_017', 'Ready_018', 'Ready_019', 'Ready_020', 'Sheep_000', 'Sheep_001', 'Sheep_002', 'Sheep_003', 'Sheep_004', 'Sheep_005', 'Sheep_006', 'Sheep_007', 'Sheep_008',
                     'surf_000', 'surf_001', 'surf_002', 'surf_003', 'surf_004', 'surf_005', 'surf_006', 'surf_007', 'surf_008', 'surf_009', 'surf_010', 'surf_011', 'surf_012', 'surf_013', 'Village_000', 'Village_001', 'Village_002', 
                    'Village_003', 'Village_004', 'Village_005', 'Village_006', 'Village_007', 'Village_008', 'Village_009', 'Village_010', 'Village_011', 'Village_012', 'Village_013', 'Village_014', 'Village_015', 'Woman_000',
                     'Woman_001', 'Woman_002', 'Woman_003', 'Woman_004', 'Woman_005', 'Woman_006', 'Woman_007', 'Woman_008', 'Woman_009', 'Woman_010']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d) #data.benchmark.Benchmark

            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)


            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
