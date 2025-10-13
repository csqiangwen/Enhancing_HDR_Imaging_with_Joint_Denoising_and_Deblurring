from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # Data option
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--hdr_dararoot', type=str, default='/disk1/wenqiang/Documents/code/HDR/our_data/synthetic')

        # model option
        self.parser.add_argument('--gpu_ids', type=str, default='2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints_hdr', help='models are saved here')
        self.parser.add_argument('--name', type=str, default='HDR', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--which_iter', type=str, default=0, help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--isCustomData', action='store_true', help='whethter to use custom data')
        self.parser.add_argument('--customName', type=str, default='Custom', help='give a name to the custom dataset')

        self.isTrain = False
